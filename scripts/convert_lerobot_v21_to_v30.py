#!/usr/bin/env python3
"""Convert one or more local LeRobot v2.1 datasets to LeRobot v3.0.

The source directories are never modified.  By default, each converted dataset
is written next to its source with a ``_v3.0`` suffix.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


V21_VERSION = "v2.1"
TARGET_VERSION = "v3.0"
TARGET_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
TARGET_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=4)
        file.write("\n")


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        else:
            result[name] = item
    return result


def _aggregate_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate v2.1 per-episode statistics with weighted moments."""
    keys = {key for item in items for key in item}
    result: dict[str, Any] = {}
    for feature in sorted(keys):
        feature_stats = [item[feature] for item in items if feature in item]
        counts = np.stack([np.asarray(item["count"], dtype=np.float64) for item in feature_stats])
        means = np.stack([np.asarray(item["mean"], dtype=np.float64) for item in feature_stats])
        variances = np.stack(
            [np.square(np.asarray(item["std"], dtype=np.float64)) for item in feature_stats]
        )
        total_count = counts.sum(axis=0)
        weights = counts
        while weights.ndim < means.ndim:
            weights = np.expand_dims(weights, axis=-1)
        mean = (means * weights).sum(axis=0) / total_count
        variance = ((variances + np.square(means - mean)) * weights).sum(axis=0) / total_count
        result[feature] = {
            "min": np.min(np.stack([np.asarray(item["min"]) for item in feature_stats]), axis=0).tolist(),
            "max": np.max(np.stack([np.asarray(item["max"]) for item in feature_stats]), axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": total_count.astype(np.int64).tolist(),
        }
    return result


def _v21_path(info: dict[str, Any], episode_index: int, *, video_key: str | None = None) -> Path:
    values = {
        "episode_chunk": episode_index // int(info["chunks_size"]),
        "episode_index": episode_index,
    }
    if video_key is None:
        return Path(info["data_path"].format(**values))
    values["video_key"] = video_key
    return Path(info["video_path"].format(**values))


def _groups_by_size(paths: list[Path], maximum_bytes: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_size = 0
    for index, path in enumerate(paths):
        size = path.stat().st_size
        if current and current_size + size > maximum_bytes:
            groups.append(current)
            current = []
            current_size = 0
        current.append(index)
        current_size += size
    if current:
        groups.append(current)
    return groups


def _shard_path(root: Path, template: str, file_index: int, files_per_chunk: int, **kwargs: Any) -> Path:
    return root / template.format(
        chunk_index=file_index // files_per_chunk,
        file_index=file_index % files_per_chunk,
        **kwargs,
    )


def _merge_parquet(paths: Iterable[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    rows = 0
    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            if schema is None:
                schema = parquet.schema_arrow
                writer = pq.ParquetWriter(output, schema)
            elif not parquet.schema_arrow.equals(schema, check_metadata=False):
                raise ValueError(f"Parquet schema mismatch in {path}")
            assert writer is not None and schema is not None
            for row_group in range(parquet.num_row_groups):
                table = parquet.read_row_group(row_group)
                table = table.replace_schema_metadata(schema.metadata)
                writer.write_table(table)
                rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows


def _quote_concat_path(path: Path) -> str:
    # ffmpeg concat files use single-quoted strings; an embedded quote is escaped
    # using the same close/escape/reopen convention as a POSIX shell string.
    return "'" + str(path.resolve()).replace("'", "'\\''") + "'"


def _merge_videos(paths: list[Path], output: Path, ffmpeg: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as listing:
        for path in paths:
            listing.write(f"file {_quote_concat_path(path)}\n")
        listing.flush()
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            listing.name,
            "-map",
            "0:v:0",
            "-an",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg failed while creating {output}:\n{completed.stderr.strip()}"
        )


def _count_video_packets(path: Path, ffprobe: str) -> int:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"ffprobe failed for {path}:\n{completed.stderr.strip()}")
    try:
        return int(completed.stdout.strip())
    except ValueError as error:
        raise RuntimeError(f"ffprobe did not return a packet count for {path}") from error


def _write_episode_metadata(rows: list[dict[str, Any]], output: Path, rows_per_file: int) -> None:
    for file_index, start in enumerate(range(0, len(rows), rows_per_file)):
        path = _shard_path(
            output,
            "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            file_index,
            1000,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows[start : start + rows_per_file]), path)


def _validate_source(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info_path = source / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset (missing {info_path})")
    info = _load_json(info_path)
    if info.get("codebase_version") != V21_VERSION:
        raise ValueError(
            f"Expected LeRobot {V21_VERSION} at {source}, got {info.get('codebase_version')!r}"
        )
    episodes = sorted(_load_jsonl(source / "meta/episodes.jsonl"), key=lambda item: item["episode_index"])
    expected_indices = list(range(len(episodes)))
    actual_indices = [int(item["episode_index"]) for item in episodes]
    if actual_indices != expected_indices:
        raise ValueError(f"Episode indices must be contiguous from zero, got {actual_indices[:10]}")
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError("meta/info.json total_episodes does not match meta/episodes.jsonl")
    return info, episodes


def convert_dataset(
    source: Path,
    output: Path,
    *,
    data_file_size_mb: int = 100,
    video_file_size_mb: int = 500,
    files_per_chunk: int = 1000,
    episode_metadata_rows: int = 1000,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> Path:
    """Convert ``source`` into ``output`` and return the completed output path."""
    source = source.resolve()
    output = output.resolve()
    info, episodes = _validate_source(source)
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite it: {output}")
    if output == source or source in output.parents:
        raise ValueError("Output must not be the source directory or a directory inside it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))

    try:
        data_paths = [source / _v21_path(info, int(episode["episode_index"])) for episode in episodes]
        missing = [path for path in data_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing v2.1 data file: {missing[0]}")

        output_episodes = [dict(episode) for episode in episodes]
        data_offset = 0
        data_groups = _groups_by_size(data_paths, data_file_size_mb * 1024 * 1024)
        for file_index, group in enumerate(data_groups):
            target = _shard_path(temporary, TARGET_DATA_PATH, file_index, files_per_chunk)
            rows_written = _merge_parquet((data_paths[index] for index in group), target)
            expected_rows = sum(int(episodes[index]["length"]) for index in group)
            if rows_written != expected_rows:
                raise ValueError(
                    f"Data row mismatch in {target}: expected {expected_rows}, wrote {rows_written}"
                )
            for index in group:
                length = int(episodes[index]["length"])
                output_episodes[index].update(
                    {
                        "data/chunk_index": file_index // files_per_chunk,
                        "data/file_index": file_index % files_per_chunk,
                        "dataset_from_index": data_offset,
                        "dataset_to_index": data_offset + length,
                    }
                )
                data_offset += length
        if data_offset != int(info["total_frames"]):
            raise ValueError(
                f"Frame count mismatch: info.json says {info['total_frames']}, converted {data_offset}"
            )

        video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
        fps = float(info["fps"])
        for video_key in video_keys:
            video_paths = [
                source / _v21_path(info, int(episode["episode_index"]), video_key=video_key)
                for episode in episodes
            ]
            missing = [path for path in video_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing v2.1 video file: {missing[0]}")
            for file_index, group in enumerate(
                _groups_by_size(video_paths, video_file_size_mb * 1024 * 1024)
            ):
                target = _shard_path(
                    temporary, TARGET_VIDEO_PATH, file_index, files_per_chunk, video_key=video_key
                )
                _merge_videos([video_paths[index] for index in group], target, ffmpeg)
                expected_packets = sum(int(episodes[index]["length"]) for index in group)
                actual_packets = _count_video_packets(target, ffprobe)
                if actual_packets != expected_packets:
                    raise ValueError(
                        f"Video frame/packet mismatch in {target}: "
                        f"expected {expected_packets}, found {actual_packets}"
                    )
                frame_offset = 0
                for index in group:
                    length = int(episodes[index]["length"])
                    prefix = f"videos/{video_key}"
                    output_episodes[index].update(
                        {
                            f"{prefix}/chunk_index": file_index // files_per_chunk,
                            f"{prefix}/file_index": file_index % files_per_chunk,
                            f"{prefix}/from_timestamp": frame_offset / fps,
                            f"{prefix}/to_timestamp": (frame_offset + length) / fps,
                        }
                    )
                    frame_offset += length

        tasks = sorted(_load_jsonl(source / "meta/tasks.jsonl"), key=lambda item: item["task_index"])
        tasks_frame = pd.DataFrame(tasks).set_index("task")
        tasks_frame.index.name = "task"
        tasks_path = temporary / "meta/tasks.parquet"
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_frame.to_parquet(tasks_path)

        stats_items = _load_jsonl(source / "meta/episodes_stats.jsonl")
        if len(stats_items) != len(episodes):
            raise ValueError("meta/episodes_stats.jsonl does not contain one row per episode")
        stats_by_episode = {int(item["episode_index"]): item["stats"] for item in stats_items}
        _write_json(temporary / "meta/stats.json", _aggregate_stats(list(stats_by_episode.values())))
        for row in output_episodes:
            episode_stats = stats_by_episode[int(row["episode_index"])]
            row.update(_flatten({"stats": episode_stats}))
        _write_episode_metadata(output_episodes, temporary, episode_metadata_rows)

        output_info = dict(info)
        output_info.update(
            {
                "codebase_version": TARGET_VERSION,
                "chunks_size": files_per_chunk,
                "data_path": TARGET_DATA_PATH,
                "video_path": TARGET_VIDEO_PATH if video_keys else None,
                "data_files_size_in_mb": data_file_size_mb,
                "video_files_size_in_mb": video_file_size_mb,
            }
        )
        output_info.pop("total_chunks", None)
        _write_json(temporary / "meta/info.json", output_info)

        annotations = source / "annotations"
        if annotations.is_dir():
            shutil.copytree(annotations, temporary / "annotations")
        for filename in ("README.md", "LICENSE"):
            if (source / filename).is_file():
                shutil.copy2(source / filename, temporary / filename)

        os.replace(temporary, output)
        return output
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="LeRobot v2.1 dataset root(s)")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Place all outputs below this directory instead of beside each source",
    )
    parser.add_argument("--suffix", default="_v3.0", help="Suffix appended to each source directory name")
    parser.add_argument(
        "--strip-source-suffix",
        default="",
        help="Remove this suffix from each source directory name before appending --suffix",
    )
    parser.add_argument("--data-file-size-mb", type=_positive_int, default=100)
    parser.add_argument("--video-file-size-mb", type=_positive_int, default=500)
    parser.add_argument("--files-per-chunk", type=_positive_int, default=1000)
    parser.add_argument("--episode-metadata-rows", type=_positive_int, default=1000)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for source in args.sources:
        parent = args.output_root if args.output_root is not None else source.parent
        output_name = source.name
        if args.strip_source_suffix:
            if not output_name.endswith(args.strip_source_suffix):
                raise ValueError(
                    f"Source name {output_name!r} does not end with "
                    f"--strip-source-suffix={args.strip_source_suffix!r}"
                )
            output_name = output_name[: -len(args.strip_source_suffix)]
        output = parent / f"{output_name}{args.suffix}"
        print(f"Converting {source} -> {output}", flush=True)
        converted = convert_dataset(
            source,
            output,
            data_file_size_mb=args.data_file_size_mb,
            video_file_size_mb=args.video_file_size_mb,
            files_per_chunk=args.files_per_chunk,
            episode_metadata_rows=args.episode_metadata_rows,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
        )
        print(f"Completed: {converted}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

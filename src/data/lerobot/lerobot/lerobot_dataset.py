import contextlib
import copy
from bisect import bisect_right
from functools import partial
import logging
from pathlib import Path
from typing import Callable

import datasets
import torch
import torch.utils
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

from ..constants import HF_LEROBOT_HOME
from ..resources import shared_resource
from .datasets.compute_stats import aggregate_stats
from .datasets.utils import (
    check_delta_timestamps,
    create_lerobot_dataset_card,
    get_delta_indices,
    get_episode_data_index,
    get_hf_features_from_features,
    hf_transform_to_torch,
    load_episodes,
    load_info,
    load_stats,
    load_tasks,
)
from .datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    get_safe_default_codec,
)

CODEBASE_VERSION = "v3.0"


class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self._local_only = root is not None
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if self._local_only:
                raise
            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        version = self.info.get("codebase_version")
        if version != CODEBASE_VERSION:
            raise ValueError(
                f"Unsupported LeRobot dataset version {version!r} at {self.root}; "
                f"expected {CODEBASE_VERSION}."
            )

        if self.total_tasks > 0:
            self.tasks, self.task_to_task_index = load_tasks(self.root)
        else:
            self.tasks, self.task_to_task_index = {}, {}
        self.episodes = load_episodes(self.root)
        self.stats = load_stats(self.root)
        for episode_index, episode in self.episodes.items():
            for video_key in self.video_keys:
                required_video_fields = (
                    f"videos/{video_key}/chunk_index",
                    f"videos/{video_key}/file_index",
                    f"videos/{video_key}/from_timestamp",
                    f"videos/{video_key}/to_timestamp",
                )
                missing = [field for field in required_video_fields if episode.get(field) is None]
                if missing:
                    raise ValueError(
                        f"LeRobot v3 episode {episode_index} is missing video metadata: {missing}"
                    )

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def get_data_file_path(self, ep_index: int) -> Path:
        episode = self.episodes[ep_index]
        return Path(
            self.data_path.format(
                chunk_index=int(episode["data/chunk_index"]),
                file_index=int(episode["data/file_index"]),
            )
        )

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        episode = self.episodes[ep_index]
        return Path(
            self.video_path.format(
                video_key=vid_key,
                chunk_index=int(episode[f"videos/{vid_key}/chunk_index"]),
                file_index=int(episode[f"videos/{vid_key}/file_index"]),
            )
        )

    @property
    def data_path(self) -> str:
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        return self.info["chunks_size"]

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        image_output_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.repo_id = repo_id
        self._local_only = root is not None
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.image_output_dtype = image_output_dtype
        self._force_cache_sync = force_cache_sync
        self.delta_indices = None
        self.during_training = True

        self.root.mkdir(exist_ok=True, parents=True)

        self.meta = shared_resource(
            ("metadata", str(self.root.resolve()), self.revision),
            lambda: LeRobotDatasetMetadata(self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync),
            refresh=force_cache_sync,
        )
        self._configure_row_format()
        self._selected_episode_ids = (
            list(self.episodes) if self.episodes is not None else list(self.meta.episodes)
        )
        missing_episodes = set(self._selected_episode_ids).difference(self.meta.episodes)
        if missing_episodes:
            raise ValueError(
                f"Requested episodes are absent from {self.root}: {sorted(missing_episodes)}"
            )

        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            if self._local_only:
                raise
            self.revision = CODEBASE_VERSION
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self._selected_episode_ids
        )
        self._episode_ends = self.episode_data_index["to"].tolist()

        if self.delta_timestamps is not None:
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        if not hub_api.file_exists(self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            version_tag = self.meta.info["codebase_version"]
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=version_tag, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=version_tag, revision=branch, repo_type="dataset")

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()

        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self._selected_episode_ids
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return list(dict.fromkeys(fpaths))

    def load_hf_dataset(self) -> datasets.Dataset:
        files = sorted(str(path) for path in (self.root / "data").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No LeRobot v3 data parquet files found under {self.root / 'data'}")
        hf_dataset = shared_resource(
            ("frames", str(self.root.resolve()), self.revision),
            lambda: load_dataset("parquet", data_files=files, split="train"),
            refresh=self._force_cache_sync,
        )
        if len(hf_dataset) != self.meta.total_frames:
            raise ValueError(
                f"LeRobot v3 data frame count mismatch at {self.root}: "
                f"metadata={self.meta.total_frames}, parquet={len(hf_dataset)}"
            )

        # Arrow tables are immutable; only formatting state needs a private wrapper.
        formatted = copy.copy(hf_dataset)
        if self.meta.image_keys:
            formatted.set_transform(partial(hf_transform_to_torch, image_output_dtype=self.image_output_dtype))
        else:
            formatted.set_format("torch")
        return formatted

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        if self.hf_dataset is None:
            return sum(self.meta.episodes[ep_idx]["length"] for ep_idx in self._selected_episode_ids)
        return self._episode_ends[-1] if self._episode_ends else 0

    @property
    def num_episodes(self) -> int:
        return len(self._selected_episode_ids)

    def _get_local_episode_position(self, idx: int) -> int:
        if idx < 0:
            idx += self.num_frames
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.num_frames}")
        return bisect_right(self._episode_ends, idx)

    def _to_storage_index(self, idx: int) -> int:
        episode_position = self._get_local_episode_position(idx)
        local_start = self.episode_data_index["from"][episode_position].item()
        episode_id = self._selected_episode_ids[episode_position]
        storage_start = self.meta.episodes[episode_id]["dataset_from_index"]
        return int(storage_start + idx - local_start)

    def _to_storage_indices(self, indices: list[int]) -> list[int]:
        return [self._to_storage_index(index) for index in indices]

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        ep_start = int(self.episode_data_index["from"][ep_idx])
        ep_end = int(self.episode_data_index["to"][ep_idx])
        windows = {}
        query_indices, padding = {}, {}
        for key, delta_indices in self.delta_indices.items():
            window = tuple(delta_indices)
            if window not in windows:
                positions = [idx + delta for delta in window]
                windows[window] = (
                    [max(ep_start, min(ep_end - 1, position)) for position in positions],
                    torch.tensor([position < ep_start or position >= ep_end for position in positions], dtype=torch.bool),
                )
            query_indices[key], padding[f"{key}_is_pad"] = windows[window]
        return query_indices, padding

    @staticmethod
    def _tensor_column(values: torch.Tensor) -> torch.Tensor:
        return values

    @staticmethod
    def _list_indices(indices: list[int]) -> list[int]:
        return indices

    @staticmethod
    def _take_tensor_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return values[indices]

    @staticmethod
    def _take_list_rows(values: list[torch.Tensor], indices: list[int]) -> torch.Tensor:
        return torch.stack([values[index] for index in indices])

    def _configure_row_format(self) -> None:
        if self.meta.image_keys:
            self._as_tensor = torch.stack
            self._gather_indices = self._list_indices
            self._take_rows = self._take_list_rows
        else:
            self._as_tensor = self._tensor_column
            self._gather_indices = partial(torch.tensor, dtype=torch.long)
            self._take_rows = self._take_tensor_rows

    def _get_query_timestamps(self, current_ts, query_indices=None):
        result, fetched = {}, {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                indices = tuple(query_indices[key])
                if indices not in fetched:
                    batch = self.hf_dataset[self._to_storage_indices(indices)]
                    fetched[indices] = self._as_tensor(batch["timestamp"]).tolist()
                result[key] = fetched[indices]
            else:
                result[key] = [current_ts]
        return result

    def _query_hf_dataset(self, query_indices):
        return self._query_hf_dataset_fast(query_indices)

    def _query_hf_dataset_fast(self, query_indices):
        result, fetched = {}, {}
        for key, indices in query_indices.items():
            if key in self.meta.video_keys or ("images" in key and not self.during_training):
                continue
            window = tuple(indices)
            if window not in fetched:
                fetched[window] = self.hf_dataset[self._to_storage_indices(window)]
            result[key] = self._as_tensor(fetched[window][key])
        return result

    def _read_sample_rows(self, idx, query_indices):
        # Read the union once; gather each window back in its original order,
        # including repeated boundary frames used for padding.
        logical = list(dict.fromkeys([idx] + [i for rows in query_indices.values() for i in rows]))
        positions = {index: position for position, index in enumerate(logical)}
        # Every window is clamped to the same episode, so one storage offset suffices.
        storage_offset = self._to_storage_index(idx) - idx
        batch = self.hf_dataset[[index + storage_offset for index in logical]]
        row = {key: values[positions[idx]] for key, values in batch.items()}
        result, timestamps, gathers, timestamp_windows = {}, {}, {}, {}
        for key, indices in query_indices.items():
            window = tuple(indices)
            if window not in gathers:
                gathers[window] = self._gather_indices([positions[i] for i in indices])
            if key in self.meta.video_keys:
                if window not in timestamp_windows:
                    timestamp_windows[window] = self._take_rows(batch["timestamp"], gathers[window]).tolist()
                timestamps[key] = timestamp_windows[window]
            elif not ("images" in key and not self.during_training):
                result[key] = self._take_rows(batch[key], gathers[window])
        return row, result, timestamps

    def get_episode_data(self, episode_id: int) -> dict:
        ep_start = self.episode_data_index["from"][episode_id].item()
        ep_end = self.episode_data_index["to"][episode_id].item()
        q_idx = self._to_storage_indices(list(range(ep_start, ep_end)))
        selected_data = self.hf_dataset[q_idx]
        res_keys = self.meta.features.keys() - set(self.meta.video_keys)
        res = {key: self._as_tensor(selected_data[key]) for key in res_keys}
        return res

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            from_timestamp = self.meta.episodes[ep_idx].get(
                f"videos/{vid_key}/from_timestamp"
            )
            if from_timestamp is None:
                raise ValueError(
                    f"LeRobot v3 episode {ep_idx} is missing the video timestamp offset for '{vid_key}'"
                )
            query_ts = [float(from_timestamp) + timestamp for timestamp in query_ts]
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend, self.image_output_dtype)
            item[vid_key] = frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        if idx < 0:
            idx += self.num_frames
        episode_position = self._get_local_episode_position(idx)
        query_indices, query_timestamps = {}, None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, episode_position)
        else:
            padding = {}
        item, query_result, query_timestamps = self._read_sample_rows(idx, query_indices)
        ep_idx = item["episode_index"].item()
        expected_ep_idx = self._selected_episode_ids[episode_position]
        if ep_idx != expected_ep_idx:
            raise ValueError(f"LeRobot episode metadata/data mismatch at logical index {idx}: expected episode {expected_ep_idx}, got {ep_idx}")
        item = {**item, **padding, **query_result}
        if self.meta.video_keys and self.during_training:
            current_ts = item["timestamp"].item()
            for key in self.meta.video_keys:
                query_timestamps.setdefault(key, [current_ts])
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        if "task_index" in item:
            task_idx = int(item["task_index"].item())
            if task_idx not in self.meta.tasks:
                raise ValueError(f"Unknown task_index {task_idx} in episode {ep_idx}")
            item["task"] = self.meta.tasks[task_idx]
        else:
            episode_tasks = self.meta.episodes[ep_idx].get("tasks") or []
            if len(episode_tasks) != 1:
                raise ValueError(
                    f"Episode {ep_idx} has no per-frame task_index and does not declare exactly one task"
                )
            item["task"] = str(episode_tasks[0])
        if "coarse_task_index" in item:
            coarse_task_index = item["coarse_task_index"].item()
            item["coarse_task"] = self.meta.tasks[coarse_task_index]

        if "operating_hand_index" in item:
            operating_hand_index = item["operating_hand_index"].item()
            item["operating_hand"] = self.meta.tasks[operating_hand_index]
        
        if "subtask_annotation" in item and hasattr(self.meta, "annotations"):
            index = item["subtask_annotation"][0].item()
            item["subtask"] = self.meta.annotations["subtask"][index]
        
        if "atomic_task_index" in item and item["atomic_task_index"] is not None:
            atomic_task_index = item["atomic_task_index"].item()
            item["subtask"] = self.meta.tasks[int(atomic_task_index)]
            
        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

class MultiLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs: list[str],
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        image_output_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dataset_dirs = dataset_dirs
        ds_roots = [Path(ds_dir) for ds_dir in dataset_dirs]
        ds_names = [ds_dir for ds_dir in dataset_dirs]
        self.ds_names = ds_names
        self.ds_roots = ds_roots
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(ds_names, 0.0001)
        self._datasets = []
        for ds_root, ds_name in zip(ds_roots, ds_names, strict=True):
            _dataset = LeRobotDataset(
                ds_name,
                root=ds_root,
                episodes=episodes[ds_name] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[ds_name],
                download_videos=download_videos,
                video_backend=video_backend,
                image_output_dtype=image_output_dtype,
            )
            self._datasets.append(_dataset)

        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for ds_name, ds in zip(self.ds_names, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logging.warning(
                    f"keys {extra_keys} of {ds_name} were disabled as they are not contained in all the "
                    "other datasets."
                )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])

    def set_during_training(self, during_training: bool):
        for dataset in self._datasets:
            dataset.during_training = during_training

    @property
    def repo_id_to_index(self):
        return {repo_id: i for i, repo_id in enumerate(self.ds_names)}

    @property
    def repo_index_to_id(self):
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features})
        return features

    @property
    def camera_keys(self) -> list[str]:
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        return 1 / self.fps - 1e-4
    
    def get_episode_data(self, episode_idx: int) -> dict:
        for dataset in self._datasets:
            if episode_idx < dataset.num_episodes:
                return dataset.get_episode_data(episode_idx)
            else:
                episode_idx -= dataset.num_episodes
        raise IndexError(f"Episode index {episode_idx} out of bounds.")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Dataset Names: '{self.ds_names}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )

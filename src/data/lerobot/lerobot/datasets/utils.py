#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib.resources
import io
import json
from collections.abc import Iterator
from itertools import accumulate
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace
from typing import Any
import array

import datasets
import jsonlines
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from datasets.table import embed_table_storage
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from PIL import Image as PILImage
from torchvision import transforms

from functools import partial

DEFAULT_CHUNK_SIZE = 1000  # Max number of episodes per chunk

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
TASKS_PATH = "meta/tasks.parquet"
EPISODES_DIR = "meta/episodes"
V21_EPISODES_PATH = "meta/episodes.jsonl"
V21_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
V21_TASKS_PATH = "meta/tasks.jsonl"

ANNOTATION_PATHS = {
    "subtask": "annotations/subtask_annotations.jsonl",
    "scene": "annotations/scene_annotations.jsonl",
    "gripper_mode": "annotations/gripper_mode_annotation.jsonl",
    "gripper_activity": "annotations/gripper_activity_annotation.jsonl",
    "eef_velocity": "annotations/eef_velocity_annotation.jsonl",
    "eef_acc_mag": "annotations/eef_acc_mag_annotation.jsonl",
    "eef_direction": "annotations/eef_direction_annotation.jsonl",
}

V21_DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
V21_DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpeg"

DATASET_CARD_TEMPLATE = """
---
# Metadata will go there
---
This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## {}

"""

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "coarse_task_index": {"dtype": "int64", "shape": (1,), "names": None}, 
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    "coarse_quality_index": {"dtype": "int64", "shape": (1,), "names": None}, 
    "quality_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    serialized_dict = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, np.generic):
            serialized_dict[key] = value.item()
        elif isinstance(value, (int, float)):
            serialized_dict[key] = value
        else:
            raise NotImplementedError(f"The value '{value}' of type '{type(value)}' is not supported.")
    return unflatten_dict(serialized_dict)


def embed_images(dataset: datasets.Dataset) -> datasets.Dataset:
    # Embed image bytes into the table before saving to parquet
    format = dataset.format
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(embed_table_storage, batched=False)
    dataset = dataset.with_format(**format)
    return dataset


def load_json(fpath: Path) -> Any:
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_jsonlines(fpath: Path) -> list[Any]:
    # allow \t and \n
    loose_loader = partial(json.loads, strict=False)
    with jsonlines.open(fpath, "r", loads=loose_loader) as reader:
        return list(reader)


def write_jsonlines(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(data)


def append_jsonlines(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "a") as writer:
        writer.write(data)


def write_info(info: dict, local_dir: Path):
    write_json(info, local_dir / INFO_PATH)


def load_info(local_dir: Path) -> dict:
    info = load_json(local_dir / INFO_PATH)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info


def write_stats(stats: dict, local_dir: Path):
    serialized_stats = serialize_dict(stats)
    write_json(serialized_stats, local_dir / STATS_PATH)


def cast_stats_to_numpy(stats) -> dict[str, dict[str, np.ndarray]]:
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_stats(local_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    if not (local_dir / STATS_PATH).exists():
        return None
    stats = load_json(local_dir / STATS_PATH)
    return cast_stats_to_numpy(stats)


def write_task_v21(task_index: int, task: dict, local_dir: Path):
    task_dict = {
        "task_index": task_index,
        "task": task,
    }
    append_jsonlines(task_dict, local_dir / V21_TASKS_PATH)


def load_tasks_v21(local_dir: Path) -> tuple[dict, dict]:
    """Load task metadata from a LeRobot v2.1 JSONL file."""
    tasks = load_jsonlines(local_dir / V21_TASKS_PATH)
    tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index


def load_tasks(local_dir: Path) -> tuple[dict[int, str], dict[str, int]]:
    """Load the v3 task table without relying on its physical row order."""
    path = local_dir / TASKS_PATH
    frame = pd.read_parquet(path)
    if "task_index" not in frame.columns:
        raise ValueError(f"LeRobot v3 task metadata is missing 'task_index': {path}")

    if "task" in frame.columns:
        task_names = frame["task"].tolist()
    elif not isinstance(frame.index, pd.RangeIndex):
        task_names = frame.index.tolist()
    elif "__index_level_0__" in frame.columns:
        task_names = frame["__index_level_0__"].tolist()
    else:
        raise ValueError(
            "LeRobot v3 task metadata must store task text in a 'task' column "
            f"or the dataframe index: {path}"
        )

    task_rows = [
        (int(index), str(task))
        for index, task in zip(frame["task_index"], task_names, strict=True)
    ]
    tasks = dict(sorted(task_rows))
    if len(tasks) != len(task_rows):
        raise ValueError(f"LeRobot v3 task metadata contains duplicate task indices: {path}")
    return tasks, {task: index for index, task in tasks.items()}


def load_episodes(local_dir: Path) -> dict[int, dict]:
    """Load all relational v3 episode metadata shards."""
    episodes_dir = local_dir / EPISODES_DIR
    files = sorted(episodes_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot v3 episode metadata found under {episodes_dir}")

    episodes: dict[int, dict] = {}
    required = {
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    }
    for path in files:
        table = pq.read_table(path)
        missing = required.difference(table.column_names)
        if missing:
            raise ValueError(f"LeRobot v3 episode metadata {path} is missing columns: {sorted(missing)}")
        for episode in table.to_pylist():
            episode_index = int(episode["episode_index"])
            if episode_index in episodes:
                raise ValueError(f"Duplicate LeRobot v3 episode_index {episode_index} in {path}")
            episode["episode_index"] = episode_index
            episode["length"] = int(episode["length"])
            episode["dataset_from_index"] = int(episode["dataset_from_index"])
            episode["dataset_to_index"] = int(episode["dataset_to_index"])
            if episode["dataset_to_index"] - episode["dataset_from_index"] != episode["length"]:
                raise ValueError(
                    f"LeRobot v3 episode {episode_index} has inconsistent length and dataset offsets"
                )
            episodes[episode_index] = episode

    return dict(sorted(episodes.items()))

def load_annotations(local_dir: Path) -> dict[str, dict[int, str]]:
    annotations = {}
    for key, path in ANNOTATION_PATHS.items():
        anno = load_jsonlines(local_dir / path)
        anno = {item[f"{key}_index"]: item[key] for item in sorted(anno, key=lambda x: x[f"{key}_index"])}
        annotations[key] = anno
    return annotations

def write_episode_v21(episode: dict, local_dir: Path):
    append_jsonlines(episode, local_dir / V21_EPISODES_PATH)


def load_episodes_v21(local_dir: Path) -> dict:
    """Load episode metadata from a LeRobot v2.1 JSONL file."""
    episodes = load_jsonlines(local_dir / V21_EPISODES_PATH)
    return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}


def write_episode_stats_v21(episode_index: int, episode_stats: dict, local_dir: Path):
    # We wrap episode_stats in a dictionary since `episode_stats["episode_index"]`
    # is a dictionary of stats and not an integer.
    episode_stats = {"episode_index": episode_index, "stats": serialize_dict(episode_stats)}
    append_jsonlines(episode_stats, local_dir / V21_EPISODES_STATS_PATH)


def load_episodes_stats_v21(local_dir: Path) -> dict:
    """Load per-episode statistics from a LeRobot v2.1 JSONL file."""
    episodes_stats = load_jsonlines(local_dir / V21_EPISODES_STATS_PATH)
    return {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }
def load_image_as_numpy(
    fpath: str | Path, dtype: np.dtype = np.float32, channel_first: bool = True
) -> np.ndarray:
    img = PILImage.open(fpath).convert("RGB")
    img_array = np.array(img, dtype=dtype)
    if channel_first:  # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
    if np.issubdtype(dtype, np.floating):
        img_array /= 255.0
    return img_array


def hf_transform_to_torch(items_dict: dict[torch.Tensor | None]):
    """Get a transform function that convert items from Hugging Face dataset (pyarrow)
    to torch tensors. Importantly, images are converted from PIL, which corresponds to
    a channel last representation (h w c) of uint8 type, to a torch image representation
    with channel first (c h w) of float32 type in range [0,1].
    """
    for key in items_dict:
        first_item = items_dict[key][0]
        if type(first_item) == dict and 'bytes' in first_item and first_item['bytes'] is not None:
            to_pil = lambda img: PILImage.open(io.BytesIO(img['bytes']))
            items_dict[key] = [to_pil(img) for img in items_dict[key]]
            first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif first_item is None:
            pass
        else:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict


def get_hf_features_from_features(features: dict) -> datasets.Features:
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
        else:
            raise ValueError(f"Corresponding feature is not valid: {ft}")

    return datasets.Features(hf_features)


def _validate_feature_names(features: dict[str, dict]) -> None:
    invalid_features = {name: ft for name, ft in features.items() if "/" in name}
    if invalid_features:
        raise ValueError(f"Feature names should not contain '/'. Found '/' in '{invalid_features}'.")


def hw_to_dataset_features(
    hw_features: dict[str, type | tuple], prefix: str, use_video: bool = True
) -> dict[str, dict]:
    features = {}
    joint_fts = {key: ftype for key, ftype in hw_features.items() if ftype is float}
    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}

    if joint_fts and prefix == "action":
        features[prefix] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    if joint_fts and prefix == "observation":
        features[f"{prefix}.state"] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    for key, shape in cam_fts.items():
        features[f"{prefix}.images.{key}"] = {
            "dtype": "video" if use_video else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    _validate_feature_names(features)
    return features


def build_dataset_frame(
    ds_features: dict[str, dict], values: dict[str, Any], prefix: str
) -> dict[str, np.ndarray]:
    frame = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]

    return frame


def create_empty_dataset_info_v21(
    codebase_version: str,
    fps: int,
    features: dict,
    use_videos: bool,
    robot_type: str | None = None,
) -> dict:
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {},
        "data_path": V21_DEFAULT_PARQUET_PATH,
        "video_path": V21_DEFAULT_VIDEO_PATH if use_videos else None,
        "features": features,
    }


def get_episode_data_index(
    episode_dicts: dict[dict], episodes: list[int] | None = None
) -> dict[str, torch.Tensor]:
    episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in episode_dicts.items()}
    if episodes is not None:
        episode_lengths = {ep_idx: episode_lengths[ep_idx] for ep_idx in episodes}

    cumulative_lengths = list(accumulate(episode_lengths.values()))
    return {
        "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
        "to": torch.LongTensor(cumulative_lengths),
    }


def check_timestamps_sync(
    timestamps: np.ndarray,
    episode_indices: np.ndarray,
    episode_data_index: dict[str, np.ndarray],
    fps: int,
    tolerance_s: float,
    raise_value_error: bool = True,
) -> bool:
    """
    This check is to make sure that each timestamp is separated from the next by (1/fps) +/- tolerance
    to account for possible numerical error.

    Args:
        timestamps (np.ndarray): Array of timestamps in seconds.
        episode_indices (np.ndarray): Array indicating the episode index for each timestamp.
        episode_data_index (dict[str, np.ndarray]): A dictionary that includes 'to',
            which identifies indices for the end of each episode.
        fps (int): Frames per second. Used to check the expected difference between consecutive timestamps.
        tolerance_s (float): Allowed deviation from the expected (1/fps) difference.
        raise_value_error (bool): Whether to raise a ValueError if the check fails.

    Returns:
        bool: True if all checked timestamp differences lie within tolerance, False otherwise.

    Raises:
        ValueError: If the check fails and `raise_value_error` is True.
    """
    if timestamps.shape != episode_indices.shape:
        raise ValueError(
            "timestamps and episode_indices should have the same shape. "
            f"Found {timestamps.shape=} and {episode_indices.shape=}."
        )

    # Consecutive differences
    diffs = np.diff(timestamps)
    within_tolerance = np.abs(diffs - (1.0 / fps)) <= tolerance_s

    # Mask to ignore differences at the boundaries between episodes
    mask = np.ones(len(diffs), dtype=bool)
    ignored_diffs = episode_data_index["to"][:-1] - 1  # indices at the end of each episode
    mask[ignored_diffs] = False
    filtered_within_tolerance = within_tolerance[mask]

    # Check if all remaining diffs are within tolerance
    if not np.all(filtered_within_tolerance):
        # Track original indices before masking
        original_indices = np.arange(len(diffs))
        filtered_indices = original_indices[mask]
        outside_tolerance_filtered_indices = np.nonzero(~filtered_within_tolerance)[0]
        outside_tolerance_indices = filtered_indices[outside_tolerance_filtered_indices]

        outside_tolerances = []
        for idx in outside_tolerance_indices:
            entry = {
                "timestamps": [timestamps[idx], timestamps[idx + 1]],
                "diff": diffs[idx],
                "episode_index": episode_indices[idx].item()
                if hasattr(episode_indices[idx], "item")
                else episode_indices[idx],
            }
            outside_tolerances.append(entry)

        if raise_value_error:
            raise ValueError(
                f"""One or several timestamps unexpectedly violate the tolerance inside episode range.
                This might be due to synchronization issues during data collection.
                \n{pformat(outside_tolerances)}"""
            )
        return False

    return True


def check_delta_timestamps(
    delta_timestamps: dict[str, list[float]], fps: int, tolerance_s: float, raise_value_error: bool = True
) -> bool:
    """This will check if all the values in delta_timestamps are multiples of 1/fps +/- tolerance.
    This is to ensure that these delta_timestamps added to any timestamp from a dataset will themselves be
    actual timestamps from the dataset.
    """
    outside_tolerance = {}
    for key, delta_ts in delta_timestamps.items():
        within_tolerance = [abs(ts * fps - round(ts * fps)) / fps <= tolerance_s for ts in delta_ts]
        if not all(within_tolerance):
            outside_tolerance[key] = [
                ts for ts, is_within in zip(delta_ts, within_tolerance, strict=True) if not is_within
            ]

    if len(outside_tolerance) > 0:
        if raise_value_error:
            raise ValueError(
                f"""
                The following delta_timestamps are found outside of tolerance range.
                Please make sure they are multiples of 1/{fps} +/- tolerance and adjust
                their values accordingly.
                \n{pformat(outside_tolerance)}
                """
            )
        return False

    return True


def get_delta_indices(delta_timestamps: dict[str, list[float]], fps: int) -> dict[str, list[int]]:
    delta_indices = {}
    for key, delta_ts in delta_timestamps.items():
        delta_indices[key] = [round(d * fps) for d in delta_ts]

    return delta_indices


def cycle(iterable):
    """The equivalent of itertools.cycle, but safe for Pytorch dataloaders.

    See https://github.com/pytorch/pytorch/issues/23900 for information on why itertools.cycle is not safe.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def create_branch(repo_id, *, branch: str, repo_type: str | None = None) -> None:
    """Create a branch on a existing Hugging Face repo. Delete the branch if it already
    exists before creating it.
    """
    api = HfApi()

    branches = api.list_repo_refs(repo_id, repo_type=repo_type).branches
    refs = [branch.ref for branch in branches]
    ref = f"refs/heads/{branch}"
    if ref in refs:
        api.delete_branch(repo_id, repo_type=repo_type, branch=branch)

    api.create_branch(repo_id, repo_type=repo_type, branch=branch)


def create_lerobot_dataset_card(
    tags: list | None = None,
    dataset_info: dict | None = None,
    **kwargs,
) -> DatasetCard:
    """
    Keyword arguments will be used to replace values in src/lerobot/datasets/card_template.md.
    Note: If specified, license must be one of https://huggingface.co/docs/hub/repositories-licenses.
    """
    card_tags = ["LeRobot"]

    if tags:
        card_tags += tags
    if dataset_info:
        dataset_structure = "[meta/info.json](meta/info.json):\n"
        dataset_structure += f"```json\n{json.dumps(dataset_info, indent=4)}\n```\n"
        kwargs = {**kwargs, "dataset_structure": dataset_structure}
    card_data = DatasetCardData(
        license=kwargs.get("license"),
        tags=card_tags,
        task_categories=["robotics"],
        configs=[
            {
                "config_name": "default",
                "data_files": "data/*/*.parquet",
            }
        ],
    )

    card_template = (importlib.resources.files("lerobot.datasets") / "card_template.md").read_text()

    return DatasetCard.from_template(
        card_data=card_data,
        template_str=card_template,
        **kwargs,
    )


class IterableNamespace(SimpleNamespace):
    """Namespace with mapping access and recursive conversion of dictionaries."""

    def __init__(self, dictionary: dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if dictionary is not None:
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, IterableNamespace(value))
                else:
                    setattr(self, key, value)

    def __iter__(self) -> Iterator[str]:
        return iter(vars(self))

    def __getitem__(self, key: str) -> Any:
        return vars(self)[key]

    def items(self):
        return vars(self).items()

    def values(self):
        return vars(self).values()

    def keys(self):
        return vars(self).keys()


def validate_frame(frame: dict, features: dict):
    expected_features = set(features) - set(DEFAULT_FEATURES)
    actual_features = set(frame)

    error_message = validate_features_presence(actual_features, expected_features)

    common_features = actual_features & expected_features
    for name in common_features - {"task"}:
        error_message += validate_feature_dtype_and_shape(name, features[name], frame[name])

    if error_message:
        raise ValueError(error_message)


def validate_features_presence(actual_features: set[str], expected_features: set[str]):
    error_message = ""
    missing_features = expected_features - actual_features
    extra_features = actual_features - expected_features

    if missing_features or extra_features:
        error_message += "Feature mismatch in `frame` dictionary:\n"
        if missing_features:
            error_message += f"Missing features: {missing_features}\n"
        if extra_features:
            error_message += f"Extra features: {extra_features}\n"

    return error_message

def is_valid_numpy_dtype_string(dtype_str: str) -> bool:
    """Return whether a string names a NumPy dtype."""
    try:
        np.dtype(dtype_str)
        return True
    except TypeError:
        return False
def validate_feature_dtype_and_shape(name: str, feature: dict, value: np.ndarray | PILImage.Image | str | bytes):
    if isinstance(value, bytes) or isinstance(value, array.array): # ROS 1 and 2
        # Encoded ROS image payloads are validated when decoded.
        return ""
    expected_dtype = feature["dtype"]
    expected_shape = feature["shape"]
    if is_valid_numpy_dtype_string(expected_dtype):
        return validate_feature_numpy_array(name, expected_dtype, expected_shape, value)
    elif expected_dtype in ["image", "video"]:
        return validate_feature_image_or_video(name, expected_shape, value)
    elif expected_dtype == "string":
        return validate_feature_string(name, value)
    else:
        raise NotImplementedError(f"The feature dtype '{expected_dtype}' is not implemented yet.")


def validate_feature_numpy_array(
    name: str, expected_dtype: str, expected_shape: list[int], value: np.ndarray
):
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_dtype = value.dtype
        actual_shape = value.shape

        if actual_dtype != np.dtype(expected_dtype):
            error_message += f"The feature '{name}' of dtype '{actual_dtype}' is not of the expected dtype '{expected_dtype}'.\n"

        if actual_shape != expected_shape:
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{expected_shape}'.\n"
    else:
        error_message += f"The feature '{name}' is not a 'np.ndarray'. Expected type is '{expected_dtype}', but type '{type(value)}' provided instead.\n"

    return error_message


def validate_feature_image_or_video(name: str, expected_shape: list[str], value: np.ndarray | PILImage.Image):
    # Note: The check of pixels range ([0,1] for float and [0,255] for uint8) is done by the image writer threads.
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_shape = value.shape
        c, h, w = expected_shape
        if len(actual_shape) != 3 or (actual_shape != (c, h, w) and actual_shape != (h, w, c)):
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{(c, h, w)}' or '{(h, w, c)}'.\n"
    elif isinstance(value, PILImage.Image):
        pass
    else:
        error_message += f"The feature '{name}' is expected to be of type 'PIL.Image' or 'np.ndarray' channel first or channel last, but type '{type(value)}' provided instead.\n"

    return error_message


def validate_feature_string(name: str, value: str):
    if not isinstance(value, str):
        return f"The feature '{name}' is expected to be of type 'str', but type '{type(value)}' provided instead.\n"
    return ""


def validate_episode_buffer(episode_buffer: dict, total_episodes: int, features: dict):
    if "size" not in episode_buffer:
        raise ValueError("size key not found in episode_buffer")

    if "task" not in episode_buffer:
        raise ValueError("task key not found in episode_buffer")

    if episode_buffer["episode_index"] != total_episodes:
        raise NotImplementedError(
            "You might have manually provided the episode_buffer with an episode_index that doesn't "
            "match the total number of episodes already in the dataset. This is not supported for now."
        )

    if episode_buffer["size"] == 0:
        raise ValueError("You must add one or several frames with `add_frame` before calling `add_episode`.")

    buffer_keys = set(episode_buffer.keys()) - {"task", "size"}
    if not buffer_keys == set(features):
        raise ValueError(
            f"Features from `episode_buffer` don't match the ones in `features`."
            f"In episode_buffer not in features: {buffer_keys - set(features)}"
            f"In features not in episode_buffer: {set(features) - buffer_keys}"
        )

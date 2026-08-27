import glob
import hashlib
import os
from dataclasses import dataclass
from typing import Union

import torch
from safetensors import safe_open


@dataclass
class LocalModelFile:
    root: str
    pattern: Union[str, list[str], None] = None
    path: Union[str, list[str], None] = None

    def _parse_pattern(self):
        if self.pattern in [None, "", "./"]:
            return "*"
        if isinstance(self.pattern, list):
            return self.pattern
        if self.pattern.endswith("/"):
            return self.pattern + "*"
        return self.pattern

    def resolve(self):
        if self.path is not None:
            return self.path

        root = os.path.expanduser(self.root)
        pattern = self._parse_pattern()
        if pattern == "*":
            if not os.path.isdir(root):
                raise FileNotFoundError(f"Missing local pretrained directory: {root}")
            self.path = root
            return self.path

        if isinstance(pattern, list):
            matches: list[str] = []
            missing_patterns: list[str] = []
            for item in pattern:
                item_matches = sorted(glob.glob(os.path.join(root, item)))
                if not item_matches:
                    missing_patterns.append(item)
                matches.extend(item_matches)
            if missing_patterns:
                raise FileNotFoundError(
                    f"Missing local pretrained files under {root}: patterns={missing_patterns}"
                )
        else:
            matches = sorted(glob.glob(os.path.join(root, pattern)))
            if not matches:
                raise FileNotFoundError(
                    f"Missing local pretrained files under {root}: pattern={pattern}"
                )

        self.path = matches
        if isinstance(self.path, list) and len(self.path) == 1:
            self.path = self.path[0]
        return self.path


def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_state_dict(file_path_, torch_dtype=torch_dtype, device=device))
        return state_dict
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for key in f.keys():
            value = f.get_tensor(key)
            if torch_dtype is not None:
                value = value.to(torch_dtype)
            state_dict[key] = value
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if len(state_dict) == 1:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
    if torch_dtype is not None:
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].to(torch_dtype)
    return state_dict


def _load_keys_dict_from_safetensors(file_path):
    keys_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            keys_dict[key] = f.get_slice(key).get_shape()
    return keys_dict


def _convert_state_dict_to_keys_dict(state_dict):
    keys_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            keys_dict[key] = list(value.shape)
        else:
            keys_dict[key] = _convert_state_dict_to_keys_dict(value)
    return keys_dict


def _load_keys_dict_from_bin(file_path):
    state_dict = load_state_dict_from_bin(file_path)
    return _convert_state_dict_to_keys_dict(state_dict)


def _load_keys_dict(file_path):
    if isinstance(file_path, list):
        merged = {}
        for path in file_path:
            merged.update(_load_keys_dict(path))
        return merged
    if file_path.endswith(".safetensors"):
        return _load_keys_dict_from_safetensors(file_path)
    return _load_keys_dict_from_bin(file_path)


def _convert_keys_dict_to_single_str(keys_dict, with_shape=True):
    keys = []
    for key, value in keys_dict.items():
        if isinstance(key, str):
            if isinstance(value, dict):
                keys.append(key + "|" + _convert_keys_dict_to_single_str(value, with_shape=with_shape))
            else:
                if with_shape:
                    shape = "_".join(map(str, list(value)))
                    keys.append(key + ":" + shape)
                keys.append(key)
    keys.sort()
    return ",".join(keys)


def hash_model_file(path, with_shape=True):
    keys_dict = _load_keys_dict(path)
    keys_str = _convert_keys_dict_to_single_str(keys_dict, with_shape=with_shape).encode("UTF-8")
    return hashlib.md5(keys_str).hexdigest()

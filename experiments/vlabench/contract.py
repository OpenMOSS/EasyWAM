"""VLABench data/control contract. Geometry helpers also run in the simulator env.

The official dataset stores base-relative absolute poses with extrinsic XYZ
Euler angles. EasyWAM uses the first two *rows* of the rotation matrix (the
convention of data.lerobot.utils.rotation), not OpenWAM's column convention.
State gripper 1 means closed; action gripper 1 means open in this release.
"""
from __future__ import annotations

import numpy as np

CONTRACT_VERSION = 1
CAMERA_INDICES = (2, 3, 0)
CAMERA_SIZES = ((256, 320), (128, 160), (128, 160))
FIXED_TRACKS = (
    "track_1_in_distribution", "track_2_cross_category", "track_3_common_sense",
    "track_4_semantic_instruction", "track_6_unseen_texture",
)


def _finite(value, width: int, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != width or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite [...,{width}], got {value.shape}")
    return value


def euler_xyz_to_matrix(euler) -> np.ndarray:
    x, y, z = np.moveaxis(_finite(euler, 3, "Euler XYZ"), -1, 0)
    cx, cy, cz, sx, sy, sz = np.cos(x), np.cos(y), np.cos(z), np.sin(x), np.sin(y), np.sin(z)
    # Extrinsic xyz: Rz @ Ry @ Rx, matching scipy Rotation.from_euler('xyz').
    return np.stack((cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx,
                     sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx,
                     -sy, cy*sx, cy*cx), axis=-1).reshape(x.shape + (3, 3))


def matrix_to_rot6d(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite [...,3,3] rotation matrices")
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def rot6d_to_matrix(value) -> np.ndarray:
    value = _finite(value, 6, "rotation-6D")
    a, b = value[..., :3], value[..., 3:]
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate rotation-6D first row")
    a = a / norm
    b = b - (a*b).sum(axis=-1, keepdims=True)*a
    norm = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate rotation-6D parallel rows")
    b = b / norm
    return np.stack((a, b, np.cross(a, b)), axis=-2)


def matrix_to_euler_xyz(matrix) -> np.ndarray:
    r = np.asarray(matrix, dtype=np.float64)
    pitch = np.arcsin(np.clip(-r[..., 2, 0], -1, 1))
    singular = np.hypot(r[..., 0, 0], r[..., 1, 0]) < 1e-7
    roll = np.where(singular, 0., np.arctan2(r[..., 2, 1], r[..., 2, 2]))
    yaw = np.where(singular, np.arctan2(-r[..., 0, 1], r[..., 1, 1]),
                   np.arctan2(r[..., 1, 0], r[..., 0, 0]))
    return np.stack((roll, pitch, yaw), axis=-1)


def quaternion_wxyz_to_matrix(quaternion) -> np.ndarray:
    q = _finite(quaternion, 4, "quaternion wxyz")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Zero quaternion")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(w.shape + (3, 3))


def eef7_to_eef10(value) -> np.ndarray:
    value = _finite(value, 7, "dataset EEF pose")
    return np.concatenate((value[..., :3], matrix_to_rot6d(euler_xyz_to_matrix(value[..., 3:6])),
                           value[..., 6:7]), axis=-1).astype(np.float32)


def observation_to_state(observation: dict) -> np.ndarray:
    ee = _finite(observation["ee_state"], 8, "ee_state")
    base = _finite(observation["robot_frame"], 3, "robot_frame")
    if ee.ndim != 1 or base.ndim != 1:
        raise ValueError("Expected one robot observation")
    return np.concatenate((ee[:3]-base, matrix_to_rot6d(quaternion_wxyz_to_matrix(ee[3:7])),
                           ee[7:8])).astype(np.float32)


def action_to_target(action, robot_frame, *, gripper_threshold=.5, gripper_width=.04):
    action = _finite(action, 10, "action")
    base = _finite(robot_frame, 3, "robot_frame")
    if action.ndim != 1 or base.ndim != 1:
        raise ValueError("Expected one action and one robot frame")
    if not 0 <= gripper_threshold <= 1 or not np.isfinite(gripper_width) or gripper_width <= 0:
        raise ValueError("Invalid gripper threshold/width")
    return (action[:3] + base, matrix_to_euler_xyz(rot6d_to_matrix(action[3:9])),
            np.full(2, gripper_width if action[9] >= gripper_threshold else 0.))


def observation_cameras(observation: dict) -> list[np.ndarray]:
    rgb = np.asarray(observation["rgb"])
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or len(rgb) < 4:
        raise ValueError("rgb must be uint8 [at least 4 cameras,H,W,3]")
    return [np.ascontiguousarray(rgb[index]) for index in CAMERA_INDICES]


def image_tensor(cameras):
    """Match the dataset's ToTensor -> Resize -> robotwin canvas -> Normalize."""
    import torch
    from torchvision.transforms.functional import resize

    if len(cameras) != 3:
        raise ValueError("Exactly three cameras required")
    views = []
    for image, size in zip(cameras, CAMERA_SIZES):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("Camera image must be uint8 HWC RGB")
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.
        views.append(resize(tensor, list(size), antialias=True))
    canvas = torch.cat((views[0], torch.cat(views[1:], dim=-1)), dim=-2)
    return (canvas * 2. - 1.).unsqueeze(0)

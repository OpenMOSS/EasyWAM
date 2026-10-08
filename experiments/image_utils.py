import numpy as np
from numpy.typing import NDArray
import torch
from PIL import Image


def resize_rgb(image: NDArray[np.uint8], size_wh: tuple[int, int]) -> NDArray[np.uint8]:
    if image.shape[:2] == (size_wh[1], size_wh[0]):
        return image
    return np.asarray(
        Image.fromarray(image, mode="RGB").resize(size_wh, resample=Image.BILINEAR),
        dtype=np.uint8,
    )


def center_crop_resize(image: NDArray[np.uint8], width: int, height: int) -> NDArray[np.uint8]:
    if image.shape[:2] == (height, width):
        return image
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
    )
    rw, rh = resized.size
    if (rw, rh) != (width, height):
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        resized = resized.crop((left, top, left + width, top + height))
    return np.asarray(resized, dtype=np.uint8)


def concatenate_head_and_wrists(
    head: NDArray[np.uint8], left: NDArray[np.uint8], right: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    if left.shape[0] != right.shape[0] or left.shape[1] + right.shape[1] != head.shape[1]:
        raise ValueError("Wrist images must form one row matching the head image width.")
    canvas = np.empty((head.shape[0] + left.shape[0], head.shape[1], 3), dtype=np.uint8)
    canvas[: head.shape[0]] = head
    canvas[head.shape[0] :, : left.shape[1]] = left
    canvas[head.shape[0] :, left.shape[1] :] = right
    return canvas


def rgb_to_tensor(
    rgb: NDArray[np.uint8], *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    if not rgb.flags.writeable or any(stride < 0 for stride in rgb.strides):
        rgb = rgb.copy(order="C")
    image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return image * (2.0 / 255.0) - 1.0

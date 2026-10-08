from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pil_frames_to_video_tensor(frames: Sequence[Image.Image]) -> torch.Tensor:
    if len(frames) == 0:
        raise ValueError("`frames` must be non-empty.")

    frame_tensors = []
    for frame in frames:
        arr = np.array(frame.convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        frame_tensors.append(x)
    return torch.stack(frame_tensors, dim=1)


def _gaussian_kernel_2d(kernel_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    return kernel_2d.repeat(channels, 1, 1, 1)


def video_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> float:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")

    pred = pred.float()
    target = target.float()
    mse = (pred - target).pow(2).mean(dim=(0, 2, 3))  # [T]
    psnr = 10.0 * torch.log10((data_range * data_range) / (mse + eps))
    return float(psnr.mean().item())


def video_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    return _ssim_values(
        (pred, target), ((0, 1),), data_range, kernel_size, sigma, k1, k2
    )[0]


def video_ssim_triplet(
    pred: torch.Tensor, target: torch.Tensor, reconstruction: torch.Tensor
) -> tuple[float, float, float]:
    return tuple(_ssim_values((pred, target, reconstruction), ((0, 1), (2, 1), (0, 2))))


@torch.no_grad()
def _ssim_values(
    videos: Sequence[torch.Tensor],
    pairs: Sequence[tuple[int, int]],
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> list[float]:
    if not pairs:
        return []
    if kernel_size % 2 == 0:
        raise ValueError("`kernel_size` must be odd.")
    used = sorted({index for pair in pairs for index in pair})
    for i, j in pairs:
        if videos[i].shape != videos[j].shape:
            raise ValueError(f"Shape mismatch: pred={tuple(videos[i].shape)} target={tuple(videos[j].shape)}")
    for index in used:
        video = videos[index]
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(f"Expected [3, T, H, W], got {tuple(video.shape)}")
    first = videos[used[0]]
    kernel = _gaussian_kernel_2d(kernel_size, sigma, 3, first.device, torch.float32)
    pad = kernel_size // 2
    statistics = {}
    layouts = {}

    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    results = []
    for i, j in pairs:

        for index in tuple(layouts):
            if index != i and index != j:
                del layouts[index]
        for index in dict.fromkeys((i, j)):
            if index not in layouts:
                layouts[index] = videos[index].float().permute(1, 0, 2, 3).contiguous()
            if index not in statistics:
                image = layouts[index]
                mean = F.conv2d(image, kernel, padding=pad, groups=3)
                variance = F.conv2d(image * image, kernel, padding=pad, groups=3)
                variance.sub_(mean * mean)
                statistics[index] = (mean, variance)
                del image, mean, variance
        x, y = layouts[i], layouts[j]
        mu_x, var_x = statistics[i]
        mu_y, var_y = statistics[j]
        numerator = mu_x * mu_y
        covariance = F.conv2d(x * y, kernel, padding=pad, groups=3)
        covariance.sub_(numerator)
        numerator.mul_(2.0).add_(c1)
        covariance.mul_(2.0).add_(c2)
        numerator.mul_(covariance)
        del covariance
        denominator = mu_x * mu_x
        denominator.add_(mu_y * mu_y).add_(c1)
        variance_sum = var_x + var_y
        variance_sum.add_(c2)
        denominator.mul_(variance_sum).add_(1e-12)
        del variance_sum
        numerator.div_(denominator)
        results.append(float(numerator.mean().item()))
        del numerator, denominator, x, y, mu_x, mu_y, var_x, var_y
    return results

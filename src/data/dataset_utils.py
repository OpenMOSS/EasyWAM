
from typing import Optional

import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image


def obtain_image_size(data: torch.Tensor | Image.Image) -> tuple[int, int]:
    """Return the width and height of an image or video tensor."""

    if isinstance(data, Image.Image):
        width, height = data.size
    elif isinstance(data, torch.Tensor):
        height, width = data.size()[-2:]
    else:
        raise ValueError("data to random crop should be PIL Image or tensor")

    return width, height

class ResizeSmallestSideAspectPreserving:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor | Image.Image:
        """Resize until both dimensions meet the target size."""

        assert self.args is not None, "Please specify args in augmentations"

        img_w, img_h = self.args["img_w"], self.args["img_h"]

        orig_w, orig_h = obtain_image_size(video)
        scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert (
            target_size[0] >= img_h and target_size[1] >= img_w
        ), f"Resize error. orig {(orig_w, orig_h)} desire {(img_w, img_h)} compute {target_size}"

        return transforms_F.resize(
            video,
            size=target_size,  # type: ignore
            interpolation=self.args.get("interpolation", transforms_F.InterpolationMode.BICUBIC),
            antialias=True,
        )


class CenterCrop:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor | Image.Image:
        """Center crop to the requested size."""
        assert (
            (self.args is not None) and ("img_w" in self.args) and ("img_h" in self.args)
        ), "Please specify size in args"

        img_w, img_h = self.args["img_w"], self.args["img_h"]
        return transforms_F.center_crop(video, [img_h, img_w])


class Normalize:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor:
        """Convert to a tensor and normalize by mean and standard deviation."""
        assert self.args is not None, "Please specify args"

        mean = self.args["mean"]
        std = self.args["std"]

        if isinstance(video, torch.Tensor):
            data = video.to(dtype=torch.float32)
            if video.dtype == torch.uint8:
                data = data / 255.0
            data = data.to(dtype=torch.get_default_dtype())
        else:
            data = transforms_F.to_tensor(video)

        return transforms_F.normalize(tensor=data, mean=mean, std=std)

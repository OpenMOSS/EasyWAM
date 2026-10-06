from typing import Dict

import torch
from torch.nn.functional import pad


class ConcatLeftAlign:
    def __init__(
        self, 
        action_target_dim: int | None = None, 
        state_target_dim: int | None = None
    ):
        self.action_target_dim = action_target_dim
        self.state_target_dim = state_target_dim
        self._padding_masks = {}

    def set_shape_meta(self, shape_meta):
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        if "action" in batch:
            batch["action"] = self._concat(batch["action"], self.action_meta)
            batch["action"], batch["action_dim_is_pad"] = self._pad(batch["action"], self.action_target_dim)

        batch["state"] = self._concat(batch["state"], self.state_meta)
        batch["state"], batch["state_dim_is_pad"] = self._pad(batch["state"], self.state_target_dim)

        return batch

    def backward(self, batch):
        if self.state_target_dim is not None:
            assert batch["state"].shape[-1] == self.state_target_dim
        batch["state"] = self._crop(batch["state"], self.state_meta)
        batch["state"] = self._split(batch["state"], self.state_meta)
        
        if self.action_target_dim is not None:
            assert batch["action"].shape[-1] == self.action_target_dim
        batch["action"] = self._crop(batch["action"], self.action_meta)
        batch["action"] = self._split(batch["action"], self.action_meta)

        return batch

    def _pad(self, x: torch.Tensor, dim: int):
        if dim is None:
            dim = x.shape[-1]
        
        assert x.ndim == 2 and x.shape[-1] <= dim
        pad_dim = dim - x.shape[-1]
        x_padded = pad(x, (0, pad_dim)) if pad_dim else x
        cache_key = (dim, x.shape[-1], x.device)
        if cache_key not in self._padding_masks:
            self._padding_masks[cache_key] = torch.arange(dim, device=x.device) >= x.shape[-1]
        mask = self._padding_masks[cache_key]
        return x_padded, mask

    @staticmethod
    def _crop(x: torch.Tensor, meta: int):
        assert x.ndim == 3
        dim = sum([m["shape"] for m in meta])
        x = x[:, :, :dim]
        return x
    
    @staticmethod
    def _concat(x: Dict[str, torch.Tensor], meta: Dict[str, Dict]):
        x = x[meta[0]["key"]] if len(meta) == 1 else torch.cat([x[m["key"]] for m in meta], dim=-1)
        assert x.ndim == 2
        return x

    @staticmethod
    def _split(x: torch.Tensor, meta: Dict[str, Dict]):
        assert x.ndim == 3
        y = {}
        idx = 0
        for m in meta:
            key, dim = m["key"], m["shape"]
            y[key] = x[:, :, idx: idx + dim]
            idx += dim

        return y
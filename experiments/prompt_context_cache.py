"""Small in-memory evaluation cache for deterministic text encoder outputs."""

from __future__ import annotations

import time
from typing import Optional

import torch


class PromptContextCache:
    """Keep only the current prompt on the model device.

    Evaluation prompts are encoded on demand and retained on the model device
    until ``clear`` or the next prompt.
    """

    def __init__(
        self,
        model,
    ) -> None:
        self.model = model
        self._prompt: Optional[str] = None
        self._value: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self.encode_seconds = 0.0
        self.encode_calls = 0

    def clear(self) -> None:
        self._prompt = None
        self._value = None

    def get(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt == self._prompt and self._value is not None:
            return self._value

        started = time.perf_counter()
        context, mask = self.model.encode_prompt(prompt)
        self.encode_calls += 1

        device = self.model.device
        dtype = self.model.torch_dtype
        value = (
            context.detach().to(device=device, dtype=dtype),
            mask.detach().to(device=device, dtype=torch.bool),
        )
        self.encode_seconds += time.perf_counter() - started
        self._prompt = prompt
        self._value = value
        return value

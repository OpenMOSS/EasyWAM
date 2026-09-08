"""Small in-memory evaluation cache for deterministic text encoder outputs."""

from __future__ import annotations

import time
from collections import OrderedDict
import torch


class PromptContextCache:
    """LRU cache for evaluation prompt embeddings on the model device."""

    def __init__(self, model, max_entries: int = 1) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive.")
        self.model = model
        self.max_entries = int(max_entries)
        self._values: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.encode_seconds = 0.0
        self.encode_calls = 0

    def clear(self) -> None:
        self._values.clear()

    def get(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt in self._values:
            self._values.move_to_end(prompt)
            context, mask = self._values[prompt]
            return context.unsqueeze(0), mask.unsqueeze(0)

        started = time.perf_counter()
        context, mask = self.model.encode_prompt(prompt)
        self.encode_calls += 1

        device = self.model.device
        dtype = self.model.torch_dtype
        if context.ndim != 3 or context.shape[0] != 1 or mask.ndim != 2 or mask.shape[0] != 1:
            raise ValueError("Single-prompt encoding must return [1,L,D] context and [1,L] mask.")
        value = (
            context[0].detach().to(device=device, dtype=dtype),
            mask[0].detach().to(device=device, dtype=torch.bool),
        )
        self.encode_seconds += time.perf_counter() - started
        self._values[prompt] = value
        self._values.move_to_end(prompt)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)
        return value[0].unsqueeze(0), value[1].unsqueeze(0)

    def get_many(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        cached = {}
        for prompt in dict.fromkeys(prompts):
            if prompt in self._values:
                self._values.move_to_end(prompt)
                cached[prompt] = self._values[prompt]
        missing = list(dict.fromkeys(prompt for prompt in prompts if prompt not in self._values))
        fresh = {}
        if missing:
            started = time.perf_counter()
            contexts, masks = self.model.encode_prompt(missing)
            self.encode_calls += 1
            if contexts.shape[0] != len(missing) or masks.shape[0] != len(missing):
                raise ValueError("Text encoder batch size does not match prompt batch size.")
            for index, prompt in enumerate(missing):
                valid_positions = masks[index].to(dtype=torch.bool).nonzero(as_tuple=False)
                length = int(valid_positions[-1].item()) + 1 if len(valid_positions) else 1
                value = (
                    contexts[index, :length].detach().to(
                        device=self.model.device, dtype=self.model.torch_dtype
                    ),
                    masks[index, :length].detach().to(device=self.model.device, dtype=torch.bool),
                )
                fresh[prompt] = value
                self._values[prompt] = value
                self._values.move_to_end(prompt)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)
            self.encode_seconds += time.perf_counter() - started
        values = []
        for prompt in prompts:
            if prompt in self._values:
                self._values.move_to_end(prompt)
                values.append(self._values[prompt])
            elif prompt in cached:
                values.append(cached[prompt])
            else:
                values.append(fresh[prompt])
        max_length = max(value[0].shape[0] for value in values)
        hidden_dim = values[0][0].shape[1]
        contexts = values[0][0].new_zeros((len(values), max_length, hidden_dim))
        masks = values[0][1].new_zeros((len(values), max_length))
        for index, (context, mask) in enumerate(values):
            length = context.shape[0]
            if context.shape[1] != hidden_dim or mask.shape != (length,):
                raise ValueError("Cached prompt context shapes are inconsistent.")
            contexts[index, :length] = context
            masks[index, :length] = mask
        return contexts, masks

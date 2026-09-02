from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class Flux2Qwen3TextEncoder(nn.Module):
    """Qwen3 adapter producing the multi-layer context expected by FLUX.2."""

    def __init__(self, model: nn.Module, tokenizer, output_layers=(9, 18, 27), max_length: int = 512):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.output_layers = tuple(int(index) for index in output_layers)
        self.max_length = int(max_length)
        hidden_size = int(model.config.hidden_size)
        self.dim = hidden_size * len(self.output_layers)

    @torch.no_grad()
    def forward(self, prompts: str | Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]
        rendered = []
        for prompt in prompts:
            messages = [{"role": "user", "content": str(prompt)}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            rendered.append(text)
        batch = self.tokenizer(
            rendered,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = self.model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states
        if max(self.output_layers) >= len(hidden_states):
            raise ValueError(
                f"Qwen returned {len(hidden_states)} hidden states; requested {self.output_layers}."
            )
        context = torch.cat([hidden_states[index] for index in self.output_layers], dim=-1)
        return context, batch["attention_mask"].to(dtype=torch.bool)

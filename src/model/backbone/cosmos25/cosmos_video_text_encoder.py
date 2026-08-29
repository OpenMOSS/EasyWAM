"""Cosmos-Predict2.5 Reason1 text-conditioning adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


COSMOS_REASON_SYSTEM_PROMPT = (
    "You are a helpful assistant who will provide prompts to an image generator."
)


class Cosmos25TextEncoder(nn.Module):
    """Produce the normalized final Reason1 hidden state used for conditioning."""

    output_dim = 3584

    def __init__(self, model: nn.Module, tokenizer, max_length: int = 128):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 128,
    ) -> "Cosmos25TextEncoder":
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        model_path = str(Path(model_path))
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.requires_grad_(False)
        return cls(model=model, tokenizer=tokenizer, max_length=max_length)

    @staticmethod
    def _conversation(prompt: str) -> list[dict]:
        """Build the text-only conversation used by Cosmos-Predict2.5."""
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": COSMOS_REASON_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": str(prompt)}],
            },
        ]

    def _tokenize(self, prompt: str):
        # Apply the official Qwen chat template and tokenize in one operation.
        return self.tokenizer.apply_chat_template(
            self._conversation(prompt),
            tokenize=True,
            add_generation_prompt=False,
            add_vision_id=False,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_dict=True,
            return_tensors="pt",
        )

    @torch.no_grad()
    def forward(self, prompt: str | Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts or not all(isinstance(value, str) for value in prompts):
            raise TypeError("prompt must be a string or a non-empty sequence of strings.")
        encoded = [self._tokenize(value) for value in prompts]
        input_ids = torch.cat([value["input_ids"] for value in encoded], dim=0)
        attention_mask = torch.cat([value["attention_mask"] for value in encoded], dim=0)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) < 2:
            raise RuntimeError("Cosmos-Reason1 did not return transformer hidden states.")
        hidden = hidden_states[-1]
        hidden_float = hidden.float()
        mean = hidden_float.mean(dim=-1, keepdim=True)
        std = hidden_float.std(dim=-1, keepdim=True)
        context = ((hidden_float - mean) / (std + 1e-8)).to(hidden.dtype)
        if context.shape[-1] != self.output_dim:
            raise RuntimeError(f"Expected Reason context width {self.output_dim}, got {context.shape[-1]}.")
        return context, attention_mask.to(torch.bool)

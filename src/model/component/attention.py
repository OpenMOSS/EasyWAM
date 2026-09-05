from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from utils.logging_config import get_logger

logger = get_logger(__name__)

ATTENTION_BACKENDS = ("sdpa", "fa2", "fa3", "fa4", "auto")
AUTO_ATTENTION_BACKENDS = ("fa4", "fa3", "fa2", "sdpa")
_FLASH_KERNELS: dict[str, Callable] = {}
_FLASH_VARLEN_KERNELS: dict[str, Callable] = {}
_LOGGED_SELECTIONS: set[tuple[str, str]] = set()


def normalize_attention_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    if value not in ATTENTION_BACKENDS:
        raise ValueError(
            f"Unsupported attention backend: {backend}. "
            f"Expected one of: {list(ATTENTION_BACKENDS)}."
        )
    return value


def require_attention_backend(backend: str) -> str:
    value = normalize_attention_backend(backend)
    if value not in ("sdpa", "auto"):
        _load_flash_kernel(value)
    return value


@dataclass(frozen=True)
class AttentionSegment:
    query_start: int
    query_end: int
    key_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class KeyPaddingMask:
    valid: torch.Tensor
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int

    @classmethod
    def from_tensor(cls, mask: torch.Tensor) -> "KeyPaddingMask":
        if mask.ndim != 2:
            raise ValueError(f"Key padding mask must be [B, K], got {tuple(mask.shape)}.")
        valid = mask.to(dtype=torch.bool)
        lengths = valid.sum(dim=1, dtype=torch.int32)
        if bool((lengths == 0).any().item()):
            raise ValueError("Every sequence must contain at least one valid key token.")
        indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        cu_seqlens = F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
        return cls(valid, indices, cu_seqlens, int(lengths.max().item()))

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self) -> torch.Size:
        return self.valid.shape

    @property
    def is_fully_valid(self) -> bool:
        return self.indices.numel() == self.valid.numel()

    def dim(self) -> int:
        return 2

    def to(self, *args, **kwargs) -> "KeyPaddingMask":
        valid = self.valid.to(*args, **kwargs)
        device = valid.device
        indices = self.indices.to(device=device)
        cu_seqlens = self.cu_seqlens.to(device=device)
        if valid is self.valid and indices is self.indices and cu_seqlens is self.cu_seqlens:
            return self
        return KeyPaddingMask(valid, indices, cu_seqlens, self.max_seqlen)


@dataclass(frozen=True)
class StructuredAttentionMask:
    dense: torch.Tensor
    segments: tuple[AttentionSegment, ...]

    @property
    def ndim(self) -> int:
        return self.dense.ndim

    @property
    def shape(self) -> torch.Size:
        return self.dense.shape

    @property
    def is_fully_valid(self) -> bool:
        key_len = self.dense.shape[1]
        cursor = 0
        for segment in self.segments:
            if (
                segment.query_start != cursor
                or segment.query_end <= segment.query_start
                or segment.key_ranges != ((0, key_len),)
            ):
                return False
            cursor = segment.query_end
        return cursor == self.dense.shape[0]

    def to(self, *args, **kwargs) -> "StructuredAttentionMask":
        dense = self.dense.to(*args, **kwargs)
        if dense is self.dense:
            return self
        return StructuredAttentionMask(dense, self.segments)

    def slice(
        self,
        query_start: int,
        query_end: int,
        key_start: int = 0,
        key_end: Optional[int] = None,
    ) -> "StructuredAttentionMask":
        if key_end is None:
            key_end = self.dense.shape[1]
        if not (0 <= query_start <= query_end <= self.dense.shape[0]):
            raise ValueError("Invalid structured attention query slice.")
        if not (0 <= key_start <= key_end <= self.dense.shape[1]):
            raise ValueError("Invalid structured attention key slice.")
        if (
            query_start == 0
            and query_end == self.dense.shape[0]
            and key_start == 0
            and key_end == self.dense.shape[1]
        ):
            return self

        segments = []
        for segment in self.segments:
            start = max(segment.query_start, query_start)
            end = min(segment.query_end, query_end)
            if start >= end:
                continue
            key_ranges = []
            for range_start, range_end in segment.key_ranges:
                clipped_start = max(range_start, key_start)
                clipped_end = min(range_end, key_end)
                if clipped_start < clipped_end:
                    key_ranges.append((clipped_start - key_start, clipped_end - key_start))
            segments.append(
                AttentionSegment(
                    query_start=start - query_start,
                    query_end=end - query_start,
                    key_ranges=tuple(key_ranges),
                )
            )
        return StructuredAttentionMask(
            dense=self.dense[query_start:query_end, key_start:key_end],
            segments=tuple(segments),
        )


def build_structured_attention_mask(
    query_len: int,
    key_len: int,
    segments: Sequence[AttentionSegment],
    device: torch.device,
) -> StructuredAttentionMask:
    if query_len <= 0 or key_len <= 0:
        raise ValueError(f"Attention lengths must be positive, got query={query_len}, key={key_len}.")

    normalized_segments = []
    cursor = 0
    for segment in segments:
        if segment.query_start != cursor or not (segment.query_start < segment.query_end <= query_len):
            raise ValueError("Attention segments must cover query rows once, contiguously, and in order.")
        previous_key_end = 0
        merged_key_ranges: list[tuple[int, int]] = []
        for key_start, key_end in segment.key_ranges:
            if not (previous_key_end <= key_start < key_end <= key_len):
                raise ValueError("Attention key ranges must be ordered, non-overlapping, and in bounds.")
            if merged_key_ranges and merged_key_ranges[-1][1] == key_start:
                merged_key_ranges[-1] = (merged_key_ranges[-1][0], key_end)
            else:
                merged_key_ranges.append((key_start, key_end))
            previous_key_end = key_end
        normalized_segments.append(
            AttentionSegment(
                query_start=segment.query_start,
                query_end=segment.query_end,
                key_ranges=tuple(merged_key_ranges),
            )
        )
        cursor = segment.query_end
    if cursor != query_len:
        raise ValueError("Attention segments must cover every query row.")

    normalized_segments = tuple(normalized_segments)
    dense = torch.zeros((query_len, key_len), dtype=torch.bool, device=device)
    for segment in normalized_segments:
        for key_start, key_end in segment.key_ranges:
            dense[segment.query_start : segment.query_end, key_start:key_end] = True
    return StructuredAttentionMask(dense=dense, segments=normalized_segments)


def dense_attention_mask(
    mask: Optional[torch.Tensor | StructuredAttentionMask | KeyPaddingMask],
) -> Optional[torch.Tensor]:
    if isinstance(mask, StructuredAttentionMask):
        return mask.dense
    if isinstance(mask, KeyPaddingMask):
        return mask.valid[:, None, None, :]
    return mask


def elide_fully_valid_attention_mask(
    mask: Optional[torch.Tensor | StructuredAttentionMask | KeyPaddingMask],
) -> Optional[torch.Tensor | StructuredAttentionMask | KeyPaddingMask]:
    """Drop a boolean all-True mask because it imposes no attention constraint.

    Call this once while preparing an attention payload rather than once per
    transformer layer. Besides avoiding an unnecessary mask, this lets external
    FlashAttention kernels handle the otherwise-unmasked operation.
    """
    if mask is None:
        return mask
    if isinstance(mask, StructuredAttentionMask):
        return None if mask.is_fully_valid else mask
    if isinstance(mask, KeyPaddingMask):
        return None if mask.is_fully_valid else mask
    # Never inspect a CUDA tensor from Python here: `.item()` would serialize the
    # host with every denoising step. Common callers normalize masks while they
    # are still on CPU; externally supplied CUDA masks stay explicit.
    if (
        mask.device.type == "cpu"
        and mask.dtype == torch.bool
        and mask.numel() > 0
        and bool(mask.all().item())
    ):
        return None
    return mask


def _load_flash_kernel(backend: str) -> Callable:
    if backend in _FLASH_KERNELS:
        return _FLASH_KERNELS[backend]
    try:
        if backend == "fa2":
            kernel = import_module("flash_attn").flash_attn_func
        elif backend == "fa3":
            try:
                module = import_module("flash_attn_interface")
            except ImportError:
                module = import_module("flash_attn_3.flash_attn_interface")
            kernel = module.flash_attn_func
        elif backend == "fa4":
            kernel = import_module("flash_attn.cute").flash_attn_func
        else:
            raise ValueError(f"No external FlashAttention kernel for backend: {backend}")
    except (ImportError, AttributeError) as exc:
        package = {"fa2": "flash-attn", "fa3": "flash_attn_interface", "fa4": "flash-attn-4"}[backend]
        raise ImportError(
            f"attention_backend={backend!r} requires {package} and its flash_attn_func API."
        ) from exc
    _FLASH_KERNELS[backend] = kernel
    return kernel


def _load_flash_varlen_kernel(backend: str) -> Callable:
    if backend in _FLASH_VARLEN_KERNELS:
        return _FLASH_VARLEN_KERNELS[backend]
    try:
        if backend == "fa2":
            kernel = import_module("flash_attn").flash_attn_varlen_func
        elif backend == "fa3":
            try:
                module = import_module("flash_attn_interface")
            except ImportError:
                module = import_module("flash_attn_3.flash_attn_interface")
            kernel = module.flash_attn_varlen_func
        elif backend == "fa4":
            kernel = import_module("flash_attn.cute").flash_attn_varlen_func
        else:
            raise ValueError(f"No external variable-length kernel for backend: {backend}")
    except (ImportError, AttributeError) as exc:
        package = {"fa2": "flash-attn", "fa3": "flash_attn_interface", "fa4": "flash-attn-4"}[backend]
        raise ImportError(f"{package} does not provide flash_attn_varlen_func.") from exc
    _FLASH_VARLEN_KERNELS[backend] = kernel
    return kernel


def _external_flash_eligible(q: torch.Tensor) -> bool:
    return (
        q.device.type == "cuda"
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] <= 256
    )


def _resolve_backend(requested: str, q: torch.Tensor) -> str:
    requested = normalize_attention_backend(requested)
    if requested == "sdpa":
        return "sdpa"
    if not _external_flash_eligible(q):
        return "sdpa"
    if requested != "auto":
        _load_flash_kernel(requested)
        return requested
    for candidate in AUTO_ATTENTION_BACKENDS[:-1]:
        try:
            _load_flash_kernel(candidate)
            return candidate
        except ImportError:
            continue
    return "sdpa"


def _log_selection(requested: str, selected: str) -> None:
    key = (requested, selected)
    if key in _LOGGED_SELECTIONS:
        return
    _LOGGED_SELECTIONS.add(key)
    logger.info("Attention backend selected: requested=%s selected=%s", requested, selected)


def _call_external_flash(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    kernel = _load_flash_kernel(backend)
    if backend == "fa2":
        output = kernel(q, k, v, dropout_p=0.0, causal=False)
    else:
        output = kernel(q, k, v, causal=False)
    if isinstance(output, tuple):
        output = output[0]
    return output


def _call_external_flash_varlen(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: KeyPaddingMask,
) -> torch.Tensor:
    batch_size, query_len, num_heads, head_dim = q.shape
    if mask.shape != k.shape[:2]:
        raise ValueError(
            f"Key padding mask shape must match K/V [B, K], got {tuple(mask.shape)} "
            f"and {tuple(k.shape[:2])}."
        )
    packed_q = q.reshape(batch_size * query_len, num_heads, head_dim)
    flat_k = k.reshape(-1, num_heads, head_dim)
    flat_v = v.reshape(-1, num_heads, head_dim)
    packed_k = flat_k.index_select(0, mask.indices)
    packed_v = flat_v.index_select(0, mask.indices)
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * query_len,
        query_len,
        dtype=torch.int32,
        device=q.device,
    )
    kernel = _load_flash_varlen_kernel(backend)
    if backend == "fa2":
        output = kernel(
            packed_q,
            packed_k,
            packed_v,
            cu_seqlens_q,
            mask.cu_seqlens,
            query_len,
            mask.max_seqlen,
            dropout_p=0.0,
            causal=False,
        )
    elif backend == "fa3":
        output = kernel(
            packed_q,
            packed_k,
            packed_v,
            cu_seqlens_q,
            mask.cu_seqlens,
            query_len,
            mask.max_seqlen,
            causal=False,
        )
    else:
        output = kernel(
            packed_q,
            packed_k,
            packed_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=mask.cu_seqlens,
            max_seqlen_q=query_len,
            max_seqlen_k=mask.max_seqlen,
            causal=False,
        )
    if isinstance(output, tuple):
        output = output[0]
    return output.reshape(batch_size, query_len, num_heads, -1)


def _segmented_flash_attention(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: StructuredAttentionMask,
) -> torch.Tensor:
    outputs = []
    for segment in mask.segments:
        q_segment = q[:, segment.query_start : segment.query_end]
        if not segment.key_ranges:
            outputs.append(torch.zeros_like(q_segment))
            continue
        first_start, last_end = segment.key_ranges[0][0], segment.key_ranges[-1][1]
        ranges_are_contiguous = all(
            previous[1] == current[0]
            for previous, current in zip(segment.key_ranges, segment.key_ranges[1:])
        )
        if ranges_are_contiguous:
            k_segment = k[:, first_start:last_end]
            v_segment = v[:, first_start:last_end]
        else:
            k_segment = torch.cat(
                [k[:, start:end] for start, end in segment.key_ranges], dim=1
            )
            v_segment = torch.cat(
                [v[:, start:end] for start, end in segment.key_ranges], dim=1
            )
        outputs.append(_call_external_flash(backend, q_segment, k_segment, v_segment))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor | StructuredAttentionMask | KeyPaddingMask],
) -> torch.Tensor:
    mask = dense_attention_mask(attention_mask)
    if mask is not None:
        if mask.device != q.device:
            mask = mask.to(device=q.device)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4:
            raise ValueError(f"Attention mask must be 2D/3D/4D, got shape {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=q.dtype)
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
        dropout_p=0.0,
    )
    return output.transpose(1, 2).contiguous()


def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    attention_mask: Optional[torch.Tensor | StructuredAttentionMask | KeyPaddingMask] = None,
    backend: str = "sdpa",
) -> torch.Tensor:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must be [B, S, H*D] tensors.")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("q/k/v batch and hidden dimensions must match.")
    if q.shape[2] % num_heads != 0:
        raise ValueError(f"Attention width {q.shape[2]} is not divisible by num_heads={num_heads}.")
    if isinstance(attention_mask, KeyPaddingMask):
        if attention_mask.is_fully_valid:
            attention_mask = None
        elif attention_mask.valid.device != q.device:
            attention_mask = attention_mask.to(device=q.device)

    head_dim = q.shape[2] // num_heads
    q_heads = q.reshape(q.shape[0], q.shape[1], num_heads, head_dim)
    k_heads = k.reshape(k.shape[0], k.shape[1], num_heads, head_dim)
    v_heads = v.reshape(v.shape[0], v.shape[1], num_heads, head_dim)
    requested = normalize_attention_backend(backend)
    selected = _resolve_backend(requested, q_heads)
    use_external = selected != "sdpa" and (
        attention_mask is None
        or isinstance(attention_mask, (StructuredAttentionMask, KeyPaddingMask))
    )
    if use_external and isinstance(attention_mask, KeyPaddingMask):
        try:
            _load_flash_varlen_kernel(selected)
        except ImportError:
            use_external = False
    execution_backend = selected if use_external else "sdpa"
    if use_external and isinstance(attention_mask, KeyPaddingMask):
        execution_backend = f"{selected}-varlen"
    _log_selection(requested, execution_backend)

    if use_external and attention_mask is None:
        output = _call_external_flash(selected, q_heads, k_heads, v_heads)
    elif use_external and isinstance(attention_mask, KeyPaddingMask):
        output = _call_external_flash_varlen(
            selected, q_heads, k_heads, v_heads, attention_mask
        )
    elif use_external and isinstance(attention_mask, StructuredAttentionMask):
        output = _segmented_flash_attention(selected, q_heads, k_heads, v_heads, attention_mask)
    else:
        output = _sdpa_attention(q_heads, k_heads, v_heads, attention_mask)
    return output.reshape(output.shape[0], output.shape[1], -1)

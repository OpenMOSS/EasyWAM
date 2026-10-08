import torch
from einops import rearrange


RopeFactors = tuple[torch.Tensor, torch.Tensor]


def prepare_rope(
    freqs: torch.Tensor | RopeFactors, device: torch.device, dtype: torch.dtype
) -> RopeFactors:
    if isinstance(freqs, tuple):
        return tuple(value.to(device=device, dtype=dtype) for value in freqs)
    freqs = freqs.to(device=device)
    return freqs.real.to(dtype=dtype), freqs.imag.to(dtype=dtype)


def rope_apply(
    x: torch.Tensor, freqs: torch.Tensor | RopeFactors, num_heads: int
) -> torch.Tensor:
    return apply_prepared_rope(x, prepare_rope(freqs, x.device, x.dtype), num_heads)


def apply_prepared_rope(
    x: torch.Tensor, rope: RopeFactors, num_heads: int
) -> torch.Tensor:
    xh = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x1, x2 = xh.reshape(*xh.shape[:-1], -1, 2).unbind(-1)
    cos, sin = rope
    out = torch.stack(
        (x1 * cos - x2 * sin, x1 * sin + x2 * cos),
        dim=-1,
    )
    return out.flatten(-2).flatten(2)

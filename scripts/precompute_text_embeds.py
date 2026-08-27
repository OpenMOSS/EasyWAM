import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from data.lerobot.robot_video_dataset import DEFAULT_PROMPT
from data.lerobot.text_embedding_cache import (
    aggregate_cache_filename,
    build_aggregate_payload,
    prompt_hash,
    validate_aggregate_payload,
)
from model.wan22.helpers.loader import _load_registered_model, _resolve_configs
from model.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from utils.config_resolvers import register_default_resolvers
from utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "./checkpoints/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "./checkpoints/umt5-xxl"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _iter_dataset_nodes(node: Any, path: str = "data"):
    if isinstance(node, DictConfig):
        if "dataset_dirs" in node and node.get("dataset_dirs") is not None:
            yield path, node
        for key, value in node.items():
            yield from _iter_dataset_nodes(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for idx, value in enumerate(node):
            yield from _iter_dataset_nodes(value, f"{path}[{idx}]")


def _collect_dataset_settings(data_cfg: DictConfig):
    dataset_dirs: list[str] = []
    cache_dirs: list[Path] = []
    context_lens = set()

    for node_path, node in _iter_dataset_nodes(data_cfg, path="data"):
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue

        cache_dir = node.get("text_embedding_cache_dir")
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(
                f"Missing `text_embedding_cache_dir` for dataset node `{node_path}` "
                "(this node defines `dataset_dirs`)."
            )

        for ds in raw_dirs:
            ds_str = str(ds)
            if ds_str not in dataset_dirs:
                dataset_dirs.append(ds_str)

        cache_dir_path = Path(str(cache_dir)).expanduser()
        if cache_dir_path not in cache_dirs:
            cache_dirs.append(cache_dir_path)

        context_len = node.get("context_len")
        if context_len is not None:
            context_lens.add(int(context_len))

        logger.info("Discovered dataset node `%s` with %d dataset_dirs.", node_path, len(raw_dirs))

    return dataset_dirs, cache_dirs, context_lens


def _resolve_context_len(context_lens: set[int]) -> int:
    if len(context_lens) != 1:
        raise ValueError(
            f"Found multiple context_len values in data config: {sorted(context_lens)}. "
            "Please keep them consistent."
        )
    return next(iter(context_lens))


def _read_unique_prompts(dataset_dirs: list[str]) -> list[str]:
    prompts: list[str] = []
    seen = set()
    total_task_rows = 0

    for ds_dir in dataset_dirs:
        tasks_path = Path(ds_dir) / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Missing tasks file: {tasks_path}")

        with tasks_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "task" not in record:
                    raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
                task = str(record["task"])
                prompt = DEFAULT_PROMPT.format(task=task)
                total_task_rows += 1
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)

    logger.info(
        "Loaded %d task rows from %d datasets, deduplicated to %d prompts.",
        total_task_rows,
        len(dataset_dirs),
        len(prompts),
    )
    return prompts


def _get_override_prompt(override_instruction: Any) -> str | None:
    if override_instruction is None:
        return None
    task = str(override_instruction).strip()
    if task == "":
        return None
    return DEFAULT_PROMPT.format(task=task)


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _atomic_torch_save(payload: dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _load_existing_aggregate(
    cache_path: Path,
    *,
    context_len: int,
    encoder_id: str,
) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    try:
        validate_aggregate_payload(
            payload,
            expected_context_len=context_len,
            expected_encoder_id=encoder_id,
        )
    except Exception as exc:
        raise ValueError(
            f"Existing aggregate cache is incompatible: {cache_path}. "
            "Run with overwrite=true to rebuild it."
        ) from exc
    return payload


def _payload_to_rows(payload: dict[str, Any] | None) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if payload is None:
        return {}
    return {
        hashed: (payload["contexts"][index], payload["masks"][index])
        for index, hashed in enumerate(payload["prompt_hashes"])
    }


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed enabled: world_size=%d", world_size)
    if (not is_distributed) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            "Multi-GPU available. To use it, run: torchrun --standalone --nproc_per_node=%d scripts/precompute_text_embeds.py",
            torch.cuda.device_count(),
        )

    overwrite = _to_bool(cfg.get("overwrite", True))
    model_cfg = cfg.model
    if model_cfg is None:
        raise ValueError("`cfg.model` is required.")
    if cfg.data is None:
        raise ValueError("`cfg.data` is required.")

    dataset_dirs, cache_dirs, context_lens = _collect_dataset_settings(cfg.data)
    if not cache_dirs:
        raise ValueError("No `text_embedding_cache_dir` found under `cfg.data`.")

    context_len = _resolve_context_len(context_lens)
    override_prompt = _get_override_prompt(cfg.get("override_instruction"))
    if override_prompt is not None:
        prompts = [override_prompt]
        logger.info("Using override_instruction; skipping dataset scan and encoding exactly 1 prompt.")
    else:
        if not dataset_dirs:
            raise ValueError("No `dataset_dirs` found under `cfg.data`.")
        prompts = _read_unique_prompts(dataset_dirs)
    if not prompts:
        logger.warning("No prompts found from tasks.jsonl; nothing to do.")
        return

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
    enc_id = _model_id_to_enc_id(model_id)

    logger.info(
        "Preparing text encoder with model_id=%s tokenizer_model_id=%s device=%s dtype=%s context_len=%d overwrite=%s",
        model_id,
        tokenizer_model_id,
        device,
        torch_dtype,
        context_len,
        overwrite,
    )

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
    )
    text_config.resolve()
    tokenizer_config.resolve()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )

    cache_filename = aggregate_cache_filename(context_len, enc_id)
    existing_payloads: dict[str, dict[str, Any] | None] = {}
    cached_everywhere: set[str] = set()
    if rank == 0:
        for cache_dir in cache_dirs:
            cache_path = cache_dir / cache_filename
            existing_payloads[str(cache_dir)] = (
                None
                if overwrite
                else _load_existing_aggregate(
                    cache_path,
                    context_len=context_len,
                    encoder_id=enc_id,
                )
            )
        if not overwrite:
            existing_hash_sets = [
                set(payload["prompt_hashes"]) if payload is not None else set()
                for payload in existing_payloads.values()
            ]
            if existing_hash_sets:
                cached_everywhere = set.intersection(*existing_hash_sets)

    if is_distributed:
        cached_obj = [cached_everywhere if rank == 0 else None]
        dist.broadcast_object_list(cached_obj, src=0)
        cached_everywhere = set(cached_obj[0])

    required_prompts = list(prompts)
    prompts_to_encode = [
        prompt for prompt in required_prompts if prompt_hash(prompt) not in cached_everywhere
    ]
    local_prompts = prompts_to_encode[rank::world_size] if is_distributed else prompts_to_encode
    if rank == 0:
        logger.info(
            "Aggregate cache: required=%d cached=%d to_encode=%d overwrite=%s",
            len(required_prompts),
            len(required_prompts) - len(prompts_to_encode),
            len(prompts_to_encode),
            overwrite,
        )

    over_length_prompts = 0
    local_hashes: list[str] = []
    local_contexts: list[torch.Tensor] = []
    local_masks: list[torch.Tensor] = []
    with tqdm(
        total=len(local_prompts),
        desc=f"Encoding prompts (rank {rank}/{world_size})" if is_distributed else "Encoding prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(local_prompts), DEFAULT_BATCH_SIZE):
                batch_prompts = local_prompts[start : start + DEFAULT_BATCH_SIZE]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                over_length_prompts += int(mask.all(dim=1).sum().item())
                context = text_encoder(ids, mask)

                for i, prompt in enumerate(batch_prompts):
                    context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                    context_i[~mask_i] = 0
                    mask_i = torch.ones_like(mask_i)
                    local_hashes.append(prompt_hash(prompt))
                    local_contexts.append(context_i)
                    local_masks.append(mask_i)

                pbar.update(len(batch_prompts))

    run_token_obj = [uuid.uuid4().hex if rank == 0 else None]
    if is_distributed:
        dist.broadcast_object_list(run_token_obj, src=0)
    run_token = str(run_token_obj[0])

    if local_hashes:
        local_shard = {
            "prompt_hashes": local_hashes,
            "contexts": torch.stack(local_contexts, dim=0),
            "masks": torch.stack(local_masks, dim=0),
        }
        for cache_dir in cache_dirs:
            shard_path = cache_dir / f".{cache_filename}.{run_token}.rank{rank}.tmp"
            _atomic_torch_save(local_shard, shard_path)

    if is_distributed:
        dist.barrier()

    if rank == 0:
        required_hashes = [prompt_hash(prompt) for prompt in required_prompts]
        for cache_dir in cache_dirs:
            cache_path = cache_dir / cache_filename
            rows = _payload_to_rows(existing_payloads.get(str(cache_dir)))
            shard_paths = [
                cache_dir / f".{cache_filename}.{run_token}.rank{shard_rank}.tmp"
                for shard_rank in range(world_size)
            ]
            for shard_path in shard_paths:
                if not shard_path.is_file():
                    continue
                shard = torch.load(shard_path, map_location="cpu", weights_only=True)
                for index, hashed in enumerate(shard["prompt_hashes"]):
                    rows[hashed] = (shard["contexts"][index], shard["masks"][index])

            missing_hashes = [hashed for hashed in required_hashes if hashed not in rows]
            if missing_hashes:
                raise RuntimeError(
                    f"Failed to assemble aggregate cache {cache_path}; "
                    f"missing {len(missing_hashes)} prompts."
                )
            contexts = torch.stack([rows[hashed][0] for hashed in required_hashes], dim=0)
            masks = torch.stack([rows[hashed][1] for hashed in required_hashes], dim=0)
            payload = build_aggregate_payload(
                prompt_hashes=required_hashes,
                contexts=contexts,
                masks=masks,
                context_len=context_len,
                encoder_id=enc_id,
            )
            _atomic_torch_save(payload, cache_path)
            for shard_path in shard_paths:
                if shard_path.exists():
                    shard_path.unlink()
            logger.info(
                "Wrote aggregate text embedding cache: path=%s prompts=%d size=%.2f MiB",
                cache_path,
                len(required_hashes),
                cache_path.stat().st_size / (1024 ** 2),
            )

    over_length_global = over_length_prompts
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        over_tensor = torch.tensor([over_length_prompts], device=reduce_device, dtype=torch.long)
        dist.all_reduce(over_tensor, op=dist.ReduceOp.SUM)
        over_length_global = int(over_tensor.item())

    if (not is_distributed) or rank == 0:
        logger.info("Finished precomputing text embeddings.")
        logger.info(
            "Over-length prompts (mask all True, i.e. no padding after truncation/max_length=%d): %d/%d",
            context_len,
            over_length_global,
            len(prompts_to_encode),
        )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

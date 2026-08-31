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
    build_text_embedding_payload,
    prompt_hash,
    text_embedding_cache_filename,
)
from model.backbone.wan22.loader import _load_registered_model, _resolve_configs
from model.backbone.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from utils.config_resolvers import register_default_resolvers
from utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "./checkpoints/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "./checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl"
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
    backbone_cfg = model_cfg.get("backbone")
    if backbone_cfg is None:
        raise ValueError("`cfg.model.backbone` is required.")
    backbone_name = str(backbone_cfg.get("name", "")).lower()
    model_id = str(backbone_cfg.get("model_id", DEFAULT_MODEL_ID))
    enc_id = str(backbone_cfg.get("text_encoder_id", _model_id_to_enc_id(model_id)))

    if backbone_name == "wan22":
        text_encoder_model_id = model_id
        tokenizer_model_id = str(
            backbone_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID)
        )
    elif backbone_name == "cosmos25":
        reason_model_id = backbone_cfg.get("reason_model_id")
        if not reason_model_id:
            raise ValueError("Cosmos25 text caching requires `reason_model_id`.")
        # Cosmos-Reason bundles the matching tokenizer with the text encoder.
        text_encoder_model_id = str(reason_model_id)
        tokenizer_model_id = str(reason_model_id)
    else:
        raise ValueError(f"Unsupported backbone for text caching: {backbone_name!r}")

    logger.info(
        "Preparing text encoder for backbone=%s with text_encoder_model_id=%s "
        "tokenizer_model_id=%s device=%s dtype=%s context_len=%d overwrite=%s",
        backbone_name,
        text_encoder_model_id,
        tokenizer_model_id,
        device,
        torch_dtype,
        context_len,
        overwrite,
    )

    tokenizer = None
    if backbone_name == "wan22":
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
    elif backbone_name == "cosmos25":
        from model.backbone.cosmos25.cosmos_video_text_encoder import Cosmos25TextEncoder
        from model.backbone.cosmos25.loader import (
            load_cosmos25_text_projection,
            resolve_cosmos25_dit_path,
        )

        text_encoder = Cosmos25TextEncoder.from_pretrained(
            text_encoder_model_id, torch_dtype=torch_dtype, max_length=context_len
        ).to(device).eval()
        projection = load_cosmos25_text_projection(
            resolve_cosmos25_dit_path(model_id),
            device=device,
            torch_dtype=torch_dtype,
        )
        text_encoder.set_projector(projection)

    stats = {
        str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0}
        for cache_dir in cache_dirs
    }
    local_prompts = prompts[rank::world_size] if is_distributed else prompts
    fully_cached_local = 0
    if not overwrite:
        prompts_to_encode: list[str] = []
        for prompt in local_prompts:
            hashed = prompt_hash(prompt)
            filename = text_embedding_cache_filename(
                hashed,
                context_len,
                enc_id,
                is_hash=True,
            )
            if all((cache_dir / filename).is_file() for cache_dir in cache_dirs):
                fully_cached_local += 1
                for cache_dir in cache_dirs:
                    stats[str(cache_dir)]["skip"] += 1
            else:
                prompts_to_encode.append(prompt)
        local_prompts = prompts_to_encode

    prompts_encoded_local = len(local_prompts)
    prompts_encoded_global = prompts_encoded_local
    fully_cached_global = fully_cached_local
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        count_tensor = torch.tensor(
            [prompts_encoded_local, fully_cached_local],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        prompts_encoded_global = int(count_tensor[0].item())
        fully_cached_global = int(count_tensor[1].item())
    if (not is_distributed) or rank == 0:
        logger.info(
            "Per-prompt cache: required=%d cached=%d to_encode=%d overwrite=%s",
            len(prompts),
            fully_cached_global,
            prompts_encoded_global,
            overwrite,
        )

    over_length_prompts = 0
    with tqdm(
        total=len(local_prompts),
        desc=f"Encoding prompts (rank {rank}/{world_size})" if is_distributed else "Encoding prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            encode_batch_size = 1 if backbone_name == "cosmos25" else DEFAULT_BATCH_SIZE
            for start in range(0, len(local_prompts), encode_batch_size):
                batch_prompts = local_prompts[start : start + encode_batch_size]
                if tokenizer is None:
                    if backbone_name == "cosmos25":
                        context, mask, token_mask = text_encoder(
                            batch_prompts, return_token_mask=True
                        )
                    else:
                        context, mask = text_encoder(batch_prompts)
                        token_mask = mask
                    mask = mask.to(device=device, dtype=torch.bool)
                else:
                    ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                    ids = ids.to(device)
                    mask = mask.to(device=device, dtype=torch.bool)
                    context = text_encoder(ids, mask)
                    token_mask = mask
                over_length_prompts += int(token_mask.all(dim=1).sum().item())

                for i, prompt in enumerate(batch_prompts):
                    hashed = prompt_hash(prompt)
                    context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                    if backbone_name != "cosmos25":
                        context_i[~mask_i] = 0
                        mask_i = torch.ones_like(mask_i)
                    payload = build_text_embedding_payload(
                        context=context_i,
                        mask=mask_i,
                        context_len=context_len,
                        encoder_id=enc_id,
                        prompt_digest=hashed,
                    )
                    filename = text_embedding_cache_filename(
                        hashed,
                        context_len,
                        enc_id,
                        is_hash=True,
                    )
                    for cache_dir in cache_dirs:
                        cache_path = cache_dir / filename
                        key = str(cache_dir)
                        if cache_path.exists() and not overwrite:
                            stats[key]["skip"] += 1
                            continue
                        if cache_path.exists():
                            stats[key]["overwrite"] += 1
                        else:
                            stats[key]["new"] += 1
                        _atomic_torch_save(payload, cache_path)

                pbar.update(len(batch_prompts))

    over_length_global = over_length_prompts
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        over_tensor = torch.tensor([over_length_prompts], device=reduce_device, dtype=torch.long)
        dist.all_reduce(over_tensor, op=dist.ReduceOp.SUM)
        over_length_global = int(over_tensor.item())
        counts_tensor = torch.tensor(
            [
                [stats[str(cache_dir)]["new"], stats[str(cache_dir)]["overwrite"], stats[str(cache_dir)]["skip"]]
                for cache_dir in cache_dirs
            ],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            for index, cache_dir in enumerate(cache_dirs):
                key = str(cache_dir)
                stats[key]["new"] = int(counts_tensor[index, 0].item())
                stats[key]["overwrite"] = int(counts_tensor[index, 1].item())
                stats[key]["skip"] = int(counts_tensor[index, 2].item())

    if (not is_distributed) or rank == 0:
        logger.info("Finished precomputing text embeddings.")
        logger.info(
            "Over-length prompts (mask all True, i.e. no padding after truncation/max_length=%d): %d/%d",
            context_len,
            over_length_global,
            prompts_encoded_global,
        )
        for cache_dir in cache_dirs:
            key = str(cache_dir)
            logger.info(
                "Cache dir: %s | new=%d overwrite=%d skip=%d",
                key,
                stats[key]["new"],
                stats[key]["overwrite"],
                stats[key]["skip"],
            )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

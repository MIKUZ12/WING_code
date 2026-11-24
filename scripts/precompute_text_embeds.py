import hashlib
import json
import logging
import os
import re
import struct
import uuid
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from wing.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wing.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from wing.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from wing.utils.config_resolvers import register_default_resolvers
from wing.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_PACKED_BATCH_SIZE = 256
DEFAULT_PACKED_SHARD_SIZE = 8192
PACKED_SCHEMA = "wing_text_context_packed_v1"
BF16_ELEMENT_SIZE = 2
U64_ELEMENT_SIZE = 8


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


def _read_indexed_prompts(dataset_dir: str) -> list[str]:
    """Read every task in task_index order; packed caches are not deduplicated."""
    tasks_path = Path(dataset_dir) / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing tasks file: {tasks_path}")

    prompts: list[str] = []
    with tasks_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "task_index" not in record:
                raise KeyError(f"Missing `task_index` field at {tasks_path}:{line_idx}")
            task_index = record["task_index"]
            expected_index = len(prompts)
            if type(task_index) is not int or task_index != expected_index:
                raise ValueError(
                    f"Expected task_index={expected_index} at {tasks_path}:{line_idx}, "
                    f"got {task_index!r}. Packed caches require contiguous task_index 0..N-1."
                )
            if "task" not in record:
                raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
            prompts.append(DEFAULT_PROMPT.format(task=str(record["task"])))

    logger.info("Loaded %d indexed task prompts from %s.", len(prompts), tasks_path)
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


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _atomic_write_text(text: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, output_path)


def _atomic_json_save(payload: dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _packed_shard_paths(cache_dir: Path, shard_id: int) -> tuple[Path, Path, Path]:
    prefix = f"shard_{shard_id:06d}"
    shard_dir = cache_dir / "packed_shards"
    return (
        shard_dir / f"{prefix}.contexts.bf16",
        shard_dir / f"{prefix}.offsets.u64",
        shard_dir / f"{prefix}.complete.json",
    )


def _validate_packed_shard(
    cache_dir: Path,
    shard_id: int,
    shard_start: int,
    shard_end: int,
    context_len: int,
    hidden_dim: int,
) -> dict[str, Any] | None:
    context_path, offsets_path, complete_path = _packed_shard_paths(cache_dir, shard_id)
    if not complete_path.exists():
        return None

    try:
        with complete_path.open("r", encoding="utf-8") as f:
            marker = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Invalid packed shard marker: {complete_path}") from exc

    record_count = shard_end - shard_start
    expected_marker = {
        "schema": PACKED_SCHEMA,
        "shard_id": shard_id,
        "start": shard_start,
        "end": shard_end,
        "record_count": record_count,
        "context_len": context_len,
        "hidden_dim": hidden_dim,
        "dtype": "bfloat16",
    }
    for key, expected in expected_marker.items():
        if marker.get(key) != expected:
            raise RuntimeError(
                f"Packed shard marker mismatch at {complete_path}: "
                f"{key}={marker.get(key)!r}, expected {expected!r}"
            )

    token_count = marker.get("token_count")
    if type(token_count) is not int or token_count < 0 or token_count > record_count * context_len:
        raise RuntimeError(f"Invalid token_count in packed shard marker: {complete_path}")
    expected_context_bytes = token_count * hidden_dim * BF16_ELEMENT_SIZE
    expected_offsets_bytes = (record_count + 1) * U64_ELEMENT_SIZE
    if marker.get("context_bytes") != expected_context_bytes:
        raise RuntimeError(f"Invalid context_bytes in packed shard marker: {complete_path}")
    if marker.get("offsets_bytes") != expected_offsets_bytes:
        raise RuntimeError(f"Invalid offsets_bytes in packed shard marker: {complete_path}")
    if not context_path.is_file() or context_path.stat().st_size != expected_context_bytes:
        raise RuntimeError(f"Packed context file size mismatch: {context_path}")
    if not offsets_path.is_file() or offsets_path.stat().st_size != expected_offsets_bytes:
        raise RuntimeError(f"Packed offsets file size mismatch: {offsets_path}")

    with offsets_path.open("rb") as f:
        offsets_raw = f.read()
    offsets = struct.unpack(f"<{record_count + 1}Q", offsets_raw)
    if offsets[0] != 0 or offsets[-1] != token_count:
        raise RuntimeError(f"Invalid boundary offsets in packed shard: {offsets_path}")
    if any(left > right or right - left > context_len for left, right in zip(offsets, offsets[1:])):
        raise RuntimeError(f"Invalid per-record offsets in packed shard: {offsets_path}")
    return marker


def _write_packed_shard(
    *,
    prompts: list[str],
    cache_dir: Path,
    shard_id: int,
    shard_start: int,
    shard_end: int,
    shard_size: int,
    text_batch_size: int,
    context_len: int,
    hidden_dim: int,
    tokenizer: HuggingfaceTokenizer,
    text_encoder: torch.nn.Module,
    device: str,
) -> tuple[bool, int]:
    marker = _validate_packed_shard(
        cache_dir, shard_id, shard_start, shard_end, context_len, hidden_dim
    )
    if marker is not None:
        logger.info("Skipping complete packed shard %d (%d records).", shard_id, shard_end - shard_start)
        return False, 0

    context_path, offsets_path, complete_path = _packed_shard_paths(cache_dir, shard_id)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_suffix = f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
    context_tmp = context_path.parent / f".{context_path.name}{tmp_suffix}"
    offsets_tmp = offsets_path.parent / f".{offsets_path.name}{tmp_suffix}"
    offsets = [0]
    over_length = 0

    try:
        with context_tmp.open("wb") as context_file, torch.no_grad():
            for batch_start in range(shard_start, shard_end, text_batch_size):
                batch_end = min(batch_start + text_batch_size, shard_end)
                batch_prompts = prompts[batch_start:batch_end]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                mask_cpu = mask.to(device="cpu", dtype=torch.bool).contiguous()
                lengths = mask_cpu.sum(dim=1, dtype=torch.long)
                prefix_mask = torch.arange(context_len).unsqueeze(0) < lengths.unsqueeze(1)
                if not torch.equal(mask_cpu, prefix_mask):
                    raise RuntimeError(
                        f"Tokenizer mask is not a valid prefix in shard {shard_id}, batch {batch_start}."
                    )
                over_length += int((lengths == context_len).sum().item())

                ids = ids.to(device)
                mask_device = mask_cpu.to(device=device)
                context = text_encoder(ids, mask_device)
                if tuple(context.shape[:2]) != (len(batch_prompts), context_len):
                    raise RuntimeError(
                        f"Unexpected text context shape {tuple(context.shape)}; "
                        f"expected ({len(batch_prompts)}, {context_len}, {hidden_dim})."
                    )
                if context.shape[2] != hidden_dim:
                    raise RuntimeError(
                        f"Unexpected text hidden dimension {context.shape[2]}; expected {hidden_dim}."
                    )

                # One whole-batch D2H copy. Boolean packing happens only after it is on CPU.
                context_cpu = context.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                valid_context = context_cpu[mask_cpu].contiguous()
                valid_context.view(torch.uint8).numpy().tofile(context_file)

                for length in lengths.tolist():
                    offsets.append(offsets[-1] + int(length))

            context_file.flush()
            os.fsync(context_file.fileno())

        with offsets_tmp.open("wb") as offsets_file:
            offsets_file.write(struct.pack(f"<{len(offsets)}Q", *offsets))
            offsets_file.flush()
            os.fsync(offsets_file.fileno())

        record_count = shard_end - shard_start
        token_count = offsets[-1]
        marker = {
            "schema": PACKED_SCHEMA,
            "status": "complete",
            "shard_id": shard_id,
            "start": shard_start,
            "end": shard_end,
            "record_count": record_count,
            "token_count": token_count,
            "context_len": context_len,
            "hidden_dim": hidden_dim,
            "dtype": "bfloat16",
            "shard_size": shard_size,
            "context_bytes": token_count * hidden_dim * BF16_ELEMENT_SIZE,
            "offsets_bytes": (record_count + 1) * U64_ELEMENT_SIZE,
        }
        if context_tmp.stat().st_size != marker["context_bytes"]:
            raise RuntimeError(f"Packed context temp file size mismatch: {context_tmp}")
        if offsets_tmp.stat().st_size != marker["offsets_bytes"]:
            raise RuntimeError(f"Packed offsets temp file size mismatch: {offsets_tmp}")

        os.replace(context_tmp, context_path)
        os.replace(offsets_tmp, offsets_path)
        _atomic_json_save(marker, complete_path)
        _validate_packed_shard(cache_dir, shard_id, shard_start, shard_end, context_len, hidden_dim)
        return True, over_length
    finally:
        for tmp_path in (context_tmp, offsets_tmp):
            if tmp_path.exists():
                tmp_path.unlink()


def _run_packed_cache(
    *,
    prompts: list[str],
    dataset_dir: str,
    cache_dir: Path,
    context_len: int,
    shard_size: int,
    text_batch_size: int,
    model_id: str,
    tokenizer_model_id: str,
    tokenizer: HuggingfaceTokenizer,
    text_encoder: torch.nn.Module,
    device: str,
    is_distributed: bool,
    rank: int,
    world_size: int,
):
    hidden_dim = int(text_encoder.dim)
    task_count = len(prompts)
    shard_count = (task_count + shard_size - 1) // shard_size
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "packed_shards").mkdir(parents=True, exist_ok=True)

    owned_shards = [shard_id for shard_id in range(shard_count) if shard_id % world_size == rank]
    logger.info(
        "Packed cache: tasks=%d shards=%d shard_size=%d text_batch_size=%d rank=%d/%d owns=%d shards",
        task_count,
        shard_count,
        shard_size,
        text_batch_size,
        rank,
        world_size,
        len(owned_shards),
    )
    written_local = 0
    skipped_local = 0
    over_length_local = 0
    with tqdm(
        owned_shards,
        desc=f"Packed shards (rank {rank}/{world_size})" if is_distributed else "Packed shards",
        unit="shard",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        for shard_id in pbar:
            shard_start = shard_id * shard_size
            shard_end = min(shard_start + shard_size, task_count)
            written, over_length = _write_packed_shard(
                prompts=prompts,
                cache_dir=cache_dir,
                shard_id=shard_id,
                shard_start=shard_start,
                shard_end=shard_end,
                shard_size=shard_size,
                text_batch_size=text_batch_size,
                context_len=context_len,
                hidden_dim=hidden_dim,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                device=device,
            )
            written_local += int(written)
            skipped_local += int(not written)
            over_length_local += over_length

    counters = torch.tensor(
        [written_local, skipped_local, over_length_local],
        device=torch.device(device) if device.startswith("cuda") else torch.device("cpu"),
        dtype=torch.long,
    )
    if is_distributed:
        dist.all_reduce(counters, op=dist.ReduceOp.SUM)
        dist.barrier()

    if rank == 0:
        markers = []
        for shard_id in range(shard_count):
            shard_start = shard_id * shard_size
            shard_end = min(shard_start + shard_size, task_count)
            marker = _validate_packed_shard(
                cache_dir, shard_id, shard_start, shard_end, context_len, hidden_dim
            )
            if marker is None:
                raise RuntimeError(f"Missing complete packed shard {shard_id}/{shard_count}: {cache_dir}")
            markers.append(marker)

        manifest = {
            "schema": PACKED_SCHEMA,
            "schema_version": 1,
            "status": "complete",
            "task_count": task_count,
            "context_len": context_len,
            "hidden_dim": hidden_dim,
            "dtype": "bfloat16",
            "shard_size": shard_size,
            "shard_count": shard_count,
            "prompt": DEFAULT_PROMPT,
            "model": model_id,
            "tokenizer": tokenizer_model_id,
            "dataset_root": str(Path(dataset_dir).expanduser().resolve()),
        }
        _atomic_json_save(manifest, cache_dir / "packed_manifest.json")
        logger.info(
            "Finished packed text cache: verified=%d complete shards, written=%d skipped=%d "
            "over_length=%d/%d manifest=%s",
            len(markers),
            int(counters[0].item()),
            int(counters[1].item()),
            int(counters[2].item()),
            task_count,
            cache_dir / "packed_manifest.json",
        )

    if is_distributed:
        dist.barrier()


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
    cache_format = str(cfg.get("cache_format", "legacy")).strip().lower()
    if cache_format not in {"legacy", "files", "packed"}:
        raise ValueError(f"Unsupported cache_format={cache_format!r}; expected legacy, files, or packed.")

    if cache_format == "packed":
        if len(dataset_dirs) != 1 or len(cache_dirs) != 1:
            raise ValueError(
                "cache_format=packed requires exactly one dataset_dir and one text_embedding_cache_dir; "
                f"found {len(dataset_dirs)} and {len(cache_dirs)}."
            )
        if _get_override_prompt(cfg.get("override_instruction")) is not None:
            raise ValueError("cache_format=packed does not support override_instruction.")
        packed_shard_size = int(cfg.get("packed_shard_size", DEFAULT_PACKED_SHARD_SIZE))
        text_batch_size = int(cfg.get("text_batch_size", DEFAULT_PACKED_BATCH_SIZE))
        if packed_shard_size <= 0 or text_batch_size <= 0:
            raise ValueError("packed_shard_size and text_batch_size must both be positive.")
        prompts = _read_indexed_prompts(dataset_dirs[0])
    else:
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
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
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
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

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

    if cache_format == "packed":
        _run_packed_cache(
            prompts=prompts,
            dataset_dir=dataset_dirs[0],
            cache_dir=cache_dirs[0],
            context_len=context_len,
            shard_size=packed_shard_size,
            text_batch_size=text_batch_size,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            is_distributed=is_distributed,
            rank=rank,
            world_size=world_size,
        )
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        return

    stats = {
        str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0}
        for cache_dir in cache_dirs
    }

    prompts = prompts[rank::world_size] if is_distributed else prompts

    if not overwrite:
        fully_cached_local = 0
        prompts_to_encode: list[str] = []
        for prompt in prompts:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"
            fully_cached = True
            for cache_dir in cache_dirs:
                cache_path = cache_dir / filename
                if not cache_path.exists():
                    fully_cached = False
                    break
            if fully_cached:
                fully_cached_local += 1
                for cache_dir in cache_dirs:
                    stats[str(cache_dir)]["skip"] += 1
            else:
                prompts_to_encode.append(prompt)

        prompts = prompts_to_encode

        fully_cached_global = fully_cached_local
        to_encode_global = len(prompts)
        if is_distributed:
            reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
            count_tensor = torch.tensor([fully_cached_local, len(prompts)], device=reduce_device, dtype=torch.long)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            fully_cached_global = int(count_tensor[0].item())
            to_encode_global = int(count_tensor[1].item())

        if (not is_distributed) or rank == 0:
            logger.info(
                "overwrite=false: fully cached prompts=%d, prompts to encode=%d",
                fully_cached_global,
                to_encode_global,
            )

    logger.info("Writing caches to %d directories.", len(cache_dirs))
    prompts_encoded_local = len(prompts)
    prompts_encoded_global = prompts_encoded_local
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        count_tensor = torch.tensor([prompts_encoded_local], device=reduce_device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        prompts_encoded_global = int(count_tensor.item())

    over_length_prompts = 0
    with tqdm(
        total=len(prompts),
        desc=f"Encoding prompts (rank {rank}/{world_size})" if is_distributed else "Encoding prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(prompts), DEFAULT_BATCH_SIZE):
                batch_prompts = prompts[start : start + DEFAULT_BATCH_SIZE]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                over_length_prompts += int(mask.all(dim=1).sum().item())
                context = text_encoder(ids, mask)

                for i, prompt in enumerate(batch_prompts):
                    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                    payload = {
                        "context": context_i,
                        "mask": mask_i,
                    }

                    for cache_dir in cache_dirs:
                        cache_path = cache_dir / f"{hashed}.t5_len{context_len}.{enc_id}.pt"
                        key = str(cache_dir)
                        if cache_path.exists() and not overwrite:
                            stats[key]["skip"] += 1
                            continue

                        if cache_path.exists():
                            stats[key]["overwrite"] += 1
                        else:
                            stats[key]["new"] += 1

                        _atomic_torch_save(payload, cache_path)
                        _atomic_write_text(prompt, cache_dir / "prompt.txt")

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
            for idx, cache_dir in enumerate(cache_dirs):
                key = str(cache_dir)
                stats[key]["new"] = int(counts_tensor[idx, 0].item())
                stats[key]["overwrite"] = int(counts_tensor[idx, 1].item())
                stats[key]["skip"] = int(counts_tensor[idx, 2].item())

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


import logging
import json
import inspect
import os
import re
from contextlib import nullcontext
from collections.abc import Mapping
from math import ceil, cos, pi
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .models.wan22.wing import normalize_training_phase
from .utils.checkpoint_retention import (
    apply_checkpoint_prune,
    apply_checkpoint_state_prune,
    apply_checkpoint_weights_prune,
    plan_checkpoint_prune,
    plan_checkpoint_state_prune,
    plan_checkpoint_weights_prune,
)
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


def _validate_training_phase_eval(training_phase, eval_every: int) -> None:
    """Phase-I val batching is intentionally unsupported in the first prototype."""
    if training_phase == "phase1_ego_planner" and int(eval_every) != 0:
        raise ValueError(
            "phase1_ego_planner requires eval_every=0: the existing rollout evaluator "
            "does not preserve latent action targets or cached video latents."
        )


def _tensor_nonfinite_report(name: str, tensor: torch.Tensor) -> str | None:
    """Return a compact report when a floating tensor contains NaN or Inf."""
    if not torch.is_tensor(tensor) or not (tensor.is_floating_point() or tensor.is_complex()):
        return None
    data = tensor.detach()
    if data.is_complex():
        data = data.abs()
    finite_mask = torch.isfinite(data)
    if bool(finite_mask.all().item()):
        return None
    nonfinite = int((~finite_mask).sum().item())
    total = int(data.numel())
    finite_values = data[finite_mask]
    if finite_values.numel() > 0:
        min_value = float(finite_values.min().float().item())
        max_value = float(finite_values.max().float().item())
        mean_value = float(finite_values.float().mean().item())
        finite_summary = f"finite_min={min_value:.6g} finite_max={max_value:.6g} finite_mean={mean_value:.6g}"
    else:
        finite_summary = "no_finite_values"
    return (
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"nonfinite={nonfinite}/{total} {finite_summary}"
    )


def _eval_skips_rollout(val_dataset, action_head_mode: str) -> bool:
    """Whether trainer eval should skip the rollout/video/action-metric path and report val loss
    only. Skip when (a) val is multi-source (`MultiEmbodimentDataset` has per-group processors,
    which break the single `lerobot_dataset.processor` action-metric path) or (b) the model uses
    per_embodiment heads (infer/rollout doesn't pass `embodiment_id`, so any per_embodiment rollout
    raises -- including the single-source case that `isinstance(MultiEmbodimentDataset)` misses).
    This is a model capability guard: the repository currently has no maintained
    per-embodiment rollout/deployment recipe."""
    from .datasets.lerobot.multi_embodiment_dataset import MultiEmbodimentDataset
    return isinstance(val_dataset, MultiEmbodimentDataset) or action_head_mode == "per_embodiment"


def _eval_inference_proprio(model, proprio):
    """Return proprio only when the model has a proprio encoder.

    Human-video datasets retain a masked placeholder state for LeRobot schema
    compatibility. Training correctly ignores it when ``proprio_dim=None``;
    rollout evaluation must do the same instead of passing the placeholder to
    ``infer()``, whose explicit guard rejects proprio without an encoder.
    """
    return None if getattr(model, "proprio_dim", None) is None else proprio


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.backbone_learning_rate = float(
            cfg.get("backbone_learning_rate", self.learning_rate)
        )
        self.new_module_learning_rate = float(
            cfg.get("new_module_learning_rate", self.learning_rate)
        )
        if min(self.learning_rate, self.backbone_learning_rate, self.new_module_learning_rate) <= 0:
            raise ValueError("All configured learning rates must be positive.")
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.checkpoint_keep_last = int(cfg.get("checkpoint_keep_last", 0) or 0)
        if self.checkpoint_keep_last < 0:
            raise ValueError("checkpoint_keep_last must be >= 0 (0 disables pruning).")
        state_keep_last = cfg.get("checkpoint_state_keep_last", None)
        self.checkpoint_state_keep_last = (
            None if state_keep_last is None else int(state_keep_last)
        )
        if self.checkpoint_state_keep_last is not None:
            if self.checkpoint_state_keep_last < 1:
                raise ValueError("checkpoint_state_keep_last must be >= 1 when configured.")
            if self.checkpoint_keep_last < self.checkpoint_state_keep_last:
                raise ValueError(
                    "checkpoint_keep_last must be >= checkpoint_state_keep_last."
                )
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.optimizer_type = str(cfg.get("optimizer_type", "auto")).strip().lower()
        self.debug_nonfinite = bool(cfg.get("debug_nonfinite", False)) or os.environ.get(
            "WING_DEBUG_NONFINITE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.debug_nonfinite_max_reports = int(cfg.get("debug_nonfinite_max_reports", 12))
        self.freeze_video_expert = bool(cfg.get("freeze_video_expert", False))
        requested_phase = cfg.get("training_phase", None)
        if hasattr(self.model, "configure_training_phase"):
            self.model.configure_training_phase(requested_phase)
            self.training_phase = getattr(self.model, "training_phase", None)
        elif requested_phase is not None:
            raise ValueError("Configured training_phase requires a phase-aware model.")
        else:
            self.training_phase = None
        _validate_training_phase_eval(self.training_phase, self.eval_every)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.resume_allow_incomplete_state = bool(
            cfg.get("resume_allow_incomplete_state", False)
        )
        self.partial_resume = cfg.get("partial_resume", None)
        self.phase_init = cfg.get("phase_init", None)
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        tensorboard_cfg = cfg.get("tensorboard", {}) or {}
        self.tensorboard_enabled = bool(tensorboard_cfg.get("enabled", False))
        self.tensorboard_log_dir = str(
            tensorboard_cfg.get("log_dir", os.path.join(self.output_dir, "tensorboard"))
        )
        self.tensorboard_flush_secs = int(tensorboard_cfg.get("flush_secs", 30))
        if self.tensorboard_flush_secs < 1:
            raise ValueError("tensorboard.flush_secs must be >= 1.")

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        self._configure_deepspeed_gradient_clipping()

        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        deepspeed_config = deepspeed_plugin.deepspeed_config if deepspeed_plugin is not None else {}
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f ds_grad_clip=%s optimizer_type=%s",
            self.accelerator.distributed_type,
            deepspeed_config.get("zero_optimization", {}).get("stage", "disabled"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
            deepspeed_config.get("gradient_clipping"),
            self.optimizer_type,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Weight-only initialization must happen before optimizer/DeepSpeed setup.  ZeRO-1/2
        # clones independent FP32 master parameters during ``accelerator.prepare``; mutating only
        # the BF16 model afterwards leaves those masters stale, and the first optimizer step then
        # copies the stale values back over the loaded checkpoint weights.
        self._load_initial_weights_before_prepare()

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_training_phase_train_mode(
            self.model,
            training_phase=self.training_phase,
            freeze_video_expert=self.freeze_video_expert,
        )
        self._validate_spectral_training_scope(
            self.model, freeze_video_expert=self.freeze_video_expert
        )
        optimizer_groups = self._build_optimizer_param_groups(self.model)
        self.optimizer = self._build_optimizer(optimizer_groups)
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self.tensorboard_writer = None
        self._init_wandb()
        # Full Accelerate/DeepSpeed state directories include optimizer masters and therefore must
        # still be restored after ``prepare``.  Weight-only and partial checkpoints were loaded
        # above, before the optimizer existed.
        self._resume_training_state_after_prepare()
        self._init_tensorboard()
        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _collect_nonfinite_reports(self, value, *, prefix: str) -> list[str]:
        """Recursively collect NaN/Inf reports from nested batch/loss structures."""
        reports: list[str] = []
        if torch.is_tensor(value):
            report = _tensor_nonfinite_report(prefix, value)
            if report is not None:
                reports.append(report)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                reports.extend(self._collect_nonfinite_reports(item, prefix=f"{prefix}.{key}"))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                reports.extend(self._collect_nonfinite_reports(item, prefix=f"{prefix}[{index}]"))
        return reports

    def _raise_if_nonfinite_structure(self, value, *, stage: str) -> None:
        """Stop training at the first observed non-finite tensor in a nested structure."""
        if not self.debug_nonfinite:
            return
        reports = self._collect_nonfinite_reports(value, prefix=stage)
        if reports:
            shown = "\n".join(reports[: self.debug_nonfinite_max_reports])
            extra = len(reports) - self.debug_nonfinite_max_reports
            if extra > 0:
                shown += f"\n... {extra} more non-finite tensor(s) omitted"
            raise RuntimeError(f"Non-finite tensor detected during {stage}:\n{shown}")

    def _raise_if_nonfinite_named_parameters(self, *, stage: str) -> None:
        """Stop training if any trainable parameter has become NaN/Inf."""
        if not self.debug_nonfinite:
            return
        reports = []
        for name, param in self.accelerator.unwrap_model(self.model).named_parameters():
            if not param.requires_grad:
                continue
            report = _tensor_nonfinite_report(name, param)
            if report is not None:
                reports.append(report)
                if len(reports) >= self.debug_nonfinite_max_reports:
                    break
        if reports:
            raise RuntimeError(
                f"Non-finite trainable parameter detected {stage}:\n" + "\n".join(reports)
            )

    def _raise_if_nonfinite_gradients(self, *, stage: str) -> None:
        """Stop training if any trainable gradient has become NaN/Inf."""
        if not self.debug_nonfinite:
            return
        reports = []
        for name, param in self.accelerator.unwrap_model(self.model).named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            report = _tensor_nonfinite_report(f"{name}.grad", param.grad)
            if report is not None:
                reports.append(report)
                if len(reports) >= self.debug_nonfinite_max_reports:
                    break
        if reports:
            raise RuntimeError(f"Non-finite gradient detected {stage}:\n" + "\n".join(reports))

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _init_tensorboard(self):
        if not self.tensorboard_enabled or not self.accelerator.is_main_process:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoard logging is enabled in config "
                "(`tensorboard.enabled=true`) but tensorboard is not installed."
            ) from exc

        ensure_dir(self.tensorboard_log_dir)
        purge_step = self.global_step + 1 if self.global_step > 0 else None
        self.tensorboard_writer = SummaryWriter(
            log_dir=self.tensorboard_log_dir,
            purge_step=purge_step,
            flush_secs=self.tensorboard_flush_secs,
        )
        logger.info(
            "Initialized TensorBoard writer: log_dir=%s resume_step=%d purge_step=%s",
            self.tensorboard_log_dir,
            self.global_step,
            purge_step,
        )

    def _build_optimizer(self, trainable_params):
        optimizer_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": (0.9, 0.95),
        }
        valid_optimizer_types = {"auto", "torch_adamw", "fused_adam"}
        if self.optimizer_type not in valid_optimizer_types:
            raise ValueError(
                f"Unsupported optimizer_type: {self.optimizer_type}. "
                f"Expected one of: {sorted(valid_optimizer_types)}."
            )

        if self.optimizer_type == "torch_adamw":
            logger.info("Using torch.optim.AdamW optimizer (configured).")
            return torch.optim.AdamW(trainable_params, **optimizer_kwargs)

        if torch.cuda.is_available():
            try:
                from deepspeed.ops.adam import FusedAdam

                logger.info("Using DeepSpeed FusedAdam optimizer.")
                return FusedAdam(
                    trainable_params,
                    **optimizer_kwargs,
                    adam_w_mode=True,
                )
            except Exception as exc:
                if self.optimizer_type == "fused_adam":
                    raise RuntimeError("Configured DeepSpeed FusedAdam is unavailable.") from exc
                logger.warning("DeepSpeed FusedAdam unavailable; falling back to torch AdamW: %s", exc)

        elif self.optimizer_type == "fused_adam":
            raise RuntimeError("Configured DeepSpeed FusedAdam requires CUDA.")

        logger.info("Using torch.optim.AdamW optimizer.")
        return torch.optim.AdamW(trainable_params, **optimizer_kwargs)

    def _build_optimizer_param_groups(self, model):
        """Split native WING parameters from newly introduced plan modules.

        Plan adapters live inside ``model.dit`` and therefore must be removed
        from the backbone group explicitly.  Parameter-id de-duplication keeps
        shared MoT aliases from entering the optimizer twice.
        """
        all_params = self._collect_trainable_params(model)
        new_params = []
        if bool(getattr(model, "spectral_plan_enabled", False)):
            for attr in ("spectral_plan_predictor", "spectral_plan_token_projection"):
                module = getattr(model, attr, None)
                if module is not None:
                    new_params.extend(p for p in module.parameters() if p.requires_grad)
            plan_adapters = getattr(getattr(model, "mot", None), "plan_adapters", None)
            if plan_adapters is not None:
                new_params.extend(p for p in plan_adapters.parameters() if p.requires_grad)

        new_ids = {id(param) for param in new_params}
        seen = set()
        backbone_params = []
        deduped_new_params = []
        for param in all_params:
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            if param_id in new_ids:
                deduped_new_params.append(param)
            else:
                backbone_params.append(param)

        groups = []
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": self.backbone_learning_rate,
                "group_name": "backbone",
            })
        if deduped_new_params:
            groups.append({
                "params": deduped_new_params,
                "lr": self.new_module_learning_rate,
                "group_name": "new_modules",
            })
        if not groups:
            raise ValueError("No trainable parameters remain after optimizer grouping.")
        logger.info(
            "Optimizer parameter groups: backbone=%d lr=%.3g, new_modules=%d lr=%.3g",
            len(backbone_params), self.backbone_learning_rate,
            len(deduped_new_params), self.new_module_learning_rate,
        )
        return groups

    def _configure_deepspeed_gradient_clipping(self):
        """Configure clipping where DeepSpeed actually performs its optimizer step.

        With Accelerate's DeepSpeed integration, ``accelerator.backward`` delegates to
        ``DeepSpeedEngine.backward`` and ``DeepSpeedEngine.step``.  Calling
        ``accelerator.clip_grad_norm_`` afterwards only retrieves DeepSpeed's global norm, so the
        clipping threshold must be present in the DeepSpeed config before ``prepare``.
        """
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if plugin is None:
            return
        plugin.deepspeed_config["gradient_clipping"] = self.max_grad_norm

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _tensorboard_log(self, payload: dict):
        if self.tensorboard_writer is None:
            return
        for key, value in payload.items():
            self.tensorboard_writer.add_scalar(key, float(value), self.global_step)

    def _log_metrics(self, payload: dict):
        self._wandb_log(payload)
        self._tensorboard_log(payload)

    def _flush_tensorboard(self):
        writer = getattr(self, "tensorboard_writer", None)
        if writer is not None:
            writer.flush()

    def _finish_tensorboard(self):
        writer = getattr(self, "tensorboard_writer", None)
        if writer is not None:
            writer.close()
            self.tensorboard_writer = None

    def _reduce_training_loss_metrics(self, loss_dict, *, loss) -> dict[str, float]:
        """Reduce scalar training metrics across ranks."""
        global_metrics = {}
        for key, value in loss_dict.items():
            metric_tensor = torch.as_tensor(value, device=loss.device, dtype=torch.float32)
            if metric_tensor.numel() != 1:
                raise ValueError(f"Training loss metric {key!r} must be scalar.")
            global_metrics[key] = float(
                self.accelerator.gather(metric_tensor.reshape(1)).mean().item()
            )

        return global_metrics

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        # DataLoader creates an iterator base seed.  Keep that bookkeeping on an
        # independent generator so a full-state resume does not consume the model's
        # restored global Torch RNG before the next flow-matching noise sample.
        self.train_loader_generator = torch.Generator(device="cpu")
        self.train_loader_generator.manual_seed(
            self.seed + 1_000_003 * int(self.accelerator.process_index)
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            generator=self.train_loader_generator,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            # A shared multiplicative factor preserves the configured LR ratio
            # between backbone and new-module parameter groups.
            main_scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: 0.01 + 0.99 * 0.5 * (
                    1.0 + cos(pi * min(int(step), remaining_steps) / remaining_steps)
                ),
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _load_initial_weights_before_prepare(self):
        """Load model-only initialization before ZeRO creates FP32 master parameters.

        Full training-state directories are deliberately deferred to
        :meth:`_resume_training_state_after_prepare`, because ``Accelerator.load_state`` requires
        prepared model/optimizer/scheduler objects.
        """
        resume = self.resume
        partial_resume = self.partial_resume
        phase_init = getattr(self, "phase_init", None)
        has_resume = self._checkpoint_configured(resume)
        has_partial_resume = self._checkpoint_configured(partial_resume)
        has_phase_init = self._checkpoint_configured(phase_init)
        configured = sum((has_resume, has_partial_resume, has_phase_init))
        if configured > 1:
            raise ValueError("`resume`, `partial_resume`, and `phase_init` are mutually exclusive.")
        if has_phase_init:
            self._load_phase_init(phase_init)
            return
        if has_partial_resume:
            self._load_partial_resume(partial_resume)
            return
        if not has_resume:
            return
        resume_path = Path(str(resume))
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        if resume_path.is_dir():
            self._validate_training_state_dir(resume_path)
            logger.info(
                "Deferring full training-state resume until after accelerator.prepare: %s",
                resume,
            )
            return
        logger.info("Loading weight checkpoint before optimizer/DeepSpeed initialization: %s", resume)
        self.model.load_checkpoint(
            str(resume_path), optimizer=None, enforce_training_phase=True
        )
        logger.warning(
            "Loaded .pt weights before optimizer/DeepSpeed initialization; "
            "optimizer/scheduler/step/dataloader state will start fresh."
        )

    def _load_phase_init(self, phase_init):
        if getattr(self, "training_phase", None) != "policy":
            raise ValueError("`phase_init` is only valid with policy.")
        if (
            not isinstance(phase_init, Mapping)
            or "path" not in phase_init
            or not set(phase_init).issubset({"path", "random_action_expert"})
        ):
            raise ValueError(
                "`phase_init` must define `path` and may define only "
                "`random_action_expert`."
            )
        path = phase_init.get("path")
        if not self._checkpoint_configured(path):
            raise ValueError("`phase_init.path` must point to a Phase-I weights checkpoint.")
        random_action_expert = phase_init.get("random_action_expert", False)
        if not isinstance(random_action_expert, bool):
            raise ValueError("`phase_init.random_action_expert` must be a boolean.")
        checkpoint_path = Path(str(path))
        if checkpoint_path.is_dir() or checkpoint_path.suffix != ".pt":
            raise ValueError("`phase_init.path` must be a weights .pt file.")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Phase-I initialization checkpoint not found: {path}")
        summary = self.model.load_phase1_planner_checkpoint(
            str(checkpoint_path), random_action_expert=random_action_expert
        )
        logger.warning(
            "Selective Phase-I -> Phase-II initialization complete before optimizer creation: %s. "
            "Robot normalizer/coefficient head and action/proprio/projection/adapters/optimizer "
            "state remain target-initialized.",
            summary,
        )

    def _resume_training_state_after_prepare(self):
        """Restore a full prepared Accelerate/DeepSpeed state directory, if configured."""
        if (
            self._checkpoint_configured(self.partial_resume)
            or self._checkpoint_configured(getattr(self, "phase_init", None))
            or not self._checkpoint_configured(self.resume)
        ):
            return
        resume_path = Path(str(self.resume))
        if not resume_path.is_dir():
            return
        logger.info("Resuming full training state from directory: %s", self.resume)
        self.load_training_state(str(resume_path))

    def _validate_training_state_dir(self, state_dir: Path):
        required_files = (state_dir / ".complete", state_dir / "trainer_state.json")
        missing = [str(path) for path in required_files if not path.is_file()]
        if not missing:
            target_phase = getattr(self, "training_phase", None)
            if target_phase is not None:
                with open(state_dir / "trainer_state.json", "r", encoding="utf-8") as f:
                    payload = json.load(f)
                source_phase = normalize_training_phase(payload.get("training_phase"))
                if source_phase != target_phase:
                    raise ValueError(
                        "Training-state phase mismatch: "
                        f"source={source_phase!r} target={target_phase!r}."
                    )
            else:
                with open(state_dir / "trainer_state.json", "r", encoding="utf-8") as f:
                    payload = json.load(f)
                source_phase = normalize_training_phase(payload.get("training_phase"))
                if source_phase is not None:
                    raise ValueError(
                        "Training-state phase mismatch: "
                        f"source={source_phase!r} target=None."
                    )
            topology_fields = {
                "world_size": getattr(getattr(self, "accelerator", None), "num_processes", None),
                "batch_size": getattr(self, "batch_size", None),
                "gradient_accumulation_steps": getattr(
                    self, "gradient_accumulation_steps", None
                ),
                "seed": getattr(self, "seed", None),
            }
            for key, expected in topology_fields.items():
                if expected is not None and key in payload and int(payload[key]) != int(expected):
                    raise ValueError(
                        f"Training-state {key} mismatch: source={payload[key]} target={expected}."
                    )
            return
        if self.resume_allow_incomplete_state:
            logger.warning(
                "Resuming a legacy/unverified training state because "
                "resume_allow_incomplete_state=true; missing=%s",
                missing,
            )
            return
        raise ValueError(
            "Refusing to resume an incomplete or unverified training state directory. "
            f"missing={missing}. Only use resume_allow_incomplete_state=true after a manual "
            "artifact audit."
        )

    @staticmethod
    def _checkpoint_configured(value) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() not in {"", "none", "null"}
        return True

    def _load_partial_resume(self, partial_resume):
        if isinstance(partial_resume, str) or not isinstance(partial_resume, Mapping):
            raise ValueError(
                "String/non-mapping partial_resume is unsafe because endpoint compatibility is "
                "ambiguous. Configure partial_resume.path and endpoints.action/proprio explicitly."
            )

        expected_keys = {"path", "endpoints"}
        unexpected_keys = sorted(set(partial_resume) - expected_keys)
        if unexpected_keys:
            raise ValueError(
                "Unsupported partial_resume fields. Legacy tensor-filter options can create "
                f"incomplete/hybrid models; use only path and endpoints. unexpected={unexpected_keys}"
            )

        path = partial_resume.get("path")
        endpoints = partial_resume.get("endpoints")
        if not isinstance(endpoints, Mapping):
            raise ValueError(
                "`partial_resume.endpoints` is required with explicit `action` and `proprio` "
                "policies (`reset` or `load_exact`)."
            )
        expected_endpoint_keys = {"action", "proprio"}
        if set(endpoints) != expected_endpoint_keys:
            raise ValueError(
                "`partial_resume.endpoints` must define exactly `action` and `proprio`; "
                f"actual={sorted(endpoints)}."
            )
        action_endpoint_policy = str(endpoints["action"]).strip().lower()
        proprio_endpoint_policy = str(endpoints["proprio"]).strip().lower()
        valid_policies = {"reset", "load_exact"}
        if action_endpoint_policy not in valid_policies:
            raise ValueError(
                "`partial_resume.endpoints.action` must be `reset` or `load_exact`, got "
                f"{action_endpoint_policy!r}."
            )
        if proprio_endpoint_policy not in valid_policies:
            raise ValueError(
                "`partial_resume.endpoints.proprio` must be `reset` or `load_exact`, got "
                f"{proprio_endpoint_policy!r}."
            )

        if not self._checkpoint_configured(path):
            raise ValueError("`partial_resume.path` must point to a checkpoint file.")
        checkpoint_path = Path(str(path))
        if checkpoint_path.is_dir():
            raise ValueError("`partial_resume.path` must be a weights .pt file, not a training state directory.")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Partial resume checkpoint not found: {path}")

        logger.info(
            "Loading strict partial checkpoint before optimizer/DeepSpeed initialization: %s "
            "(action=%s, proprio=%s)",
            checkpoint_path,
            action_endpoint_policy,
            proprio_endpoint_policy,
        )
        summary = self.model.load_checkpoint_partial(
            str(checkpoint_path),
            action_endpoint_policy=action_endpoint_policy,
            proprio_endpoint_policy=proprio_endpoint_policy,
        )
        logger.warning(
            "Strict partial initialization complete before optimizer creation: %s. "
            "Optimizer/scheduler/step/dataloader state start fresh.",
            summary,
        )

    def _set_dit_only_train_mode(self):
        # Reapply the exact phase scope after eval/checkpoint operations.
        logger.info("Restoring configured training scope: %s", self.training_phase)
        model = self.accelerator.unwrap_model(self.model)
        self._apply_training_phase_train_mode(
            model,
            training_phase=self.training_phase,
            freeze_video_expert=self.freeze_video_expert,
        )

    @staticmethod
    def _apply_training_phase_train_mode(
        model, *, training_phase=None, freeze_video_expert: bool = False
    ):
        if training_phase is None:
            Wan22Trainer._apply_dit_only_train_mode(
                model, freeze_video_expert=freeze_video_expert
            )
            return
        if freeze_video_expert:
            raise ValueError("freeze_video_expert conflicts with explicit two-phase training.")
        model.eval()
        model.requires_grad_(False)
        if training_phase == "phase1_ego_planner":
            model.video_expert.train()
            model.video_expert.requires_grad_(True)
            predictor = getattr(model, "spectral_plan_predictor", None)
            if predictor is None:
                raise ValueError("Phase I requires spectral_plan_predictor.")
            predictor.train()
            predictor.requires_grad_(True)
            return
        if training_phase == "policy":
            Wan22Trainer._apply_dit_only_train_mode(model, freeze_video_expert=False)
            return
        raise ValueError(f"Unsupported training_phase: {training_phase!r}.")

    @staticmethod
    def _apply_dit_only_train_mode(model, *, freeze_video_expert: bool = False):
        """Set train/eval and requires_grad flags for the configured fine-tuning scope."""
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        if freeze_video_expert:
            video_expert = getattr(model, "video_expert", None)
            if video_expert is not None:
                video_expert.eval()
                video_expert.requires_grad_(False)
                logger.info("Freezing WING video expert; action/MoT/proprio train scope remains active.")
        # Unfreeze proprio projector(s): shared -> `proprio_encoder`; per_embodiment ->
        # `proprio_encoders` (native-dim ModuleList). Missing the latter left per_embodiment
        # proprio frozen (present in the forward but never trained).
        for attr in ("proprio_encoder", "proprio_encoders"):
            module = getattr(model, attr, None)
            if module is not None:
                module.train()
                module.requires_grad_(True)
        # Optional spectral predictor/projection live outside MoT, so include
        # them explicitly without changing the default WING train scope.
        if bool(getattr(model, "spectral_plan_enabled", False)):
            for attr in ("spectral_plan_predictor", "spectral_plan_token_projection"):
                module = getattr(model, attr, None)
                if module is not None:
                    module.train()
                    module.requires_grad_(True)

    @staticmethod
    def _collect_trainable_params(model):
        """The only trainable modules: DiT (MoT = video+action experts, incl. per_embodiment action
        heads) plus the proprio projector(s) -- `proprio_encoder` (shared) and/or `proprio_encoders`
        (per_embodiment ModuleList). Must mirror `_apply_dit_only_train_mode`; missing
        `proprio_encoders` left per_embodiment proprio out of the optimizer (never updated)."""
        params = [param for param in model.dit.parameters() if param.requires_grad]
        for attr in ("proprio_encoder", "proprio_encoders"):
            module = getattr(model, attr, None)
            if module is not None:
                params.extend(param for param in module.parameters() if param.requires_grad)
        if bool(getattr(model, "spectral_plan_enabled", False)):
            for attr in ("spectral_plan_predictor", "spectral_plan_token_projection"):
                module = getattr(model, attr, None)
                if module is not None:
                    params.extend(param for param in module.parameters() if param.requires_grad)
        return params

    @staticmethod
    def _validate_spectral_training_scope(model, *, freeze_video_expert: bool) -> None:
        training_phase = getattr(model, "training_phase", None)
        if training_phase is not None:
            if freeze_video_expert:
                raise ValueError("freeze_video_expert conflicts with explicit two-phase training.")
            if not bool(getattr(model, "spectral_plan_enabled", False)):
                raise ValueError("Two-phase training requires spectral_plan.enabled=true.")
            if getattr(model, "spectral_plan_training_mode", None) != "joint":
                raise ValueError("Two-phase training requires spectral_plan.training_mode=joint.")
            if getattr(model, "spectral_plan_routing_mode", None) != "fixed":
                raise ValueError("Two-phase training requires fixed Full-K4 routing.")
            if not bool(getattr(model, "spectral_plan_detach_backbone", False)):
                raise ValueError("Two-phase training requires spectral_plan.detach_backbone=true.")
            normalizer = getattr(model, "spectral_plan_normalizer", None)
            if normalizer is None or int(normalizer.low_frequencies) != 4:
                raise ValueError("Two-phase training requires all four K0-K3 coefficients.")
            if training_phase == "phase1_ego_planner":
                if float(getattr(model, "spectral_plan_lambda", 0.0)) <= 0.0:
                    raise ValueError("Phase I requires lambda_plan > 0.")
                if getattr(model, "proprio_dim", None) is not None:
                    raise ValueError("Phase-I ego planner requires proprio_dim=null.")
            elif training_phase == "policy":
                if float(getattr(model, "spectral_plan_lambda", 0.0)) <= 0.0:
                    raise ValueError("Phase II requires lambda_plan > 0.")
            else:
                raise ValueError(f"Unsupported training_phase: {training_phase!r}.")
            return
        if (
            bool(getattr(model, "spectral_plan_enabled", False))
            and getattr(model, "spectral_plan_training_mode", "joint") == "current_only"
            and not freeze_video_expert
        ):
            raise ValueError(
                "spectral_plan.training_mode=current_only has no video loss and therefore requires "
                "freeze_video_expert=true. Use training_mode=joint for Experiment B."
            )

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        # Preserve the per_embodiment routing label so val-loss `training_loss` can pick the
        # right head (MultiEmbodimentDataset tags each sample with a scalar group_id).
        group_id = sample.get("group_id", None)
        if group_id is not None and group_id.ndim == 0:
            group_id = group_id.unsqueeze(0)  # [1] to match the B=1 batched sample

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
            "group_id": group_id,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        # Report val loss only (skip rollout/video/action metrics) for multi-source val (per-group
        # processors break the single `lerobot_dataset.processor` metric path) and for per_embodiment
        # models (infer doesn't pass embodiment_id -> rollout would crash, incl. single-source).
        # Multi-source/per-embodiment validation has no single rollout schema.
        if _eval_skips_rollout(self.val_dataset, getattr(model, "action_head_mode", "shared")):
            vl = torch.tensor([val_loss], device=self.accelerator.device, dtype=torch.float32)
            val_loss_mean = self.accelerator.gather_for_metrics(vl).mean().item()
            if was_dit_training:
                self._set_dit_only_train_mode()
            nan = float("nan")
            return {
                "val_loss": float(val_loss_mean),
                "psnr_rg": nan, "ssim_rg": nan, "psnr_rd": nan,
                "ssim_rd": nan, "psnr_dg": nan, "ssim_dg": nan,
            }

        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        infer_proprio = _eval_inference_proprio(model, proprio)
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": infer_proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training or getattr(self, "training_phase", None) is not None:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        tmp_ckpt_path = f"{ckpt_path}.tmp"
        model.save_checkpoint(tmp_ckpt_path, optimizer=None, step=self.global_step)
        os.replace(tmp_ckpt_path, ckpt_path)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        tmp_state_file = f"{state_file}.tmp"
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "training_phase": getattr(self, "training_phase", None),
            "world_size": int(self.accelerator.num_processes),
            "batch_size": int(self.batch_size),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "seed": int(self.seed),
        }
        with open(tmp_state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        os.replace(tmp_state_file, state_file)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        complete_file = os.path.join(state_path, ".complete")
        if self.accelerator.is_main_process and os.path.exists(complete_file):
            os.remove(complete_file)
        self.accelerator.wait_for_everyone()

        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            tmp_complete_file = f"{complete_file}.tmp"
            with open(tmp_complete_file, "w", encoding="utf-8") as f:
                json.dump({"global_step": int(self.global_step)}, f, ensure_ascii=True)
            os.replace(tmp_complete_file, complete_file)
        self.accelerator.wait_for_everyone()

        pruned_steps = ()
        pruned_state_steps = ()
        if self.accelerator.is_main_process:
            self._flush_tensorboard()
            if getattr(self, "checkpoint_keep_last", 0) > 0:
                if self.checkpoint_state_keep_last is None:
                    prune_plan = plan_checkpoint_prune(
                        self.checkpoint_root,
                        keep_last=self.checkpoint_keep_last,
                    )
                    for group in prune_plan:
                        logger.info(
                            "[ckpt-prune][dry-run] kind=pair step=%d state=%s weights=%s",
                            group.step,
                            group.state_path,
                            group.weights_path,
                        )
                    pruned_steps = apply_checkpoint_prune(self.checkpoint_root, prune_plan)
                    if pruned_steps:
                        logger.info(
                            "[ckpt-prune] kind=pair removed_steps=%s keep_last=%d",
                            list(pruned_steps),
                            self.checkpoint_keep_last,
                        )
                else:
                    state_prune_plan = plan_checkpoint_state_prune(
                        self.checkpoint_root,
                        keep_last=self.checkpoint_state_keep_last,
                    )
                    weights_prune_plan = plan_checkpoint_weights_prune(
                        self.checkpoint_root,
                        keep_last=self.checkpoint_keep_last,
                    )
                    # Log both dry-run plans before mutating either retention tier.
                    for group in state_prune_plan:
                        logger.info(
                            "[ckpt-prune][dry-run] kind=full_state step=%d state=%s",
                            group.step,
                            group.state_path,
                        )
                    for checkpoint in weights_prune_plan:
                        logger.info(
                            "[ckpt-prune][dry-run] kind=weights step=%d weights=%s",
                            checkpoint.step,
                            checkpoint.weights_path,
                        )
                    pruned_state_steps = apply_checkpoint_state_prune(
                        self.checkpoint_root, state_prune_plan
                    )
                    pruned_steps = apply_checkpoint_weights_prune(
                        self.checkpoint_root, weights_prune_plan
                    )
                    if pruned_state_steps:
                        logger.info(
                            "[ckpt-prune] kind=full_state removed_steps=%s keep_last=%d",
                            list(pruned_state_steps),
                            self.checkpoint_state_keep_last,
                        )
                    if pruned_steps:
                        logger.info(
                            "[ckpt-prune] kind=weights removed_steps=%s keep_last=%d",
                            list(pruned_steps),
                            self.checkpoint_keep_last,
                        )
        self.accelerator.wait_for_everyone()

        return {
            "weights_path": ckpt_path,
            "state_path": state_path,
            "pruned_steps": list(pruned_steps),
            "pruned_state_steps": list(pruned_state_steps),
        }

    def load_training_state(self, state_dir: str):
        self._validate_training_state_dir(Path(state_dir))
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            topology_fields = {
                "world_size": int(self.accelerator.num_processes),
                "batch_size": int(self.batch_size),
                "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
                "seed": int(self.seed),
            }
            for key, expected in topology_fields.items():
                if key in payload and int(payload[key]) != expected:
                    raise ValueError(
                        f"Training-state {key} mismatch: source={payload[key]} target={expected}."
                    )

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch(self.epoch)
                self.train_sampler.set_epoch_offset(0)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        if self.global_step >= self.max_steps:
            logger.info(
                "Training state is already complete: global_step=%d max_steps=%d; exiting without rewriting checkpoints.",
                self.global_step,
                self.max_steps,
            )
            self._finish_tensorboard()
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                self.train_sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                anomaly_context = (
                    torch.autograd.detect_anomaly(check_nan=True)
                    if self.debug_nonfinite
                    else nullcontext()
                )
                with anomaly_context:
                    self._raise_if_nonfinite_structure(sample, stage="train_sample_before_forward")
                    self._raise_if_nonfinite_named_parameters(stage="before_forward")
                    with self.accelerator.autocast():
                        loss, loss_dict = train_model.training_loss(sample)
                    self._raise_if_nonfinite_structure(
                        {"loss": loss, **loss_dict},
                        stage="training_loss_output",
                    )
                    self._raise_if_nonfinite_named_parameters(stage="after_forward_before_backward")
                    self.accelerator.backward(loss)
                    self._raise_if_nonfinite_gradients(stage="after_backward_before_clip")
                    self._raise_if_nonfinite_named_parameters(stage="after_backward_before_clip")

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self._raise_if_nonfinite_gradients(stage="after_backward_and_clip")
                    self._raise_if_nonfinite_named_parameters(stage="after_backward_and_clip")
                    self._raise_if_nonfinite_structure({"grad_norm": torch.as_tensor(grad_norm)}, stage="grad_norm")
                    self._raise_if_nonfinite_named_parameters(stage="before_optimizer_step")
                    self.optimizer.step()
                    self._raise_if_nonfinite_named_parameters(stage="after_optimizer_step")
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = self._reduce_training_loss_metrics(
                        loss_dict, loss=loss
                    )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    lr_by_group = {
                        str(group.get("group_name", f"group_{index}")): float(group["lr"])
                        for index, group in enumerate(self.optimizer.param_groups)
                    }
                    current_lr = lr_by_group.get("backbone", next(iter(lr_by_group.values())))
                    current_new_lr = lr_by_group.get("new_modules")

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        lr_description = f"lr_backbone={current_lr:.2e}"
                        if current_new_lr is not None:
                            lr_description += f" lr_new={current_new_lr:.2e}"
                        description += "%s speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            lr_description,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "train/lr_backbone": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        if current_new_lr is not None:
                            wandb_payload["train/lr_new_modules"] = current_new_lr
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._log_metrics(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._log_metrics(eval_payload)


                    ckpt_info = None
                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        if ckpt_info is None:
                            ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        self._finish_tensorboard()
                        return

        ckpt_info = self.save_checkpoint()
        self._finish_tensorboard()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )

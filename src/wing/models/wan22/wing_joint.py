from typing import Any, Optional, Sequence, Union

import torch

from wing.utils.logging_config import get_logger

from .wing import WING

logger = get_logger(__name__)


class WINGJoint(WING):
    """WING variant where action attends to all video latent tokens."""

    supports_batch_inference = True

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for WINGJoint."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "WINGJoint requires `video_dit_config['action_conditioned']=false`."
            )
        # per_embodiment heads are NOT wired through the joint/idm custom infer + (idm) training_loss
        # paths -- those never pass `embodiment_id`, so a per_embodiment build would crash deep in
        # ActionDiT with a misleading error. Fail loudly here. (Inherited by WINGIDM.)
        action_dit_config = kwargs.get("action_dit_config", None) or {}
        head_mode = str(action_dit_config.get("action_head_mode", "shared"))
        if head_mode != "shared":
            raise ValueError(
                f"{cls.__name__} only supports action_head_mode='shared' (got '{head_mode}'). "
                "per_embodiment heads are implemented only for the uncond WING model; the "
                "joint/idm variants do not route embodiment_id through their infer/training_loss."
            )
        return super().from_wan22_pretrained(**kwargs)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> full video
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        if test_action_with_infer_action:
            logger.warning(
                "`WINGJoint.infer_joint` ignores `test_action_with_infer_action=True` "
                "and always runs with `test_action_with_infer_action=False`."
            )
        return super().infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            test_action_with_infer_action=False,
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[Union[str, Sequence[str]]],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        plan_intervention: str = "predicted",
        shuffled_plan_tokens: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        self.eval()

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] < 1 or input_image.shape[1] != 3:
            raise ValueError(
                "`input_image` must have shape [B,3,H,W] or [3,H,W] with "
                f"B >= 1, got {tuple(input_image.shape)}"
            )
        batch_size, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                if batch_size != 1:
                    raise ValueError(
                        f"1D `proprio` is only valid for B=1, got input_image B={batch_size}"
                    )
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == batch_size:
                pass
            else:
                raise ValueError(
                    "`proprio` must be [D] for B=1 or [B,D] matching input_image, "
                    f"got shape {tuple(proprio.shape)} for B={batch_size}"
                )
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        # Draw one video/action pair per request. With a fixed seed, every item
        # reproduces the historical B=1 random streams; without a seed, this
        # preserves the video-then-action draw order of sequential B=1 calls.
        video_samples = []
        action_samples = []
        for _ in range(batch_size):
            video_generator = (
                None
                if seed is None
                else torch.Generator(device=rand_device).manual_seed(seed)
            )
            action_generator = (
                None
                if seed is None
                else torch.Generator(device=rand_device).manual_seed(seed)
            )
            video_samples.append(
                torch.randn(
                    (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
                    generator=video_generator,
                    device=rand_device,
                    dtype=torch.float32,
                )
            )
            action_samples.append(
                torch.randn(
                    (1, action_horizon, self.action_expert.action_dim),
                    generator=action_generator,
                    device=rand_device,
                    dtype=torch.float32,
                )
            )
        latents_video = torch.cat(video_samples, dim=0).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        latents_action = torch.cat(action_samples, dim=0).to(
            device=self.device,
            dtype=self.torch_dtype,
        )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            if isinstance(prompt, Sequence) and not isinstance(prompt, str):
                if len(prompt) != batch_size:
                    raise ValueError(
                        "`prompt` batch length must match input_image "
                        f"B={batch_size}, got {len(prompt)}"
                    )
            elif batch_size != 1:
                raise ValueError(
                    "Batched `input_image` requires a sequence of one prompt per sample."
                )
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            if context.shape[0] != batch_size or context_mask.shape[0] != batch_size:
                raise ValueError(
                    "`context/context_mask` batch dimension must match input_image: "
                    f"got {context.shape[0]}/{context_mask.shape[0]} vs B={batch_size}"
                )
        context, context_mask, _ = (
            self._append_inference_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        )

        # The plan is always predicted once from the clean current observation.
        # It remains an ActionDiT-only adapter condition while the native Joint
        # path continues to denoise future video/action latents together.
        plan_condition = None
        plan_tokens = None
        if self.spectral_plan_enabled:
            timestep_current = torch.zeros(
                (first_frame_latents.shape[0],),
                dtype=first_frame_latents.dtype,
                device=self.device,
            )
            current_video_pre = self.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=timestep_current,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            predictor_visual, predictor_context = self._spectral_predictor_inputs(
                current_video_pre["tokens"], context
            )
            plan_condition = self._build_spectral_plan_condition(
                current_visual_tokens=predictor_visual,
                instruction_tokens=predictor_context,
                instruction_mask=context_mask,
                plan_intervention=plan_intervention,
                donor_plan=shuffled_plan_tokens,
            )
            plan_tokens = plan_condition["tokens"]
        elif str(plan_intervention).strip().lower() != "predicted":
            raise ValueError("Plan intervention requires spectral_plan.enabled=true.")

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.expand(batch_size).to(
                dtype=latents_video.dtype,
                device=self.device,
            )
            timestep_action = step_t_action.expand(batch_size).to(
                dtype=latents_action.dtype,
                device=self.device,
            )

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=None,
                plan_tokens=plan_tokens,
            )

            latents_video = self.infer_video_scheduler.step(pred_video_posi, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action_posi, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action.detach().to(device="cpu", dtype=torch.float32)
        result = {"action": action_out[0] if batch_size == 1 else action_out}
        if plan_condition is not None:
            coefficients = plan_condition["coefficients"].detach().to(
                device="cpu", dtype=torch.float32
            )
            result["coeff_pred"] = coefficients[0] if batch_size == 1 else coefficients
            result["plan_intervention"] = str(plan_intervention).strip().lower()
        return result

    @torch.no_grad()
    def infer_action_batch(
        self,
        prompt: Optional[Sequence[str]],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        plan_intervention: str = "predicted",
        shuffled_plan_tokens: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Joint video/action denoising with a stable ``[B,T,D]`` action ABI."""
        out = self.infer_action(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            num_video_frames=num_video_frames,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            plan_intervention=plan_intervention,
            shuffled_plan_tokens=shuffled_plan_tokens,
        )
        action = out["action"]
        if action.ndim == 2:
            action = action.unsqueeze(0)
        batch_size = 1 if input_image.ndim == 3 else int(input_image.shape[0])
        if action.ndim != 3 or action.shape[0] != batch_size:
            raise RuntimeError(
                "WINGJoint batched action output must be [B,T,D], got "
                f"{tuple(action.shape)} for B={batch_size}"
            )
        out["action"] = action
        return out

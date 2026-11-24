# Real-Robot Deployment

WING uses a persistent policy server and a robot-side RTC client. The maintained
runtime supports dual-arm 14D checkpoints, native single-arm 7D checkpoints,
and 7D checkpoints adapted to the dual-arm interface with `keep_dims`.

## External requirements

The policy server uses the normal WING installation (`pip install -e .`) and
requires a CUDA-capable PyTorch environment for the released model. The robot
computer additionally requires:

- ROS Noetic with `rospy`, `sensor_msgs`, and `cv_bridge`.
- A Piper ROS workspace providing `piper/start_ms_piper.launch`.
- The robot vendor's `pyAgxArm` Python package and native libraries.
- Linux SocketCAN, `iproute2`, `can-utils`, and `ethtool`.
- Three ROS RGB streams: front, left wrist, and right wrist.

These hardware runtimes are not vendored. ROS and the vendor SDK require their
platform-specific installation procedures.

WING requires Python 3.10 or newer. The robot interpreter must also have ROS
`rospy`/`cv_bridge` and vendor bindings built for that same Python ABI; a stock
ROS Noetic binding is not guaranteed to import from an arbitrary virtualenv.

## Configure

Copy `configs/real_deploy_wing.yaml` to an operator-owned local file. Set the
two CAN interface names, three ROS image topics, and both calibrated 7D initial
poses in that local file. Never reuse example poses from another robot.

Set model and server values in the shell used by both processes:

```bash
export WING_CHECKPOINT="$PWD/checkpoints/step_xxxxxx.pt"
export WING_DATASET_STATS="$PWD/checkpoints/dataset_stats.json"
export WING_TRAIN_CONFIG="$PWD/checkpoints/resolved_training_config.yaml"
export WING_TEXT_CACHE="$PWD/checkpoints/text_embedding_cache"
export WAN_MODEL_PATH="$PWD/checkpoints/Wan2.2-TI2V-5B"
export SPECTRAL_STATS_PATH="$PWD/checkpoints/latent_low_k4_stats.pt"
export WING_TASK_PROMPT='operator-approved task instruction'
export WING_POLICY_HOST='<policy-server-host>'
export WING_POLICY_PORT='<policy-server-port>'
export WING_POLICY_AUTHKEY='<long-random-shared-secret>'
```

The server transport uses authenticated Python object serialization. Run it
only on a trusted, firewalled robot network; do not expose it to the internet.

For a native 7D checkpoint, leave `wing.keep_dims` null and run with
`runtime.arm_mode: left` or `right`. For a subset checkpoint presented through
the 14D robot interface, set `wing.keep_dims` (for example, `7:14`) and set
`wing.full_action_pad` to the exact concatenation of `left_init_position` and
`right_init_position`. Startup rejects inconsistent values.

## Start the server

From the repository root on the policy-server machine:

```bash
python scripts/deploy_real_bot/serve_wing_policy.py \
  --config configs/real_deploy_local.yaml
```

The server loads the model once. Checkpoint classes referenced by the saved
training config must exist in the installed WING core, including `WING`,
`WINGJoint`, and the corresponding IDM classes used by older checkpoints.

### Relocate inference artifacts

The WING checkpoint is not the only file read during policy construction.
Before starting the server, relocate or override these training-time paths:

- `WING_CHECKPOINT`: the final WING policy checkpoint.
- `WING_DATASET_STATS`: action/proprio normalization JSON consumed by the
  deployment processor.
- `WAN_MODEL_PATH`: the local WAN model directory, unless the configured public
  model identifier can be downloaded. The VAE is still loaded from this source.
- `SPECTRAL_STATS_PATH`: spectral-plan normalization statistics loaded by the
  retained policy model factory.
- `WING_TEXT_CACHE`: cached context and mask for the exact formatted prompt when
  `use_text_encoder` is false. If it is true, relocate the configured tokenizer
  and text-encoder model instead.

`phase_init`, `resume`, and `partial_resume` belong to training and are not
loaded by `WINGPolicyClient`; a Phase-I checkpoint is therefore not required
for deployment. Dataset directories and VAE latent caches are also not opened
for online inference. The first dataset directory is consulted only for the
optional `keep_dims` joint-name cross-check; missing metadata emits a warning.
Prefer a scrubbed, resolved copy of the original training config and update the
paths above rather than shipping machine-specific paths.

When `WING_TRAIN_CONFIG` points to a resolved Hydra training config, edit its
`model.model_id` and `model.spectral_plan.normalization_stats_path` values
directly; environment overrides only affect values that remain interpolated.
Leaving `WING_TRAIN_CONFIG` empty uses the retained
`libero` fallback and the `WAN_MODEL_PATH` and
`SPECTRAL_STATS_PATH` environment values shown above.

Deployment forces `skip_dit_load_from_pretrain=true` and clears both
`action_dit_pretrained_path` and `dit_checkpoint_path` before model
instantiation. The final WING checkpoint supplies those full expert weights, so
training-only DiT and ActionDiT initializer files do not need to be relocated.
The VAE, prompt context (or text model), dataset normalization JSON, and
spectral normalization statistics remain required.

## Verify without motion

Start the robot-side process with action execution disabled. `--no-robot`
also skips the vendor SDK and CAN connection, but ROS and all three cameras are
still required.

The checked-in config defaults to `execute_actions: false` and
`move_to_initial: false`; therefore its default behavior sends no position
commands. Keep the explicit no-motion flags during initial verification.

```bash
python scripts/deploy_real_bot/deploy_real_wing.py \
  --config configs/real_deploy_local.yaml \
  --no-execute-actions --no-move-to-initial --no-robot --max-steps 100
```

Without `--no-robot`, the runtime connects to and enables the configured arms
but does not send position commands while `execute_actions` is false.

## Enable motion

Before motion, verify emergency stops, joint limits, camera orientation,
gripper units, checkpoint dimensions, calibrated initial poses, and a clear
workspace. Begin with conservative speed and safety thresholds. Keep an
operator at the emergency stop.

```bash
python scripts/deploy_real_bot/deploy_real_wing.py \
  --config configs/real_deploy_local.yaml \
  --execute-actions --move-to-initial --max-steps 500
```

The runtime does not move until the operator types `MOVE`. It stops when the
predicted action jump or measured state-tracking error exceeds the configured
threshold. A failed or non-finite state read aborts instead of substituting a
zero state.

## CAN setup

The optional setup script resets only interfaces explicitly supplied by the
operator and queries the arms without enabling motion:

```bash
export WING_PY="$(command -v python)"
export ROS_SETUP=./local/ros/setup.bash
export PIPER_WS_SETUP=./local/piper_workspace/setup.bash
export LEFT_CAN_INTERFACE='<left-can-interface>'
export RIGHT_CAN_INTERFACE='<right-can-interface>'
bash scripts/deploy_real_bot/repair_piper_can.sh
```

Set `EXTRA_CAN_INTERFACES` to a space-separated list only when the installation
has additional operator-approved Piper interfaces. Set `PYAGX_LD_PRELOAD` only
when the vendor SDK requires an explicit compatibility library.

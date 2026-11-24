"""RTC-style Piper robot runtime for WING policy deployment.

Supports three arm modes, selected by ``runtime.arm_mode`` in the config
(overridable from the CLI in deploy_real_wing.py):

  - ``"dual"``  : 14-dim policy driving both arms.
                  state/action layout = [L j0..j5, L grip, R j0..j5, R grip].
  - ``"right"`` : 7-dim policy driving the right arm only ([R j0..j5, R grip]).
                  The left arm is not connected by this process, while the
                  left-wrist camera stream remains required by the policy.
  - ``"left"``  : 7-dim policy driving the left arm only ([L j0..j5, L grip]).
                  The right arm is not connected by this process, while its
                  camera stream remains required.

Dual mode connects both Piper arms. Single mode connects only the selected arm.
All three RGB camera streams are required in every mode.

Gripper unit: all modes read the gripper as ``.value`` (meters, 0-0.1), which
matches the training-data gripper unit and the reference openpi right-only
deploy. The pyAgxArm parser already scales the raw CAN value (µm) to meters by
1e-6 in width mode, so ``.value`` is directly in meters. (Earlier pyAgxArm
exposed a ``.width`` alias; current versions only provide ``.value``.)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Mapping

import numpy as np

from wing.deployment.wing_remote_policy_client import WINGRemotePolicyClient
from wing.deployment.rtc_buffer import RealTimeChunkingBuffer


ARM_MODES = ("dual", "left", "right")
ACTION_DIM_DUAL = 14
ACTION_DIM_SINGLE = 7


def action_dim_for(arm_mode: str) -> int:
    """14-dim for dual arms, 7-dim for a single arm."""
    return ACTION_DIM_DUAL if arm_mode == "dual" else ACTION_DIM_SINGLE


def normalize_arm_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in ARM_MODES:
        raise ValueError(f"runtime.arm_mode must be one of {ARM_MODES}, got {value!r}")
    return mode


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section `{name}` must be a mapping.")
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _required_text(config: Mapping[str, Any], name: str) -> str:
    value = config.get(name)
    if value is None or not str(value).strip():
        raise ValueError(f"Config value `{name}` must be supplied by the operator.")
    return str(value)


def _required_pose(config: Mapping[str, Any], name: str) -> np.ndarray:
    value = config.get(name)
    if value is None:
        raise ValueError(f"Config value `robot.{name}` must be a calibrated 7D pose.")
    pose = np.asarray(value, dtype=np.float32)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"Config value `robot.{name}` must contain seven finite numbers.")
    return pose


class DualArmRosOperator:
    """Synchronize the three RGB ROS image topics used by WING.

    Identical for every arm mode: the model always consumes front + both wrist
    cameras, even when only one arm is commanded.
    """

    def __init__(self, config: Mapping[str, Any]):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.rospy = rospy
        self.Image = Image
        self.bridge = CvBridge()
        self.cameras = _section(config, "cameras")
        self.topics = _section(config, "ros_topics_dual")
        self.deques = {
            "front": deque(maxlen=2000),
            "left_wrist": deque(maxlen=2000),
            "right_wrist": deque(maxlen=2000),
        }
        self.camera_names = {
            "front": self.cameras.get("front_camera_name", "cam_high"),
            "left_wrist": self.cameras.get("left_wrist_camera_name", "cam_left_wrist"),
            "right_wrist": self.cameras.get("right_wrist_camera_name", "cam_right_wrist"),
        }

        if not all(
            _bool(self.cameras.get(flag, True))
            for flag in ["use_front", "use_wrist", "use_left_wrist", "use_right_wrist"]
        ):
            raise ValueError("WING deployment currently requires front, left wrist, and right wrist RGB cameras.")

        self.init_ros()

    def init_ros(self) -> None:
        front_topic = _required_text(self.topics, "img_front_topic")
        left_topic = _required_text(self.topics, "img_left_topic")
        right_topic = _required_text(self.topics, "img_right_topic")
        self.rospy.init_node("wing_rtc_robot_runtime", anonymous=True)
        self.rospy.Subscriber(
            front_topic,
            self.Image,
            lambda msg: self.deques["front"].append(msg),
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.rospy.Subscriber(
            left_topic,
            self.Image,
            lambda msg: self.deques["left_wrist"].append(msg),
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.rospy.Subscriber(
            right_topic,
            self.Image,
            lambda msg: self.deques["right_wrist"].append(msg),
            queue_size=1000,
            tcp_nodelay=True,
        )

    def get_frame(self) -> dict[str, np.ndarray] | None:
        if any(len(image_deque) == 0 for image_deque in self.deques.values()):
            return None

        frame_time = min(image_deque[-1].header.stamp.to_sec() for image_deque in self.deques.values())
        if any(image_deque[-1].header.stamp.to_sec() < frame_time for image_deque in self.deques.values()):
            return None

        images = {}
        for camera_key, image_deque in self.deques.items():
            while image_deque and image_deque[0].header.stamp.to_sec() < frame_time:
                image_deque.popleft()
            if not image_deque:
                return None
            images[self.camera_names[camera_key]] = self.bridge.imgmsg_to_cv2(
                image_deque.popleft(), desired_encoding="rgb8"
            )
        return images


class DualPiperController:
    """Read state and optionally execute Piper actions through pyAgxArm.

    Drives both arms in ``arm_mode="dual"`` (14-dim) or a single arm in
    ``arm_mode="left"`` / ``"right"`` (7-dim). Single-arm mode connects and
    enables only the selected arm.
    """

    def __init__(self, config: Mapping[str, Any], arm_mode: str = "dual"):
        from pyAgxArm import AgxArmFactory, create_agx_arm_config

        self.AgxArmFactory = AgxArmFactory
        self.create_agx_arm_config = create_agx_arm_config
        self.robot = _section(config, "robot")
        self.arm_mode = normalize_arm_mode(arm_mode)
        self.action_dim = action_dim_for(self.arm_mode)
        self.left_arm = None
        self.right_arm = None
        self.left_gripper = None
        self.right_gripper = None
        self.left_init_position = _required_pose(self.robot, "left_init_position")
        self.right_init_position = _required_pose(self.robot, "right_init_position")
        # Policy state/action layout depends on the mode. In single-arm mode the
        # other arm is not connected and is not part of state/action.
        if self.arm_mode == "dual":
            self.initial_position = np.concatenate(
                [self.left_init_position, self.right_init_position]
            ).astype(np.float32)
        elif self.arm_mode == "left":
            self.initial_position = self.left_init_position.astype(np.float32).copy()
        else:  # right
            self.initial_position = self.right_init_position.astype(np.float32).copy()
        self.action_safety_threshold = float(self.robot.get("action_safety_threshold", 1.5))
        self.state_safety_threshold = float(self.robot.get("state_safety_threshold", 0.3))

    def connect(self) -> bool:
        try:
            bitrate = int(self.robot.get("bitrate", 1000000))
            required: list[tuple[str, str]] = []
            if self.arm_mode in {"dual", "left"}:
                required.append(("left", _required_text(self.robot, "left_channel")))
            if self.arm_mode in {"dual", "right"}:
                required.append(("right", _required_text(self.robot, "right_channel")))

            for side, channel in required:
                print(f"[robot] connecting {side} arm on {channel} @ {bitrate}")
                cfg = self.create_agx_arm_config(
                    robot="piper",
                    comm="can",
                    channel=channel,
                    bitrate=bitrate,
                )
                arm = self.AgxArmFactory.create_arm(cfg)
                gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
                arm.connect()
                if side == "left":
                    self.left_arm = arm
                    self.left_gripper = gripper
                else:
                    self.right_arm = arm
                    self.right_gripper = gripper
            time.sleep(0.5)

            connected = []
            for side, _ in required:
                arm = self.left_arm if side == "left" else self.right_arm
                if arm is None or not arm.is_ok():
                    raise RuntimeError(f"{side} arm connection status check failed.")
                connected.append(arm)

            for arm in connected:
                arm.set_flange_vel_acc_limits(
                    max_linear_vel=float(self.robot.get("max_linear_vel", 0.5)),
                    max_angular_vel=float(self.robot.get("max_angular_vel", 0.1)),
                    max_linear_acc=float(self.robot.get("max_linear_acc", 0.1)),
                    max_angular_acc=float(self.robot.get("max_angular_acc", 0.05)),
                    timeout=1.0,
                )
                arm.set_speed_percent(int(self.robot.get("speed_pct", 15)))
                if not self._enable_arm(arm):
                    raise RuntimeError("Robot arm enable timeout.")
            return True
        except Exception as exc:
            print(f"[robot] connect failed: {exc}")
            return False

    def _enable_arm(self, arm) -> bool:
        for _ in range(5):
            if arm.enable():
                return True
            time.sleep(0.5)
        return False

    def _read_arm_into(self, arm, gripper, state: np.ndarray, base: int) -> None:
        """Fill state[base:base+6] with joint angles and state[base+6] with gripper.

        Gripper is read as ``.value`` (meters, 0-0.1; the parser already scales
        raw µm by 1e-6 in width mode) to match the training-data unit and the
        validated right-only deploy.
        """
        if arm is None or gripper is None:
            raise RuntimeError("Robot arm or gripper is not connected.")
        joint_angles = arm.get_joint_angles()
        if joint_angles is None:
            raise RuntimeError("Robot joint-state read returned no data.")
        state[base:base + 6] = joint_angles.msg
        gripper_status = gripper.get_gripper_status()
        if gripper_status is None:
            raise RuntimeError("Robot gripper-state read returned no data.")
        state[base + 6] = gripper_status.msg.value

    def get_status_and_state(self) -> np.ndarray:
        state = np.zeros(self.action_dim, dtype=np.float32)
        if self.arm_mode == "dual":
            self._read_arm_into(self.left_arm, self.left_gripper, state, 0)
            self._read_arm_into(self.right_arm, self.right_gripper, state, 7)
        elif self.arm_mode == "left":
            self._read_arm_into(self.left_arm, self.left_gripper, state, 0)
        else:
            self._read_arm_into(self.right_arm, self.right_gripper, state, 0)
        if not np.isfinite(state).all():
            raise RuntimeError("Robot state contains non-finite values.")
        return state

    def move_initial(self) -> None:
        """Move the connected arm(s) to their init poses."""
        n = 50

        def _cur(arm, fallback):
            ja = arm.get_joint_angles()
            return np.asarray(ja.msg, dtype=np.float32) if ja is not None else np.asarray(fallback, dtype=np.float32)

        left_traj = None
        right_traj = None
        if self.arm_mode in {"dual", "left"}:
            left_traj = np.linspace(_cur(self.left_arm, self.left_init_position[0:6]), self.left_init_position[0:6], n)
            left_gripper = float(np.clip(self.left_init_position[6], 0.0, 0.1))
        if self.arm_mode in {"dual", "right"}:
            right_traj = np.linspace(_cur(self.right_arm, self.right_init_position[0:6]), self.right_init_position[0:6], n)
            right_gripper = float(np.clip(self.right_init_position[6], 0.0, 0.1))
        for i in range(n):
            if left_traj is not None:
                self.left_arm.move_js(left_traj[i].tolist())
                self.left_gripper.move_gripper_m(value=left_gripper, force=1.0)
            if right_traj is not None:
                self.right_arm.move_js(right_traj[i].tolist())
                self.right_gripper.move_gripper_m(value=right_gripper, force=1.0)
            time.sleep(0.02)

    def _move_arm(self, arm, gripper, joints: np.ndarray, gripper_cmd: float) -> None:
        arm.move_js(joints.tolist())
        gripper.move_gripper_m(value=float(np.clip(gripper_cmd, 0.0, 0.1)), force=1.0)

    def move(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"Expected a finite action with shape ({self.action_dim},), got {action}.")
        if self.arm_mode == "dual":
            # 14-dim layout [L j0..j5, L grip, R j0..j5, R grip]:
            # left = joints action[0:6] + grip action[6]; right = joints action[7:13] + grip action[13].
            self._move_arm(self.left_arm, self.left_gripper, action[0:6], action[6])
            self._move_arm(self.right_arm, self.right_gripper, action[7:13], action[13])
        elif self.arm_mode == "left":
            # 7-dim left action; this process does not connect the right arm.
            self._move_arm(self.left_arm, self.left_gripper, action[0:6], action[6])
        else:  # right
            # 7-dim right action; this process does not connect the left arm.
            self._move_arm(self.right_arm, self.right_gripper, action[0:6], action[6])
        time.sleep(0.02)


class NoRobotController:
    """State-only controller for camera + policy inference without CAN access."""

    def __init__(self, config: Mapping[str, Any], arm_mode: str = "dual"):
        self.robot = _section(config, "robot")
        self.arm_mode = normalize_arm_mode(arm_mode)
        self.action_dim = action_dim_for(self.arm_mode)
        left_init = _required_pose(self.robot, "left_init_position")
        right_init = _required_pose(self.robot, "right_init_position")
        if self.arm_mode == "dual":
            self.state = np.concatenate([left_init, right_init]).astype(np.float32)
        elif self.arm_mode == "left":
            self.state = left_init.astype(np.float32).copy()
        else:
            self.state = right_init.astype(np.float32).copy()
        self.action_safety_threshold = float(self.robot.get("action_safety_threshold", 1.5))
        self.state_safety_threshold = float(self.robot.get("state_safety_threshold", 0.3))

    def connect(self) -> bool:
        print("[runtime] no_robot=True: skip pyAgxArm/CAN connection; using init/action state for inference.")
        return True

    def get_status_and_state(self) -> np.ndarray:
        return self.state.copy()

    def observe_action(self, action: np.ndarray) -> None:
        self.state = np.asarray(action, dtype=np.float32).copy()

    def move_initial(self) -> None:
        print("[runtime] no_robot=True: skip move_initial.")

    def move(self, action: np.ndarray) -> None:
        self.observe_action(action)


class RTCWINGRobotRuntime:
    """Robot-side runtime that talks to a persistent WING policy server."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.wing = _section(config, "wing")
        self.runtime = _section(config, "runtime")
        self.policy_server = _section(config, "policy_server")
        self.arm_mode = normalize_arm_mode(self.runtime.get("arm_mode", "dual"))
        if self.wing.get("keep_dims") and self.arm_mode != "dual":
            raise ValueError(
                "wing.keep_dims adapts 14D robot I/O to a model joint subset and therefore "
                "requires runtime.arm_mode='dual'. For native 7D deployment, set keep_dims=null "
                "and choose runtime.arm_mode='left' or 'right'."
            )
        self.action_dim = action_dim_for(self.arm_mode)
        print(f"[runtime] arm_mode={self.arm_mode} (action_dim={self.action_dim})")
        chunk_size = int(self.runtime.get("action_chunk_size", self.wing.get("action_horizon", 32)))
        self.rtc = RealTimeChunkingBuffer(
            chunk_size=chunk_size,
            exp_weight_factor=float(self.runtime.get("exp_weight_factor", 0.5)),
            debug=_bool(self.runtime.get("rtc_debug", False)),
        )
        self.execute_actions = _bool(self.runtime.get("execute_actions", False))
        self.no_robot = _bool(self.runtime.get("no_robot", False))
        self.max_steps = self.runtime.get("max_steps")
        self.max_steps = None if self.max_steps is None else int(self.max_steps)
        self.server_host = _required_text(self.policy_server, "host")
        self.server_port = int(self.policy_server.get("port"))
        self.server_authkey = _required_text(self.policy_server, "authkey")
        self.prompt = _required_text(self.wing, "prompt")
        self.stop_event = threading.Event()
        self.producer_thread: threading.Thread | None = None

        if self.wing.get("keep_dims"):
            expected_pad = np.concatenate(
                [
                    _required_pose(self.robot, "left_init_position"),
                    _required_pose(self.robot, "right_init_position"),
                ]
            )
            configured_pad = np.asarray(self.wing.get("full_action_pad"), dtype=np.float32)
            if configured_pad.shape != (ACTION_DIM_DUAL,) or not np.allclose(
                configured_pad, expected_pad, atol=1e-5, rtol=0.0
            ):
                raise ValueError(
                    "wing.full_action_pad must match the concatenated calibrated "
                    "robot left/right initial poses when keep_dims is configured."
                )

    def _create_robot_controller(self):
        if self.no_robot:
            return NoRobotController(self.config, arm_mode=self.arm_mode)
        return DualPiperController(self.config, arm_mode=self.arm_mode)

    def _validate_policy_metadata(self, policy: WINGRemotePolicyClient) -> None:
        metadata = policy.metadata()
        expected_state_dim = int(metadata.get("proprio_output_dim", self.action_dim))
        expected_action_dim = int(metadata.get("action_output_dim", self.action_dim))
        if expected_state_dim != self.action_dim or expected_action_dim != self.action_dim:
            raise RuntimeError(
                "Policy/server dimension mismatch: "
                f"runtime arm_mode={self.arm_mode!r} uses state/action dim {self.action_dim}, "
                f"but server expects state dim {expected_state_dim} and action dim {expected_action_dim}. "
                "Use --arm-mode dual for a 14-dim checkpoint, or serve a 7-dim checkpoint before using "
                "--arm-mode single --arm-side {left,right}."
            )
        print(
            f"[runtime] policy metadata ok: state_dim={expected_state_dim} "
            f"action_dim={expected_action_dim}"
        )

    def run(self) -> bool:
        ros_operator = DualArmRosOperator(self.config)
        policy = WINGRemotePolicyClient(
            host=self.server_host,
            port=self.server_port,
            authkey=self.server_authkey,
        )
        try:
            self._validate_policy_metadata(policy)
            robot = self._create_robot_controller()
            if not robot.connect():
                return False

            # When the model is partial-arm (keep_dims set in wing:), the
            # padded action keeps the non-controlled arm at full_action_pad
            # (which the user is expected to align with robot.{left,right}_init_position).
            # If the robot is not actually at that init pose on startup, every
            # frame would report a persistent state_tracking_l1 mismatch on the
            # padded dims, eventually tripping the safety stop. Force-enable
            # move_to_initial in this case.
            move_to_initial = _bool(self.runtime.get("move_to_initial", False))
            if self.wing.get("keep_dims") and not move_to_initial:
                print(
                    "[runtime] wing.keep_dims is set (partial-arm model); "
                    "forcing runtime.move_to_initial=true to keep the padded arm "
                    "consistent with full_action_pad."
                )
                move_to_initial = True
            self.rtc.clear()
            self.stop_event.clear()
            if self.execute_actions:
                confirmation = input(
                    "Motion is enabled. Clear the workspace and type MOVE to continue: "
                )
                if confirmation != "MOVE":
                    print("[runtime] motion cancelled by operator.")
                    return False
            else:
                print("[runtime] execute_actions=false; robot commands are disabled.")

            if move_to_initial and self.execute_actions:
                robot.move_initial()
            elif move_to_initial:
                print("[runtime] skipping move_to_initial because action execution is disabled.")

            self.producer_thread = threading.Thread(
                target=self._producer_loop,
                args=(ros_operator, robot, policy),
                daemon=True,
            )
            self.producer_thread.start()
            self._control_loop(ros_operator, robot)
            return not self.stop_event.is_set()
        finally:
            self.stop_event.set()
            if self.producer_thread is not None and self.producer_thread.is_alive():
                self.producer_thread.join(timeout=2.0)
            try:
                policy.reset()
            finally:
                policy.close()

    def _producer_loop(self, ros_operator: DualArmRosOperator, robot: DualPiperController, policy: WINGRemotePolicyClient) -> None:
        rate = ros_operator.rospy.Rate(int(self.runtime.get("rospy_rate", 50)))
        printed_wait = False
        while not self.stop_event.is_set() and not ros_operator.rospy.is_shutdown():
            cursor = self.rtc.get_control_time()
            generation = self.rtc.get_generation()
            if self.rtc.has_chunk(cursor):
                rate.sleep()
                continue

            try:
                images = ros_operator.get_frame()
                if images is None:
                    if not printed_wait:
                        print("[producer] waiting for synchronized ROS frames...")
                        printed_wait = True
                    rate.sleep()
                    continue
                printed_wait = False
                obs = {
                    "images": images,
                    "state": robot.get_status_and_state(),
                    "prompt": self.prompt,
                }
                t0 = time.perf_counter()
                action_chunk = np.asarray(policy.infer(obs), dtype=np.float32)
                infer_dt = time.perf_counter() - t0
                rate_hz = max(int(self.runtime.get("rospy_rate", 50)), 1)
                budget = self.rtc.chunk_size / rate_hz
                tag = "WARN starve" if infer_dt > budget else "ok"
                print(f"[producer] infer {infer_dt * 1000:.0f}ms budget {budget * 1000:.0f}ms ({tag})")
                if action_chunk.ndim != 2 or action_chunk.shape[1] != self.action_dim:
                    raise ValueError(f"Expected action chunk [T,{self.action_dim}], got {action_chunk.shape}")
                if not np.isfinite(action_chunk).all():
                    raise ValueError("Policy returned non-finite actions.")
                action_chunk = action_chunk[: self.rtc.chunk_size]
                self.rtc.enqueue(action_chunk, cursor, generation=generation)
            except Exception as exc:
                print(f"[producer] observation or inference failed: {exc}")
                self.stop_event.set()
                break
            rate.sleep()

    def _control_loop(self, ros_operator: DualArmRosOperator, robot: DualPiperController) -> None:
        rate = ros_operator.rospy.Rate(int(self.runtime.get("rospy_rate", 50)))
        last_action = robot.get_status_and_state()
        print(f"[runtime] start_state={last_action}")
        t = 0
        while not self.stop_event.is_set() and not ros_operator.rospy.is_shutdown():
            if self.max_steps is not None and t >= self.max_steps:
                print(f"[runtime] reached max_steps={self.max_steps}")
                break

            self.rtc.set_control_time(t)
            action = self.rtc.get_action(t)
            if action is None:
                print("[runtime] waiting for action...")
                rate.sleep()
                continue

            action = np.asarray(action, dtype=np.float32)
            if action.shape != (self.action_dim,):
                print(f"[runtime] expected action [{self.action_dim}], got {action.shape}")
                break

            prev_action_l1 = float(np.mean(np.abs(action - last_action)))
            if prev_action_l1 > robot.action_safety_threshold:
                print(f"[runtime] safety stop: action jump too large {prev_action_l1:.3f}")
                break

            current_state = robot.get_status_and_state()
            state_tracking_l1 = float(np.mean(np.abs(current_state - last_action)))
            if state_tracking_l1 > robot.state_safety_threshold:
                print(f"[runtime] safety stop: current state too far from last action {state_tracking_l1:.3f}")
                break

            target_state_l1 = float(np.mean(np.abs(current_state - action)))
            print(
                f"[runtime] t={t} action_l1={prev_action_l1:.3f} "
                f"state_tracking_l1={state_tracking_l1:.3f} target_state_l1={target_state_l1:.3f} "
                f"action={action}"
            )
            if self.execute_actions:
                robot.move(action)
                last_action = action
            elif hasattr(robot, "observe_action"):
                robot.observe_action(action)
                last_action = action
            else:
                last_action = current_state
            t += 1
            rate.sleep()

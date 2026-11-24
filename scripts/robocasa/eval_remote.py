#!/usr/bin/env python3
"""Closed-loop RoboCasa rollout against a WING joint policy server."""

import argparse
import json

import gymnasium as gym
import numpy as np
import robocasa  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

from wing.deployment.wing_remote_policy_client import WINGRemotePolicyClient
from wing.deployment.robocasa import (
    GR1_CONTROLLED_DIM,
    GR1_FULL_DIM,
    wing_action_to_robocasa,
    robocasa_observation_to_policy,
)


DEFAULT_ENV_ID = (
    "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--authkey", default="wing")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--action-horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    env = gym.make(args.env_id, enable_render=not args.no_render)
    results = []
    try:
        with WINGRemotePolicyClient(args.host, args.port, args.authkey) as policy:
            metadata = policy.metadata()
            state_dim = int(metadata["model_proprio_output_dim"])
            action_dim = int(metadata["model_action_output_dim"])
            if state_dim not in (GR1_CONTROLLED_DIM, GR1_FULL_DIM):
                raise ValueError(f"Unsupported RoboCasa policy state dim: {state_dim}.")
            if action_dim not in (GR1_CONTROLLED_DIM, GR1_FULL_DIM):
                raise ValueError(f"Unsupported RoboCasa policy action dim: {action_dim}.")
            if state_dim != action_dim:
                raise ValueError(
                    f"RoboCasa policy state/action dims must match, got {state_dim}/{action_dim}."
                )
            print(json.dumps({"policy_metadata": metadata}), flush=True)
            for episode in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + episode)
                policy.reset()
                success = False
                action_chunk = np.empty((0, action_dim), dtype=np.float32)
                chunk_offset = 0
                step = -1
                for step in range(args.max_steps):
                    if chunk_offset >= min(args.action_horizon, len(action_chunk)):
                        payload = robocasa_observation_to_policy(obs, policy_dim=state_dim)
                        action_chunk = np.asarray(policy.infer(payload), dtype=np.float32)
                        if action_chunk.ndim != 2 or action_chunk.shape[1] != action_dim:
                            raise ValueError(
                                f"Policy must return [T,{action_dim}], got {action_chunk.shape}."
                            )
                        chunk_offset = 0
                    action = action_chunk[chunk_offset]
                    chunk_offset += 1
                    obs, _, terminated, truncated, info = env.step(
                        wing_action_to_robocasa(action)
                    )
                    success = bool(info.get("success", False))
                    if success or terminated or truncated:
                        break
                result = {"episode": episode, "steps": step + 1, "success": success}
                results.append(result)
                print(json.dumps(result), flush=True)
    finally:
        env.close()

    print(json.dumps({
        "episodes": len(results),
        "successes": sum(int(item["success"]) for item in results),
    }), flush=True)


if __name__ == "__main__":
    main()

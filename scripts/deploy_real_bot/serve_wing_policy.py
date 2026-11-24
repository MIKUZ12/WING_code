#!/usr/bin/env python3
"""Start a persistent WING policy server for RTC-style robot deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from wing.deployment.wing_policy_client import WINGPolicyConfig
from wing.deployment.wing_policy_server import serve_wing_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve WING as a persistent RTC-compatible policy backend.")
    parser.add_argument("--config", default="configs/real_deploy_wing.yaml")
    parser.add_argument("--host", default=None, help="Override policy_server.host.")
    parser.add_argument("--port", type=int, default=None, help="Override policy_server.port.")
    parser.add_argument("--authkey", default=None, help="Override policy_server.authkey.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {config_path}")
    policy_config = WINGPolicyConfig.from_mapping(payload, base_dir=config_path.parent)
    endpoint = payload.get("policy_server", {})
    if not isinstance(endpoint, dict):
        raise ValueError("Config section `policy_server` must be a mapping.")
    host = args.host if args.host is not None else endpoint.get("host")
    port = args.port if args.port is not None else endpoint.get("port")
    authkey = args.authkey if args.authkey is not None else endpoint.get("authkey")
    if host is None or port is None or not authkey:
        raise ValueError(
            "Supply policy server host, port, and authkey in the config or CLI."
        )
    serve_wing_policy(policy_config, host=str(host), port=int(port), authkey=authkey)


if __name__ == "__main__":
    main()

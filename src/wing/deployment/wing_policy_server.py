"""Persistent WING policy server for RTC-style deployment."""

from __future__ import annotations

import traceback
from multiprocessing.connection import Listener
from typing import Any, Mapping

from wing.deployment.wing_policy_client import WINGPolicyClient, WINGPolicyConfig


def serve_wing_policy(
    config: WINGPolicyConfig | Mapping[str, Any],
    host: str,
    port: int,
    authkey: str | bytes,
) -> None:
    """Start a simple trusted-network WING policy server.

    The server loads WING once, then accepts RTC-style requests:
    ``update_observation``, ``get_action``, ``infer``, ``reset``, and ``shutdown``.
    """
    if not str(host).strip():
        raise ValueError("Policy server host must be supplied by the operator.")
    if not authkey:
        raise ValueError("Policy server authkey must be supplied by the operator.")
    key = authkey.encode("utf-8") if isinstance(authkey, str) else authkey
    policy = WINGPolicyClient(config)
    listener = Listener((host, int(port)), authkey=key)
    print(f"[wing-policy-server] Listening on {host}:{port}")
    try:
        while True:
            conn = listener.accept()
            print(f"[wing-policy-server] Client connected from {listener.last_accepted}")
            should_shutdown = _serve_connection(policy, conn)
            conn.close()
            if should_shutdown:
                break
    finally:
        listener.close()
        print("[wing-policy-server] Stopped.")


def _serve_connection(policy: WINGPolicyClient, conn) -> bool:
    while True:
        try:
            request = conn.recv()
        except EOFError:
            return False

        if not isinstance(request, Mapping):
            conn.send({"ok": False, "error": f"Expected request mapping, got {type(request)}"})
            continue

        op = request.get("op")
        try:
            if op == "update_observation":
                policy.update_observation(request["obs"])
                conn.send({"ok": True})
            elif op == "get_action":
                conn.send({"ok": True, "action": policy.get_action()})
            elif op == "infer":
                policy.update_observation(request["obs"])
                conn.send({"ok": True, "action": policy.get_action()})
            elif op == "metadata":
                conn.send({"ok": True, "metadata": policy.metadata()})
            elif op == "reset":
                policy.reset()
                conn.send({"ok": True})
            elif op == "shutdown":
                conn.send({"ok": True})
                return True
            else:
                conn.send({"ok": False, "error": f"Unknown operation: {op!r}"})
        except Exception as exc:
            conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})

"""Remote RTC-compatible client for a persistent WING policy server."""

from __future__ import annotations

from multiprocessing.connection import Client as ConnectionClient
from typing import Any, Mapping


class WINGRemotePolicyClient:
    """Small RPC client matching the RTC ``PolicyClient`` interface.

    This transport uses Python pickling through ``multiprocessing.connection``.
    Use it only on a trusted machine or trusted robot LAN.
    """

    def __init__(self, host: str, port: int, authkey: str | bytes):
        if not str(host).strip():
            raise ValueError("Policy server host must be supplied by the operator.")
        if not authkey:
            raise ValueError("Policy server authkey must be supplied by the operator.")
        self.address = (host, int(port))
        self.authkey = authkey.encode("utf-8") if isinstance(authkey, str) else authkey
        self.conn = ConnectionClient(self.address, authkey=self.authkey)

    def close(self) -> None:
        self.conn.close()

    def update_observation(self, obs: Mapping[str, Any]) -> None:
        self._request({"op": "update_observation", "obs": dict(obs)})

    def get_action(self):
        return self._request({"op": "get_action"})["action"]

    def infer(self, obs: Mapping[str, Any]):
        """Convenience one-shot request: update observation and return action chunk."""
        return self._request({"op": "infer", "obs": dict(obs)})["action"]

    def metadata(self) -> dict[str, Any]:
        """Return cheap server-side policy metadata without running inference."""
        return self._request({"op": "metadata"})["metadata"]

    def reset(self) -> None:
        self._request({"op": "reset"})

    def shutdown_server(self) -> None:
        self._request({"op": "shutdown"})

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.conn.send(dict(payload))
        response = self.conn.recv()
        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid WING policy server response: {response!r}")
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "WING policy server request failed."))
        return response

    def __enter__(self) -> "WINGRemotePolicyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

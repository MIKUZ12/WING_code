"""Deployment helpers for RTC-style WING real-robot inference."""

__all__ = [
    "WINGPolicyClient",
    "WINGPolicyConfig",
    "WINGRemotePolicyClient",
    "RealTimeChunkingBuffer",
    "serve_wing_policy",
]


def __getattr__(name):
    if name == "RealTimeChunkingBuffer":
        from wing.deployment.rtc_buffer import RealTimeChunkingBuffer

        return RealTimeChunkingBuffer
    if name in {"WINGPolicyClient", "WINGPolicyConfig"}:
        from wing.deployment.wing_policy_client import WINGPolicyClient, WINGPolicyConfig

        return {"WINGPolicyClient": WINGPolicyClient, "WINGPolicyConfig": WINGPolicyConfig}[name]
    if name == "WINGRemotePolicyClient":
        from wing.deployment.wing_remote_policy_client import WINGRemotePolicyClient

        return WINGRemotePolicyClient
    if name == "serve_wing_policy":
        from wing.deployment.wing_policy_server import serve_wing_policy

        return serve_wing_policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

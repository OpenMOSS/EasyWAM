"""Session-aware EasyWAM policy server for RoboTwin's XPolicyLab client."""

from __future__ import annotations

import argparse
import ast
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _session_id(frame: Any) -> str:
    return str(
        frame.evaluation_id
        or frame.trial_id
        or frame.action_case_id
        or "robotwin-default"
    )


def _parse_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _load_config(path: Path, overrides: list[str]) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Policy config must be a mapping: {path}")
    if len(overrides) % 2:
        raise ValueError("--overrides expects KEY VALUE pairs.")
    for key, value in zip(overrides[::2], overrides[1::2]):
        payload[key.lstrip("-")] = _parse_value(value)
    return payload


def _install_xpolicylab_path(sim_root: Path) -> None:
    xpolicylab_root = sim_root / "XPolicyLab"
    if not (xpolicylab_root / "client_server" / "ws" / "model_server.py").is_file():
        raise FileNotFoundError(
            "The XPolicyLab submodule is missing. Clone the simulator with "
            "--recurse-submodules or run git submodule update --init --recursive."
        )
    sys.path.insert(0, str(sim_root))
    sys.path.insert(0, str(xpolicylab_root))


def _server_class():
    from client_server.ws.model_server import PolicyServer
    from client_server.ws.protocol.messages import MessageType
    from XPolicyLab.utils.process_data import decode_obs_images

    class SessionAwarePolicyServer(PolicyServer):
        async def _invoke(self, frame: Any, command: str, payload: Any = None) -> Any:
            return await asyncio.to_thread(
                self.model.invoke, _session_id(frame), command, payload
            )

        async def _dispatch_frame(self, frame: Any):
            if frame.message_type == MessageType.CALL:
                command = frame.payload.get("func_name")
                if not isinstance(command, str) or not command or command.startswith("_"):
                    raise ValueError(f"Invalid model function: {command!r}")
                payload = frame.payload.get("obs")
                if payload is not None and command in {"update_obs", "update_obs_batch"}:
                    payload = await asyncio.to_thread(decode_obs_images, payload)
                started = time.perf_counter()
                result = await self._invoke(frame, command, payload)
                response = {"ok": True, "latency_ms": (time.perf_counter() - started) * 1000}
                if result is not None:
                    response["result"] = result
                return self._reply(frame, MessageType.CALL_RESULT, response)

            if frame.message_type == MessageType.RESET:
                result = await self._invoke(frame, "reset")
                response = {"ok": True}
                if result is not None:
                    response["result"] = result
                return self._reply(frame, MessageType.RESET_RESULT, response)

            if frame.message_type == MessageType.PREPARE_CASE:
                return self._reply(frame, MessageType.PREPARE_CASE_ACK, {"ok": True})

            if frame.message_type == MessageType.TRIAL_END:
                result = await self._invoke(frame, "trial_end", dict(frame.payload))
                response = {"ok": True}
                if result is not None:
                    response["result"] = result
                return self._reply(frame, MessageType.TRIAL_END_ACK, response)

            if frame.message_type == MessageType.CLOSE:
                close_session = getattr(self.model, "close_session", None)
                if callable(close_session):
                    await asyncio.to_thread(close_session, _session_id(frame))
                return None

            return await super()._dispatch_frame(frame)

    return SessionAwarePolicyServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    sim_root = args.robotwin_root.expanduser().resolve()
    _install_xpolicylab_path(sim_root)
    config = _load_config(args.config.expanduser().resolve(), args.overrides)

    from client_server.ws.model_server import PolicyServerConfig
    from experiments.robotwin.easywam_policy.deploy_policy import get_model

    model = get_model(config)
    server_type = _server_class()
    server = server_type(
        model,
        PolicyServerConfig(host=args.host, port=args.port),
    )
    try:
        asyncio.run(server.serve_forever())
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()

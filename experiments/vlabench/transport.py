"""Small, connection-scoped RPC bridge between independent Python environments.

Only numpy, msgpack and websockets are needed on the simulator side. A new
connection always owns a new policy session; action queues never cross clients.
"""
from __future__ import annotations

import math
import threading

import msgpack
import numpy as np

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _encode(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "buif":
            raise TypeError(f"Unsupported array dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {"__easywam_array__": True, "shape": list(array.shape),
                "dtype": array.dtype.str, "data": array.tobytes()}
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _decode(value):
    if value.get("__easywam_array__") is not True:
        return value
    dtype = np.dtype(value["dtype"])
    shape = value["shape"]
    if (dtype.kind not in "buif" or not isinstance(shape, list) or len(shape) > 6
            or any(type(n) is not int or n < 0 for n in shape)):
        raise ValueError("Invalid array metadata")
    if math.prod(shape) * dtype.itemsize != len(value["data"]):
        raise ValueError("Array byte count does not match shape")
    return np.frombuffer(value["data"], dtype=dtype).reshape(shape).copy()


def pack(value) -> bytes:
    data = msgpack.packb(value, default=_encode, use_bin_type=True)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("RPC message too large")
    return data


def unpack(data):
    if not isinstance(data, bytes) or len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Expected a bounded binary RPC message")
    return msgpack.unpackb(data, object_hook=_decode, raw=False)


class PolicyClient:
    def __init__(self, url: str, *, timeout: float = 300):
        from websockets.sync.client import connect

        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        self.connection = connect(url, open_timeout=self.timeout, close_timeout=5,
                                  max_size=MAX_MESSAGE_BYTES, compression=None)
        try:
            hello = unpack(self.connection.recv(timeout=self.timeout))
            if hello.get("protocol") != PROTOCOL_VERSION:
                raise ValueError("Incompatible EasyWAM policy protocol")
        except BaseException:
            self.connection.close()
            raise
        self._lock = threading.Lock()

    def call(self, command: str, payload=None):
        with self._lock:
            self.connection.send(pack({"command": command, "payload": payload}))
            try:
                response = unpack(self.connection.recv(timeout=self.timeout))
            except BaseException:
                # A late response cannot be mistaken for the next request.
                self.connection.close()
                raise
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Policy server failed"))
            return response.get("result")

    def reset(self, seed=None):
        return self.call("reset", {"seed": seed})

    def predict(self, payload):
        return self.call("predict", payload)

    def close(self):
        self.connection.close()


def create_server(session_factory, *, host="127.0.0.1", port=8000):
    from websockets.sync.server import serve

    def handler(connection):
        session = session_factory()
        try:
            connection.send(pack({"protocol": PROTOCOL_VERSION}))
            for raw in connection:
                try:
                    request = unpack(raw)
                    command, payload = request["command"], request.get("payload")
                    if command == "reset":
                        session.reset(**(payload or {}))
                        result = None
                    elif command == "predict":
                        result = session.predict(payload)
                    elif command == "ping":
                        result = "pong"
                    else:
                        raise ValueError(f"Unknown command: {command!r}")
                    response = {"ok": True, "result": result}
                except Exception as error:
                    response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                connection.send(pack(response))
        finally:
            session.close()

    return serve(handler, host, port, max_size=MAX_MESSAGE_BYTES, compression=None,
                 close_timeout=5)

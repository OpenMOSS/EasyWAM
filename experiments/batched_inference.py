from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import torch

from experiments.prompt_context_cache import PromptContextCache


@dataclass
class _Request:
    operation: str
    prompt: str
    image: torch.Tensor
    proprio: torch.Tensor | None
    seed: int | None
    kwargs: dict[str, Any]
    key: tuple[Any, ...]
    submitted_at: float
    future: Future


class DynamicInferenceBatcher:
    def __init__(
        self,
        model,
        *,
        max_batch_size: int,
        wait_ms: float,
        prompt_cache_size: int,
    ) -> None:
        if max_batch_size <= 0 or wait_ms < 0:
            raise ValueError("max_batch_size must be positive and wait_ms must be non-negative.")
        self.model = model
        self.max_batch_size = int(max_batch_size)
        self.wait_seconds = float(wait_ms) / 1000.0
        self.prompt_cache = PromptContextCache(model, max_entries=prompt_cache_size)
        self._queue: queue.Queue[_Request | None] = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="dynamic-inference", daemon=True)
        self._lock = threading.Lock()
        self._batches = 0
        self._requests = 0
        self._queue_seconds = 0.0
        self._inference_seconds = 0.0
        self._thread.start()

    @staticmethod
    def _key(operation: str, image: torch.Tensor, proprio: torch.Tensor | None, kwargs: dict[str, Any]):
        static_values = tuple(sorted((key, repr(value)) for key, value in kwargs.items()))
        return (
            operation,
            tuple(image.shape[-3:]),
            None if proprio is None else tuple(proprio.shape[-1:]),
            static_values,
        )

    def submit(
        self,
        operation: str,
        *,
        prompt: str,
        input_image: torch.Tensor,
        proprio: torch.Tensor | None,
        seed: int | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if operation not in {"action", "joint"}:
            raise ValueError(f"Unsupported inference operation: {operation}")
        if input_image.ndim == 4 and input_image.shape[0] == 1:
            input_image = input_image[0]
        if input_image.ndim != 3:
            raise ValueError(f"Request image must be [C,H,W], got {tuple(input_image.shape)}")
        if input_image.shape[0] != 3:
            raise ValueError(f"Request image must have 3 channels, got {input_image.shape[0]}")
        if proprio is not None and proprio.ndim == 2 and proprio.shape[0] == 1:
            proprio = proprio[0]
        if proprio is not None and proprio.ndim != 1:
            raise ValueError(f"Request proprio must be [D], got {tuple(proprio.shape)}")
        future: Future = Future()
        request = _Request(
            operation=operation,
            prompt=str(prompt),
            image=input_image.detach().to(device="cpu", dtype=torch.float32),
            proprio=None if proprio is None else proprio.detach().to(device="cpu", dtype=torch.float32),
            seed=seed,
            kwargs=dict(kwargs),
            key=self._key(operation, input_image, proprio, kwargs),
            submitted_at=time.perf_counter(),
            future=future,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("DynamicInferenceBatcher is closed.")
            self._queue.put(request)
        return future.result()

    def _collect(self, first: _Request) -> list[_Request]:
        batch = [first]
        deferred = []
        stop_seen = False
        deadline = time.perf_counter() + self.wait_seconds
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                stop_seen = True
                break
            if item.key == first.key:
                batch.append(item)
            else:
                deferred.append(item)
        for item in deferred:
            self._queue.put(item)
        if stop_seen:
            self._queue.put(None)
        return batch

    def _execute(self, requests: list[_Request]) -> None:
        execution_started = time.perf_counter()
        queue_seconds = sum(execution_started - request.submitted_at for request in requests)
        images = torch.stack([request.image for request in requests]).to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        proprio = None
        if requests[0].proprio is not None:
            if any(request.proprio is None for request in requests):
                raise ValueError("Cannot batch requests with mixed proprio presence.")
            proprio = torch.stack([request.proprio for request in requests]).to(
                device=self.model.device, dtype=self.model.torch_dtype
            )
        prompt_encode_before = self.prompt_cache.encode_seconds
        context, context_mask = self.prompt_cache.get_many(
            [request.prompt for request in requests]
        )
        prompt_encode_per_request = (
            self.prompt_cache.encode_seconds - prompt_encode_before
        ) / len(requests)
        common = dict(requests[0].kwargs)
        common.update(
            prompt=None,
            context=context,
            context_mask=context_mask,
            input_image=images,
            proprio=proprio,
            seed=[request.seed for request in requests],
        )
        method = (
            self.model.infer_action_batch
            if requests[0].operation == "action"
            else self.model.infer_joint_batch
        )
        started = time.perf_counter()
        result = method(**common)
        elapsed = time.perf_counter() - started
        actions = result["action"]
        videos = result.get("video")
        if videos is not None and videos and not isinstance(videos[0], list):
            videos = [videos]
        if actions.ndim < 1 or actions.shape[0] != len(requests):
            raise ValueError("Model action output batch size does not match the request batch.")
        if videos is not None and len(videos) != len(requests):
            raise ValueError("Model video output batch size does not match the request batch.")
        for index, request in enumerate(requests):
            value = {
                "action": actions[index],
                "prompt_encode_seconds": prompt_encode_per_request,
            }
            if videos is not None:
                value["video"] = videos[index]
            request.future.set_result(value)
        with self._lock:
            self._batches += 1
            self._requests += len(requests)
            self._queue_seconds += queue_seconds
            self._inference_seconds += elapsed

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            requests = self._collect(first)
            try:
                self._execute(requests)
            except BaseException as error:
                for request in requests:
                    request.future.set_exception(error)

    def stats(self) -> dict[str, float]:
        with self._lock:
            requests = self._requests
            batches = self._batches
            return {
                "requests": float(requests),
                "batches": float(batches),
                "mean_batch_size": requests / batches if batches else 0.0,
                "mean_queue_seconds": self._queue_seconds / requests if requests else 0.0,
                "inference_seconds": self._inference_seconds,
                "prompt_encode_seconds": self.prompt_cache.encode_seconds,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._thread.join()
        logging.info("Dynamic inference stats: %s", self.stats())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

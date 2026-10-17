"""Hydra launcher for independent VLABench simulators and batched model servers."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
from omegaconf import OmegaConf

from experiments.task_dispatch import build_worker_slots
from experiments.vlabench.registry import build_jobs, digest, file_digest, upstream_identity
from experiments.vlabench.result_utils import is_complete, read_result, result_path, summarize, write_json_atomic
from utils.config_resolvers import register_default_resolvers
from experiments.vlabench.contract import CONTRACT_VERSION

register_default_resolvers()


def _stats_path(cfg, checkpoint):
    if cfg.EVALUATION.dataset_stats_path:
        path = Path(str(cfg.EVALUATION.dataset_stats_path)).expanduser().resolve()
    else:
        path = next((parent / "dataset_stats.json" for parent in checkpoint.parents
                     if (parent / "dataset_stats.json").is_file()), None)
    if path is None or not path.is_file():
        raise FileNotFoundError("Set EVALUATION.dataset_stats_path to the training dataset_stats.json")
    return path


def make_manifest(cfg, jobs, upstream):
    checkpoint = Path(str(cfg.ckpt)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    stats = _stats_path(cfg, checkpoint)
    ev = OmegaConf.to_container(cfg.EVALUATION, resolve=True)
    # Placement, concurrency and output paths do not change experiment identity.
    for name in ("output_dir", "vlabench_root", "sim_python", "render_backend", "video_mode",
                 "startup_timeout_seconds", "episode_timeout_seconds", "request_timeout_seconds"):
        ev.pop(name, None)
    ev["dataset_stats_path"] = None
    identity = {"contract_version": CONTRACT_VERSION, "upstream": upstream,
                "checkpoint_sha256": file_digest(checkpoint), "stats_sha256": file_digest(stats),
                "seed": int(cfg.seed), "mixed_precision": str(cfg.mixed_precision),
                "model": OmegaConf.to_container(cfg.model, resolve=True),
                "processor": OmegaConf.to_container(cfg.data.train.processor, resolve=True),
                "execution": {"vae_micro_batch_size": cfg.vae_micro_batch_size,
                              "inference_cross_kv_reuse": cfg.inference_cross_kv_reuse},
                "evaluation": ev, "jobs_digest": digest(jobs)}
    return {"fingerprint": digest(identity), "identity": identity,
            "checkpoint": str(checkpoint), "dataset_stats": str(stats)}


def _free_port():
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def _stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _wait_ready(process, url, timeout):
    from experiments.vlabench.transport import PolicyClient
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited with code {process.returncode}")
        try:
            client = PolicyClient(url, timeout=1)
            try:
                if client.call("ping") == "pong":
                    return
            finally:
                client.close()
        except (OSError, TimeoutError):
            time.sleep(.2)
    raise TimeoutError(f"Server startup timed out: {url}")


def run(cfg):
    ev, multi = cfg.EVALUATION, cfg.MULTIRUN
    output = Path(str(ev.output_dir)).expanduser().resolve()
    root = Path(str(ev.vlabench_root)).expanduser().resolve()
    jobs = build_jobs(root, list(ev.tracks), None if ev.tasks is None else list(ev.tasks),
                      num_trials=ev.num_trials, seed=int(cfg.seed),
                      custom_track_file=ev.custom_track_file)
    upstream = upstream_identity(root)
    if bool(multi.create_only):
        if (output / "manifest.json").exists():
            raise ValueError("create_only requires an output directory without an existing experiment")
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output / "tasks.json", jobs)
        write_json_atomic(output / "upstream.json", upstream)
        OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), output / "manager_config.yaml")
        return {"created_jobs": len(jobs), "output_dir": str(output)}

    if cfg.ckpt is None:
        raise ValueError("ckpt is required except in create_only mode")
    manifest = make_manifest(cfg, jobs, upstream)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Output directory belongs to another experiment; choose a new output_dir")
    fingerprint = manifest["fingerprint"]
    # Validate resource settings before writing any run artifacts.
    actor_count = int(multi.env_num_per_worker)
    batch_size = int(multi.inference_batch_size)
    if actor_count <= 0 or not 0 < batch_size <= actor_count:
        raise ValueError("Require 0 < inference_batch_size <= env_num_per_worker")
    if ev.video_mode not in {"none", "all", "failures"}:
        raise ValueError("video_mode must be none, all or failures")
    if ev.render_backend not in {"osmesa", "egl", "glfw"}:
        raise ValueError("render_backend must be osmesa, egl or glfw")
    if any(float(ev[key]) <= 0 for key in ("startup_timeout_seconds", "episode_timeout_seconds", "request_timeout_seconds")):
        raise ValueError("Timeouts must be positive")
    pending = [job for job in jobs if not is_complete(output, job, fingerprint)]
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(output / "tasks.json", jobs)
    OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), output / "manager_config.yaml")
    if not pending:
        result = summarize(output, jobs, fingerprint)
        if not result["overall"]["complete"]:
            raise RuntimeError(f"VLABench run incomplete; inspect {output / 'summary.json'}")
        return result
    if str(ev.device) == "cpu":
        slots = build_worker_slots(num_gpus=1, workers_per_gpu=int(multi.workers_per_gpu), pending_jobs=len(pending))
    else:
        slots = build_worker_slots(num_gpus=int(multi.num_gpus), gpu_ids=multi.gpu_ids,
                                   workers_per_gpu=int(multi.workers_per_gpu), pending_jobs=len(pending))
    (output / "workers").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    processes, handles, urls = [], [], []
    simulator_processes = set()
    simulator_lock = threading.Lock()
    stopping = threading.Event()

    def run_one(job, url):
        destination = result_path(output, job)
        key = f"{job['track']}_{job['task_name']}_{job['episode_id']:04d}"
        job_file = output / "workers" / f"{key}.json"
        write_json_atomic(job_file, job)
        command = [str(ev.sim_python), str(ROOT / "experiments/vlabench/eval_vlabench_episode.py"),
                   "--root", str(root), "--job", str(job_file), "--result", str(destination),
                   "--server", url, "--fingerprint", fingerprint,
                   "--request-timeout", str(ev.request_timeout_seconds),
                   "--gripper-threshold", str(ev.gripper_threshold), "--gripper-width", str(ev.gripper_width),
                   "--video-mode", str(ev.video_mode)]
        if ev.episode_horizon is not None:
            command.extend(["--horizon", str(ev.episode_horizon)])
        environment = os.environ.copy()
        environment.update(MUJOCO_GL=str(ev.render_backend), PYTHONUNBUFFERED="1",
                           CUDA_VISIBLE_DEVICES="", VLABENCH_ROOT=str(root / "VLABench"))
        error = None
        for attempt in range(2):
            if stopping.is_set():
                return
            with (output / "logs" / f"{key}.log").open("a") as log:
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    with simulator_lock:
                        simulator_processes.add(process)
                    try:
                        code = process.wait(timeout=float(ev.episode_timeout_seconds))
                    except subprocess.TimeoutExpired:
                        _stop(process)
                        raise TimeoutError("Simulator episode timed out")
                    finally:
                        with simulator_lock:
                            simulator_processes.discard(process)
                    if code:
                        raise RuntimeError(f"Simulator exit code {code}; see {key}.log")
                    result = read_result(output, job, fingerprint)
                    if result is None or result.get("status") != "completed":
                        raise RuntimeError("Simulator did not write a valid episode result")
                    return
                except (OSError, RuntimeError, TimeoutError) as exception:
                    error = f"{type(exception).__name__}: {exception}"
        write_json_atomic(destination, {**{k: v for k, v in job.items() if k != "episode_config"},
                                        "status": "error", "error": error, "attempts": 2,
                                        "fingerprint": fingerprint})

    try:
        for slot in slots:
            config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            config.ckpt = manifest["checkpoint"]
            config.EVALUATION.dataset_stats_path = manifest["dataset_stats"]
            config.EVALUATION.device = "cpu" if str(ev.device) == "cpu" else "cuda:0"
            config_path = output / "workers" / f"model_{slot.worker_index}.yaml"
            OmegaConf.save(config, config_path)
            port = _free_port()
            log = (output / "logs" / f"model_{slot.worker_index}.log").open("a")
            handles.append(log)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES="" if str(ev.device) == "cpu" else str(slot.gpu_id),
                       PYTHONUNBUFFERED="1")
            process = subprocess.Popen([sys.executable, str(ROOT / "experiments/vlabench/policy_server.py"),
                                        "--config", str(config_path), "--port", str(port)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            url = f"ws://127.0.0.1:{port}"
            _wait_ready(process, url, float(ev.startup_timeout_seconds))
            urls.extend([url]*actor_count)
        # Each actor keeps one server assignment. Episode processes isolate native
        # simulator crashes; model weights remain resident throughout the run.
        import queue
        queue_jobs = queue.Queue()
        for job in pending:
            queue_jobs.put(job)

        def actor(url):
            while not stopping.is_set():
                try:
                    job = queue_jobs.get_nowait()
                except queue.Empty:
                    return
                if any(process.poll() is not None for process in processes):
                    stopping.set()
                    raise RuntimeError("A model server exited; inspect model logs")
                run_one(job, url)

        with ThreadPoolExecutor(max_workers=len(urls)) as executor:
            futures = [executor.submit(actor, url) for url in urls]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                stopping.set()
                with simulator_lock:
                    for process in list(simulator_processes):
                        _stop(process)
                for future in futures:
                    future.cancel()
                raise
    finally:
        stopping.set()
        with simulator_lock:
            for process in list(simulator_processes):
                _stop(process)
        for process in processes:
            _stop(process)
        for handle in handles:
            handle.close()
        summarize(output, jobs, fingerprint)
    result = summarize(output, jobs, fingerprint)
    if not result["overall"]["complete"]:
        raise RuntimeError(f"VLABench run incomplete; inspect {output / 'summary.json'}")
    return result


@hydra.main(version_base="1.3", config_path="../../configs", config_name="benchmark/sim_vlabench")
def main(cfg):
    print(json.dumps(run(cfg), ensure_ascii=False))


if __name__ == "__main__":
    main()

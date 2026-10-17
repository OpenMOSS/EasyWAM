"""Isolated simulator process for one official episode (no model imports)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--server", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--gripper-threshold", type=float, default=.5)
    parser.add_argument("--gripper-width", type=float, default=.04)
    parser.add_argument("--video-mode", choices=("none", "failures", "all"), default="none")
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()))
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ["VLABENCH_ROOT"] = str(args.root.resolve() / "VLABench")

    from experiments.vlabench.episode_runner import run_episode
    from experiments.vlabench.policy import EasyWAMVLABenchPolicy
    from experiments.vlabench.result_utils import write_json_atomic

    job = json.loads(args.job.read_text())
    policy = EasyWAMVLABenchPolicy(args.server, timeout=args.request_timeout,
                                  gripper_threshold=args.gripper_threshold,
                                  gripper_width=args.gripper_width)
    started = time.monotonic()
    # Stream to a temporary video instead of retaining a full rollout in RAM.
    writer = None
    video_path = args.result.with_suffix(".mp4")
    temporary_video = video_path.with_name(video_path.stem + ".partial.mp4")
    try:
        if args.video_mode != "none":
            import imageio.v2 as imageio
            args.result.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(temporary_video), fps=10, macro_block_size=1)

        def capture(rgb):
            import numpy as np
            writer.append_data(np.vstack((np.hstack(rgb[:2]), np.hstack(rgb[2:4]))))

        result = run_episode(policy, job["task_name"], job["episode_config"],
                             seed=job["seed"], horizon=args.horizon, capture_frame=capture if writer else None)
        if writer:
            writer.close()
            writer = None
            if args.video_mode == "all" or not result["success"]:
                os.replace(temporary_video, video_path)
        write_json_atomic(args.result, {**{k: v for k, v in job.items() if k != "episode_config"},
                                        **result, "status": "completed", "fingerprint": args.fingerprint,
                                        "duration_seconds": time.monotonic()-started})
    finally:
        if writer:
            writer.close()
        temporary_video.unlink(missing_ok=True)
        policy.close()


if __name__ == "__main__":
    main()

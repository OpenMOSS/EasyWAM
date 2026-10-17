"""Run a model server in the EasyWAM environment; no VLABench imports here."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main():
    from omegaconf import OmegaConf
    from experiments.vlabench.runtime import VLABenchRuntime
    from experiments.vlabench.transport import create_server

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    runtime = VLABenchRuntime(OmegaConf.load(args.config))
    try:
        with create_server(runtime.new_session, host=args.host, port=args.port) as server:
            server.serve_forever()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

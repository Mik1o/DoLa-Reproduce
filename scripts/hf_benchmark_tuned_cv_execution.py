"""Synthetic benchmark for tuned CV execution strategy accounting."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark legacy config-summed warmup vs per-sample union-layer reuse.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--sleep-per-layer-ms", type=float, default=5.0)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    dynamic_buckets = config["dynamic_bucket_candidates"]
    static_layers = [int(layer) for layer in config["static_layer_candidates"]]

    legacy_validation_factor = sum(len(bucket["candidate_premature_layers"]) for bucket in dynamic_buckets) + len(static_layers)
    union_validation_factor = len(
        {
            *static_layers,
            *(int(layer) for bucket in dynamic_buckets for layer in bucket["candidate_premature_layers"]),
        }
    )

    def run_simulated(layer_factor: int) -> dict[str, float | int]:
        start = time.perf_counter()
        for _ in range(args.samples):
            time.sleep((args.sleep_per_layer_ms / 1000.0) * layer_factor)
        duration = time.perf_counter() - start
        return {
            "samples": args.samples,
            "layer_factor": layer_factor,
            "total_wall_time_seconds": duration,
            "avg_seconds_per_sample": duration / args.samples,
            "forward_count": args.samples,
            "cache_hits": 0,
        }

    legacy = run_simulated(legacy_validation_factor)
    online = run_simulated(union_validation_factor)
    payload = {
        "benchmark_type": "synthetic_stage_strategy",
        "config": str(args.config),
        "sleep_per_layer_ms": args.sleep_per_layer_ms,
        "legacy_config_summed_warmup": legacy,
        "per_sample_online_union_reuse": online,
        "speedup_ratio": legacy["total_wall_time_seconds"] / max(online["total_wall_time_seconds"], 1e-9),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

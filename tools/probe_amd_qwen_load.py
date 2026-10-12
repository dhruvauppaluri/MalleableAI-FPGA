"""Bounded diagnostic for Qwen3 CPU load, GPU transfer, and one forward pass.

This does no training or quality scoring and never opens a held-out suite.
Run under an external `timeout` so a blocked native operation cannot spend
GPU time indefinitely. The exclusive output directory preserves phase events
even if the process is force-killed.
"""

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
import traceback


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    model_path = Path(args.model).resolve(strict=True)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_new(output / "claim.json", {
        "kind": "amd-qwen-load-probe", "model_path": str(model_path),
        "source_script_sha256": source_sha256,
        "heldout_opened": False, "release_quality_claim": False,
        "automatic_switching_allowed": False,
    })
    events = output / "events.jsonl"

    def event(phase, **details):
        payload = {"phase": phase,
                   "elapsed_seconds": round(time.monotonic() - started, 3),
                   "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   **details}
        with events.open("a") as stream:
            stream.write(json.dumps(payload, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(payload, allow_nan=False), flush=True)

    faulthandler.enable()
    faulthandler.dump_traceback_later(20, repeat=True, file=sys.stderr)
    try:
        event("importing_libraries")
        import torch
        from transformers import AutoModelForCausalLM
        event("libraries_imported", torch=torch.__version__, hip=torch.version.hip)
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm/CUDA GPU unavailable")
        event("gpu_detected", gpu=torch.cuda.get_device_name(0),
              gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        torch.manual_seed(20261013)
        random.seed(20261013)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        event("load_settings_set", deterministic_algorithms=True,
              allow_tf32=False)
        event("cpu_load_started")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32)
        event("cpu_load_completed")
        model.eval()
        event("gpu_transfer_started")
        model = model.to(torch.device("cuda"))
        torch.cuda.synchronize()
        event("gpu_transfer_completed",
              gpu_allocated_bytes=torch.cuda.memory_allocated())
        event("forward_started")
        with torch.inference_mode():
            output_logits = model(
                input_ids=torch.tensor([[1, 2, 3, 4]], device="cuda"),
                use_cache=False).logits
            torch.cuda.synchronize()
        event("forward_completed", shape=list(output_logits.shape))
        write_new(output / "result.json", {
            "state": "completed", "source_script_sha256": source_sha256,
            "elapsed_seconds": time.monotonic() - started,
            "heldout_opened": False, "release_quality_claim": False,
            "automatic_switching_allowed": False,
        })
    except BaseException as error:
        event("failed", error_type=type(error).__name__, error=str(error))
        write_new(output / "failure.json", {
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "heldout_opened": False,
        })
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()

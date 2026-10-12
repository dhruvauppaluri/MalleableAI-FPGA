"""Record bounded ROCm host-to-GPU copy health before paid model work.

Run under an external timeout. Set HSA_ENABLE_SDMA before launching Python;
changing it after torch is imported does not provide a valid comparison.
This probe opens no model, data, or held-out suite.
"""

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size-mib", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.size_mib <= 256:
        parser.error("--size-mib must be between 1 and 256")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sdma = os.environ.get("HSA_ENABLE_SDMA", "unset")
    write_new(output / "claim.json", {
        "kind": "amd-host-to-gpu-transfer-probe",
        "source_script_sha256": source_sha256,
        "hsa_enable_sdma": sdma,
        "size_mib": args.size_mib,
        "heldout_opened": False,
        "release_quality_claim": False,
        "automatic_switching_allowed": False,
    })
    events = output / "events.jsonl"

    def event(phase, **details):
        record = {"phase": phase,
                  "elapsed_seconds": round(time.monotonic() - started, 3),
                  "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  **details}
        with events.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record, allow_nan=False), flush=True)

    faulthandler.enable()
    faulthandler.dump_traceback_later(10, repeat=True, file=sys.stderr)
    try:
        event("importing_torch", hsa_enable_sdma=sdma)
        import torch
        event("torch_imported", torch=torch.__version__, hip=torch.version.hip)
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm/CUDA GPU unavailable")
        event("gpu_detected", gpu=torch.cuda.get_device_name(0),
              gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        event("tiny_copy_started")
        tiny = torch.tensor([1.0], dtype=torch.float32).to("cuda")
        torch.cuda.synchronize()
        event("tiny_copy_completed", bytes=tiny.numel() * tiny.element_size())
        values = torch.ones(args.size_mib * 1024 * 1024 // 4,
                            dtype=torch.float32)
        event("bulk_copy_started", size_mib=args.size_mib)
        device_values = values.to("cuda")
        torch.cuda.synchronize()
        event("bulk_copy_completed", bytes=device_values.numel() * 4)
        write_new(output / "result.json", {
            "state": "completed", "source_script_sha256": source_sha256,
            "hsa_enable_sdma": sdma,
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

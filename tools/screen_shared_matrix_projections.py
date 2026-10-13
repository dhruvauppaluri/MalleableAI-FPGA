"""Exact-identity CUDA development screen for one Qwen3 layer pair.

This measures gate-only and up-only 12→13 copies. It does not train, open a
release held-out suite, or repeat the earlier both-projection pair screen.
"""

import argparse
import json
import os
from pathlib import Path
import time
import traceback

from run_shared_matrix_finetune import digest, documents, sequences, write_new
from screen_shared_matrix_pairs import score


SOURCE_LAYER = 12
TARGET_LAYER = 13
PANELS = (("gate_proj",), ("up_proj",))
DEV_FILE_SHA256 = "ec4a17fa316b6bdb4448aa3681ee6779f79bfbbb3ba090a2b5155cd52ff12"
DEV_TOKEN_SHA256 = "927d7b5d36ac9540b37f952f6ad3fbb03d304f19bb0739b7d276d52f057ab536"
MODEL_SHA256 = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
}
MAX_WALL_SECONDS = 120


def choose_panel(rows):
    """Select only a development panel meeting both teacher-facing limits."""
    eligible = [row for row in rows if row["teacher_agreement"] >= 0.90
                and row["nll_change_percent"] <= 5.0]
    return min(eligible, key=lambda row: (
        -row["teacher_agreement"], row["nll"], row["panel"])) if eligible else None


def run(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("this screen requires an NVIDIA CUDA PyTorch runtime")
    model_path = Path(args.model).resolve(strict=True)
    dev_path = Path(args.dev_jsonl).resolve(strict=True)
    if digest(dev_path) != DEV_FILE_SHA256:
        raise ValueError("development file differs from the published frozen corpus")
    for name, expected in MODEL_SHA256.items():
        if digest(model_path / name) != expected:
            raise ValueError(f"pinned model file differs: {name}")
    config = AutoConfig.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen3" or config.num_hidden_layers <= TARGET_LAYER:
        raise ValueError("model lacks the screened Qwen3 layer pair")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        use_fast=True)
    dev, token_sha = sequences(tokenizer, documents(dev_path), 32, 128)
    if token_sha != DEV_TOKEN_SHA256:
        raise ValueError("development tokens differ from the published freeze")

    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    claim = {
        "schema_version": 1,
        "kind": "qwen3-pair-12-13-single-projection-development-screen",
        "source_script_sha256": digest(__file__),
        "score_script_sha256": digest(Path(__file__).with_name(
            "screen_shared_matrix_pairs.py")),
        "model_files_sha256": MODEL_SHA256,
        "dev_file_sha256": DEV_FILE_SHA256,
        "dev_token_sha256": DEV_TOKEN_SHA256,
        "source_layer": SOURCE_LAYER,
        "target_layer": TARGET_LAYER,
        "panels": [list(panel) for panel in PANELS],
        "selection": "teacher agreement >=90%, NLL increase <=5%; rank by agreement then NLL",
        "max_wall_seconds": MAX_WALL_SECONDS,
        "heldout_opened": False,
        "release_quality_claim": False,
        "automatic_switching_allowed": False,
    }
    write_new(root / "claim.json", claim)
    started = time.monotonic()
    deadline = started + MAX_WALL_SECONDS

    def progress(phase, **details):
        event = {"phase": phase, "elapsed_seconds": round(time.monotonic() - started, 3),
                 **details}
        with (root / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(event, allow_nan=False), flush=True)

    try:
        torch.manual_seed(20261013)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        amp = bool(torch.cuda.is_bf16_supported())
        progress("loading_model")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True,
            trust_remote_code=False, dtype=torch.float32).to("cuda")
        model.eval()
        progress("model_loaded", gpu=torch.cuda.get_device_name(0),
                 gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        batches = [torch.tensor([row], device="cuda", dtype=torch.long)
                   for row in dev]
        baseline, reference = score(model, batches, None, amp, deadline)
        progress("baseline_scored", **baseline)
        rows = []
        for panel in PANELS:
            name = "+".join(panel)
            progress("panel_started", panel=name)
            source = model.model.layers[SOURCE_LAYER].mlp
            target = model.model.layers[TARGET_LAYER].mlp
            originals = {part: getattr(target, part).weight.detach().clone()
                         for part in panel}
            try:
                with torch.no_grad():
                    for part in panel:
                        getattr(target, part).weight.copy_(
                            getattr(source, part).weight)
                metrics, _ = score(model, batches, reference, amp, deadline)
            finally:
                with torch.no_grad():
                    for part, original in originals.items():
                        getattr(target, part).weight.copy_(original)
                if any(not torch.equal(getattr(target, part).weight, original)
                       for part, original in originals.items()):
                    raise RuntimeError("model weights were not restored")
            row = {"panel": name, "projections": list(panel), **metrics,
                   "nll_change_percent": 100 * (
                       metrics["nll"] / baseline["nll"] - 1),
                   "accuracy_change_points": 100 * (
                       metrics["next_token_accuracy"] -
                       baseline["next_token_accuracy"])}
            rows.append(row)
            progress("panel_completed", **row)
        selected = choose_panel(rows)
        report = {
            "schema_version": 1,
            "kind": "qwen3-pair-12-13-single-projection-development-result",
            "claim_sha256": digest(root / "claim.json"),
            "baseline": baseline,
            "panels": rows,
            "selected_panel": selected["panel"] if selected else None,
            "wall_seconds": time.monotonic() - started,
            "software": {"torch": torch.__version__,
                         "transformers": __import__("transformers").__version__,
                         "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(0),
                         "bf16_autocast": amp},
            "heldout_opened": False,
            "release_quality_claim": False,
            "automatic_switching_allowed": False,
            "limitations": [
                "Development selection only; no release held-out evaluation.",
                "CUDA and ROCm may differ numerically; selected training must score on Radeon.",
                "No training, FPGA timing or automatic switching measured.",
            ],
        }
        write_new(root / "result.json", report)
        progress("completed", selected_panel=report["selected_panel"])
    except BaseException as error:
        progress("failed", error_type=type(error).__name__, error=str(error))
        write_new(root / "failure.json", {
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "heldout_opened": False,
        })
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dev-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Zero-step Qwen3 gate/up copy screen on the frozen development corpus.

This is a development diagnostic. It neither trains nor opens a release
held-out suite. Every attempt needs a new output directory.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from run_shared_matrix_finetune import digest, load_inputs, write_new


PROJECTIONS = ("gate_proj", "up_proj")
MAX_WALL_SECONDS = 300


def adjacent_pairs(layer_count):
    if not isinstance(layer_count, int) or layer_count < 2:
        raise ValueError("at least two model layers are required")
    return [(source, source + 1) for source in range(layer_count - 1)]


def rank_rows(rows):
    """Lower development NLL wins; accuracy and layer index break ties."""
    return sorted(rows, key=lambda row: (
        row["nll"], -row["next_token_accuracy"], row["source_layer"]))


def score(model, batches, reference, amp, deadline):
    import torch
    import torch.nn.functional as F

    model.eval()
    nll_sum = correct = agree = count = 0
    predictions = [] if reference is None else None
    with torch.inference_mode():
        for index, batch in enumerate(batches):
            if time.monotonic() > deadline:
                raise TimeoutError("development screen wall budget exhausted")
            inputs, targets = batch[:, :-1], batch[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(input_ids=inputs, use_cache=False).logits
            prediction = logits.argmax(-1)
            nll_sum += F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1), reduction="sum").item()
            correct += int(prediction.eq(targets).sum())
            count += targets.numel()
            if reference is None:
                predictions.append(prediction.detach().clone())
            else:
                agree += int(prediction.eq(reference[index]).sum())
    return {
        "nll": nll_sum / count,
        "next_token_accuracy": correct / count,
        "teacher_agreement": 1.0 if reference is None else agree / count,
        "targets": count,
    }, predictions


def run(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    freeze_path = Path(args.freeze).resolve(strict=True)
    freeze = json.loads(freeze_path.read_text())
    if (freeze.get("kind") != "qwen3-pair-shared-real-text-development-pilot"
            or freeze.get("heldout_opened") is not False
            or freeze.get("release_quality_claim") is not False):
        raise ValueError("requires an unopened development-only Qwen3 freeze")
    runner = Path(__file__).with_name("run_shared_matrix_finetune.py")
    if freeze["source_script_sha256"] != digest(runner):
        raise ValueError("frozen input runner no longer matches its source")
    config = AutoConfig.from_pretrained(
        freeze["model_path"], local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen3":
        raise ValueError("requires Qwen3 model")
    pairs = adjacent_pairs(config.num_hidden_layers)
    _, dev = load_inputs(freeze)  # Rechecks checkpoint, corpus and token hashes.
    if len(dev) != freeze["recipe"]["dev_sequences"]:
        raise ValueError("frozen development sequence count changed")
    if not torch.cuda.is_available():
        raise RuntimeError("the bounded screen requires a ROCm/CUDA GPU")

    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "claim.json", {
        "schema_version": 1,
        "kind": "qwen3-adjacent-gate-up-copy-development-screen",
        "source_script_sha256": digest(__file__),
        "input_freeze_sha256": digest(freeze_path),
        "dev_file_sha256": freeze["dev_file_sha256"],
        "dev_token_sha256": freeze["dev_token_sha256"],
        "pairs": [{"source_layer": a, "target_layer": b} for a, b in pairs],
        "projections": list(PROJECTIONS),
        "max_wall_seconds": MAX_WALL_SECONDS,
        "heldout_opened": False,
        "release_quality_claim": False,
        "automatic_switching_allowed": False,
    })

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
        torch.manual_seed(freeze["recipe"]["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        amp = bool(torch.cuda.is_bf16_supported())
        progress("loading_model")
        model = AutoModelForCausalLM.from_pretrained(
            freeze["model_path"], local_files_only=True,
            trust_remote_code=False, dtype=torch.float32).to("cuda")
        model.eval()
        progress("model_loaded")
        batches = [torch.tensor([sequence], device="cuda", dtype=torch.long)
                   for sequence in dev]
        baseline, reference = score(model, batches, None, amp, deadline)
        progress("baseline_scored", **baseline)
        rows = []
        for source_index, target_index in pairs:
            if time.monotonic() > deadline:
                raise TimeoutError("development screen wall budget exhausted")
            progress("pair_started", source_layer=source_index,
                     target_layer=target_index)
            source = model.model.layers[source_index].mlp
            target = model.model.layers[target_index].mlp
            originals = {name: getattr(target, name).weight.detach().clone()
                         for name in PROJECTIONS}
            try:
                with torch.no_grad():
                    for name in PROJECTIONS:
                        getattr(target, name).weight.copy_(
                            getattr(source, name).weight)
                metrics, _ = score(model, batches, reference, amp, deadline)
            finally:
                with torch.no_grad():
                    for name, original in originals.items():
                        getattr(target, name).weight.copy_(original)
                if any(not torch.equal(getattr(target, name).weight, original)
                       for name, original in originals.items()):
                    raise RuntimeError("model weights were not restored")
            row = {"source_layer": source_index, "target_layer": target_index,
                   **metrics,
                   "nll_change_percent": 100 * (metrics["nll"] / baseline["nll"] - 1),
                   "accuracy_change_points": 100 * (
                       metrics["next_token_accuracy"] -
                       baseline["next_token_accuracy"])}
            rows.append(row)
            with (root / "pair-results.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            progress("pair_completed", **row)
        report = {
            "schema_version": 1,
            "kind": "qwen3-adjacent-gate-up-copy-development-screen-result",
            "claim_sha256": digest(root / "claim.json"),
            "baseline": baseline,
            "pairs_by_nll": rank_rows(rows),
            "wall_seconds": time.monotonic() - started,
            "gpu": torch.cuda.get_device_name(0),
            "heldout_opened": False,
            "release_quality_claim": False,
            "automatic_switching_allowed": False,
            "limitations": [
                "Zero-step copied weights only; no shared-model training or quality approval.",
                "Ranking uses development data and cannot serve as a release held-out result.",
                "No FPGA implementation, timing, or switching costs were measured.",
            ],
        }
        write_new(root / "result.json", report)
        progress("completed", best_pair=report["pairs_by_nll"][0])
    except BaseException as error:
        progress("failed", error_type=type(error).__name__, error=str(error))
        write_new(root / "failure.json", {
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "heldout_opened": False,
        })
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

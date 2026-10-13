"""Bounded, matched Qwen3-0.6B training pilot on fresh real text.

Only train and development text are accepted. This tool never opens a release
held-out suite, downloads a model, or claims a deployable shared image.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback


RECIPE = {
    "model": "Qwen3-0.6B",
    "source_layer": 0,
    "target_layer": 1,
    "projections": ["gate_proj", "up_proj"],
    "sequence_length": 128,
    "train_sequences": 128,
    "dev_sequences": 32,
    "batch_size": 1,
    "steps": 40,
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "distillation_temperature": 2.0,
    "cross_entropy_fraction": 0.5,
    "seed": 20261013,
    "max_wall_minutes": 20,
    "dev_nll_relative_limit": 1.05,
    "dev_accuracy_drop_limit": 0.01,
}
SCREEN_SELECTED_PAIR = (12, 13)
SCREEN_SELECTED_RECIPE = {
    **RECIPE,
    "source_layer": SCREEN_SELECTED_PAIR[0],
    "target_layer": SCREEN_SELECTED_PAIR[1],
    "dev_teacher_nll_relative_limit": 1.05,
    "dev_teacher_agreement_min": 0.90,
}
IDENTITY_FILES = (
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "model.safetensors.index.json", "download-manifest.json",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recipe_for_pair(source_layer, target_layer):
    pair = (source_layer, target_layer)
    if pair == (RECIPE["source_layer"], RECIPE["target_layer"]):
        return copy.deepcopy(RECIPE)
    if pair == SCREEN_SELECTED_PAIR:
        return copy.deepcopy(SCREEN_SELECTED_RECIPE)
    raise ValueError("pair must be original 0:1 or screened 12:13")


def development_gate(metrics, recipe):
    teacher = metrics["teacher"]
    ordinary = metrics["ordinary"]
    shared = metrics["shared"]
    matched_control = (
        shared["nll"] <= recipe["dev_nll_relative_limit"] * ordinary["nll"]
        and shared["next_token_accuracy"] >=
        ordinary["next_token_accuracy"] - recipe["dev_accuracy_drop_limit"])
    teacher_anchor = (
        shared["nll"] <= recipe.get("dev_teacher_nll_relative_limit", float("inf"))
        * teacher["nll"]
        and shared["teacher_agreement"] >=
        recipe.get("dev_teacher_agreement_min", 0.0))
    return {"matched_control": matched_control,
            "teacher_anchor": teacher_anchor,
            "passed": matched_control and teacher_anchor}


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def documents(path, minimum_chars=64):
    seen, result = set(), []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            value = record if isinstance(record, str) else record.get("text")
            if not isinstance(value, str):
                raise ValueError(f"{path}:{line_number}: expected a text string")
            value = value.strip()
            if len(value) < minimum_chars:
                continue
            key = hashlib.sha256(value.encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                result.append((key, value))
    if not result:
        raise ValueError(f"{path}: no usable documents")
    return result


def sequences(tokenizer, docs, count, length):
    tokens = []
    for _, value in docs:
        tokens.extend(tokenizer.encode(value, add_special_tokens=False))
        tokens.append(tokenizer.eos_token_id)
        if len(tokens) >= count * (length + 1):
            break
    width = length + 1
    if len(tokens) < count * width:
        raise ValueError(f"only {len(tokens)} tokens; need {count * width}")
    result = [tokens[i * width:(i + 1) * width] for i in range(count)]
    serial = json.dumps(result, separators=(",", ":")).encode()
    return result, hashlib.sha256(serial).hexdigest()


def model_identity(path):
    path = Path(path)
    result = {}
    for name in IDENTITY_FILES:
        candidate = path / name
        if candidate.is_file():
            result[name] = digest(candidate)
    weights = sorted(path.glob("*.safetensors"))
    if not weights or "config.json" not in result or "tokenizer.json" not in result:
        raise ValueError("local model needs config, tokenizer.json and safetensors weights")
    for candidate in weights:
        result[candidate.name] = digest(candidate)
    return result


def load_inputs(freeze):
    from transformers import AutoTokenizer

    model_path = Path(freeze["model_path"])
    if model_identity(model_path) != freeze["model_files_sha256"]:
        raise ValueError("model files changed since freeze")
    for split in ("train", "dev"):
        if digest(freeze[f"{split}_path"]) != freeze[f"{split}_file_sha256"]:
            raise ValueError(f"{split} text changed since freeze")
    if freeze.get("corpus_manifest_path"):
        if digest(freeze["corpus_manifest_path"]) != freeze["corpus_manifest_sha256"]:
            raise ValueError("corpus provenance manifest changed since freeze")
    train_docs = documents(freeze["train_path"])
    dev_docs = documents(freeze["dev_path"])
    if {key for key, _ in train_docs} & {key for key, _ in dev_docs}:
        raise ValueError("train and development documents overlap")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    recipe = freeze["recipe"]
    train, train_sha = sequences(tokenizer, train_docs, recipe["train_sequences"],
                                 recipe["sequence_length"])
    dev, dev_sha = sequences(tokenizer, dev_docs, recipe["dev_sequences"],
                             recipe["sequence_length"])
    if (("train_token_sha256" in freeze and
         train_sha != freeze["train_token_sha256"]) or
            ("dev_token_sha256" in freeze and
             dev_sha != freeze["dev_token_sha256"])):
        raise ValueError("tokenized data changed since freeze")
    return train, dev


def freeze_inputs(args):
    import transformers
    if not hasattr(transformers, "Qwen3ForCausalLM"):
        raise RuntimeError("installed Transformers lacks Qwen3ForCausalLM")
    recipe = recipe_for_pair(args.source_layer, args.target_layer)
    model_path = Path(args.model).resolve(strict=True)
    config = transformers.AutoConfig.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False)
    if (config.model_type != "qwen3" or
            config.num_hidden_layers <= recipe["target_layer"]):
        raise ValueError("this frozen recipe requires a multi-layer Qwen3 model")
    train_path = Path(args.train_jsonl).resolve(strict=True)
    dev_path = Path(args.dev_jsonl).resolve(strict=True)
    manifest = {
        "schema_version": 1,
        "kind": "qwen3-pair-shared-real-text-development-pilot",
        "source_script_sha256": digest(__file__),
        "model_path": str(model_path),
        "model_files_sha256": model_identity(model_path),
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "train_file_sha256": digest(train_path),
        "dev_file_sha256": digest(dev_path),
        "recipe": recipe,
        "release_quality_claim": False,
        "automatic_switching_allowed": False,
        "heldout_opened": False,
        "compiler_selective_pair_layout_supported": False,
    }
    corpus_manifest = train_path.parent / "corpus-manifest.json"
    if train_path.parent == dev_path.parent and corpus_manifest.is_file():
        provenance = json.loads(corpus_manifest.read_text())
        files = provenance.get("files", {})
        if (files.get(train_path.name, {}).get("sha256") !=
                manifest["train_file_sha256"] or
                files.get(dev_path.name, {}).get("sha256") !=
                manifest["dev_file_sha256"]):
            raise ValueError("corpus manifest hashes do not match text inputs")
        manifest["corpus_manifest_path"] = str(corpus_manifest)
        manifest["corpus_manifest_sha256"] = digest(corpus_manifest)
    train, dev = load_inputs(manifest)
    for split, data in (("train", train), ("dev", dev)):
        manifest[f"{split}_token_sha256"] = hashlib.sha256(
            json.dumps(data, separators=(",", ":")).encode()).hexdigest()
    write_new(args.output, manifest)
    print(json.dumps({"freeze": str(Path(args.output).resolve()),
                      "train_sequences": len(train), "dev_sequences": len(dev)}))


def select_projections(model, recipe, tie):
    import torch

    source = model.model.layers[recipe["source_layer"]].mlp
    target = model.model.layers[recipe["target_layer"]].mlp
    with torch.no_grad():
        for name in recipe["projections"]:
            getattr(target, name).weight.copy_(getattr(source, name).weight)
    if tie:
        for name in recipe["projections"]:
            getattr(target, name).weight = getattr(source, name).weight
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in (source, target):
        for name in recipe["projections"]:
            getattr(layer, name).weight.requires_grad_(True)
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def score_models(teacher, ordinary, shared, data, recipe, device, amp, deadline):
    import torch
    import torch.nn.functional as F

    for model in (teacher, ordinary, shared):
        model.eval()
    sums = {name: {"nll_sum": 0.0, "correct": 0, "agree": 0,
                   "agree_teacher_correct": 0,
                   "agree_teacher_wrong": 0,
                   "correct_teacher_wrong": 0}
            for name in ("teacher", "ordinary", "shared")}
    total = teacher_correct = 0
    with torch.inference_mode():
        for start in range(0, len(data), recipe["batch_size"]):
            if time.monotonic() > deadline:
                raise TimeoutError("frozen wall-clock budget reached during scoring")
            batch = torch.tensor(data[start:start + recipe["batch_size"]],
                                 dtype=torch.long, device=device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = {name: model(input_ids=inputs, use_cache=False).logits
                          for name, model in (("teacher", teacher),
                                              ("ordinary", ordinary),
                                              ("shared", shared))}
            reference = logits["teacher"].argmax(-1)
            teacher_right = reference.eq(targets)
            teacher_correct += int(teacher_right.sum())
            total += targets.numel()
            for name, value in logits.items():
                predicted = value.argmax(-1)
                agrees = predicted.eq(reference)
                stats = sums[name]
                stats["nll_sum"] += F.cross_entropy(
                    value.float().reshape(-1, value.shape[-1]),
                    targets.reshape(-1), reduction="sum").item()
                stats["correct"] += int(predicted.eq(targets).sum())
                stats["agree"] += int(agrees.sum())
                stats["agree_teacher_correct"] += int((agrees & teacher_right).sum())
                stats["agree_teacher_wrong"] += int((agrees & ~teacher_right).sum())
                stats["correct_teacher_wrong"] += int(
                    (predicted.eq(targets) & ~teacher_right).sum())
    result = {}
    for name, stats in sums.items():
        result[name] = {
            "nll": stats["nll_sum"] / total,
            "next_token_accuracy": stats["correct"] / total,
            "teacher_agreement": stats["agree"] / total,
            "agreement_when_teacher_correct":
                stats["agree_teacher_correct"] / teacher_correct if teacher_correct else None,
            "agreement_when_teacher_wrong":
                stats["agree_teacher_wrong"] / (total - teacher_correct)
                if total > teacher_correct else None,
            "correct_when_teacher_wrong":
                stats["correct_teacher_wrong"] / (total - teacher_correct)
                if total > teacher_correct else None,
            "targets": total,
        }
    return result


def train_step(model, optimizer, inputs, targets, teacher_logits, recipe, amp):
    import torch
    import torch.nn.functional as F

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(input_ids=inputs, use_cache=False).logits
    student = logits.float()
    reference = teacher_logits.float()
    ce = F.cross_entropy(student.reshape(-1, student.shape[-1]),
                         targets.reshape(-1))
    temperature = recipe["distillation_temperature"]
    kd = F.kl_div(F.log_softmax(student / temperature, dim=-1),
                  F.softmax(reference / temperature, dim=-1),
                  reduction="sum") / targets.numel() * temperature ** 2
    fraction = recipe["cross_entropy_fraction"]
    loss = fraction * ce + (1 - fraction) * kd
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    return float(loss.detach())


def save_projection_delta(model, recipe, path):
    from safetensors.torch import save_file
    tensors = {}
    for layer_index in (recipe["source_layer"], recipe["target_layer"]):
        layer = model.model.layers[layer_index].mlp
        for name in recipe["projections"]:
            key = f"model.layers.{layer_index}.mlp.{name}.weight"
            tensors[key] = getattr(layer, name).weight.detach().cpu().contiguous().clone()
    save_file(tensors, str(path))
    return digest(path)


def run(args):
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA GPU unavailable")
    freeze_path = Path(args.freeze).resolve(strict=True)
    freeze = json.loads(freeze_path.read_text())
    if freeze["source_script_sha256"] != digest(__file__):
        raise ValueError("runner changed since freeze")
    if freeze["kind"] != "qwen3-pair-shared-real-text-development-pilot":
        raise ValueError("wrong freeze kind")
    recipe = freeze["recipe"]
    if recipe != recipe_for_pair(recipe["source_layer"],
                                 recipe["target_layer"]):
        raise ValueError("recipe changed since freeze")
    if "train_token_sha256" not in freeze or "dev_token_sha256" not in freeze:
        raise ValueError("incomplete frozen token identities")
    train, dev = load_inputs(freeze)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "claim.json", {
        "freeze_sha256": digest(freeze_path),
        "attempt_state": "claimed",
        "source_script_sha256": digest(__file__),
        "heldout_opened": False,
    })
    started = time.monotonic()
    deadline = started + recipe["max_wall_minutes"] * 60
    progress_path = out / "progress.jsonl"
    def progress(phase, **details):
        event = {"phase": phase, "elapsed_seconds": round(time.monotonic() - started, 3),
                 **details}
        with progress_path.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(event, allow_nan=False), flush=True)

    try:
        progress("claimed", steps_planned=recipe["steps"],
                 wall_budget_seconds=recipe["max_wall_minutes"] * 60)
        torch.manual_seed(recipe["seed"])
        random.seed(recipe["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda")
        amp = bool(torch.cuda.is_bf16_supported())
        model_path = freeze["model_path"]
        def load():
            return AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=False,
                dtype=torch.float32).to(device)
        progress("loading_teacher")
        teacher = load()
        progress("teacher_loaded")
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        progress("loading_ordinary")
        ordinary = load()
        progress("ordinary_loaded")
        ordinary_trainable = select_projections(ordinary, recipe, False)
        progress("loading_shared")
        shared = load()
        progress("shared_loaded")
        shared_trainable = select_projections(shared, recipe, True)
        ordinary_state, shared_state = ordinary.state_dict(), shared.state_dict()
        if ordinary_state.keys() != shared_state.keys() or any(
                not torch.equal(value, shared_state[key])
                for key, value in ordinary_state.items()):
            raise RuntimeError("matched student initial weights differ")
        source = shared.model.layers[recipe["source_layer"]].mlp
        target = shared.model.layers[recipe["target_layer"]].mlp
        if any(getattr(source, name).weight is not getattr(target, name).weight
               for name in recipe["projections"]):
            raise RuntimeError("shared parameters are not tied")
        optimizers = {
            "ordinary": torch.optim.AdamW(
                [p for p in ordinary.parameters() if p.requires_grad],
                lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"]),
            "shared": torch.optim.AdamW(
                [p for p in shared.parameters() if p.requires_grad],
                lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"]),
        }
        order = list(range(len(train)))
        random.Random(recipe["seed"]).shuffle(order)
        losses = []
        for step in range(recipe["steps"]):
            if time.monotonic() > deadline:
                raise TimeoutError("frozen wall-clock budget reached before scoring")
            progress("training_step_started", step=step + 1)
            indices = [order[(step * recipe["batch_size"] + j) % len(order)]
                       for j in range(recipe["batch_size"])]
            batch = torch.tensor([train[i] for i in indices],
                                 dtype=torch.long, device=device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=amp):
                reference = teacher(input_ids=inputs, use_cache=False).logits
            torch.manual_seed(recipe["seed"] + step)
            ordinary_loss = train_step(
                ordinary, optimizers["ordinary"], inputs, targets,
                reference, recipe, amp)
            torch.manual_seed(recipe["seed"] + step)
            shared_loss = train_step(
                shared, optimizers["shared"], inputs, targets,
                reference, recipe, amp)
            losses.append([ordinary_loss, shared_loss])
            progress("training_step_completed", step=step + 1,
                     ordinary_loss=ordinary_loss, shared_loss=shared_loss)
        if time.monotonic() > deadline:
            raise TimeoutError("frozen wall-clock budget reached before scoring")
        progress("scoring_started", sequences=len(dev))
        metrics = score_models(teacher, ordinary, shared, dev, recipe, device,
                               amp, deadline)
        progress("scoring_completed")
        ordinary_metrics, shared_metrics = metrics["ordinary"], metrics["shared"]
        gate_checks = development_gate(metrics, recipe)
        gate = gate_checks["passed"]
        ordinary_delta = out / "ordinary-projections.safetensors"
        shared_delta = out / "shared-projections.safetensors"
        ordinary_sha = save_projection_delta(ordinary, recipe, ordinary_delta)
        shared_sha = save_projection_delta(shared, recipe, shared_delta)
        torch.cuda.synchronize()
        report = {
            "schema_version": 1,
            "kind": "qwen3-pair-shared-real-text-development-result",
            "freeze_sha256": digest(freeze_path),
            "source_script_sha256": digest(__file__),
            "model_files_sha256": freeze["model_files_sha256"],
            "train_file_sha256": freeze["train_file_sha256"],
            "dev_file_sha256": freeze["dev_file_sha256"],
            "train_token_sha256": freeze["train_token_sha256"],
            "dev_token_sha256": freeze["dev_token_sha256"],
            "software": {"torch": torch.__version__,
                         "transformers": __import__("transformers").__version__,
                         "hip": torch.version.hip,
                         "gpu": torch.cuda.get_device_name(0),
                         "bf16_autocast": amp},
            "recipe": recipe,
            "trainable_parameters": {"ordinary": ordinary_trainable,
                                     "shared": shared_trainable},
            "paired_initial_weights_equal": True,
            "shared_parameter_identity_verified": True,
            "training_losses": losses,
            "development": metrics,
            "development_gate_checks": gate_checks,
            "passed_development_gate": gate,
            "projection_delta_sha256": {"ordinary": ordinary_sha,
                                        "shared": shared_sha},
            "wall_seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "release_quality_claim": False,
            "heldout_opened": False,
            "automatic_switching_allowed": False,
            "compiler_selective_pair_layout_supported": False,
        }
        write_new(out / "result.json", report)
        progress("completed", passed_development_gate=gate)
        print(json.dumps({"result": str(out / "result.json"),
                          "passed_development_gate": gate,
                          "ordinary_nll": ordinary_metrics["nll"],
                          "shared_nll": shared_metrics["nll"],
                          "ordinary_accuracy": ordinary_metrics["next_token_accuracy"],
                          "shared_accuracy": shared_metrics["next_token_accuracy"],
                          "wall_seconds": report["wall_seconds"]}), flush=True)
    except BaseException as error:
        progress("failed", error_type=type(error).__name__, error=str(error))
        write_new(out / "failure.json", {
            "state": "failed_or_budget_stopped",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "heldout_opened": False,
            "wall_seconds": time.monotonic() - started,
        })
        raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--model", required=True)
    freeze_parser.add_argument("--train-jsonl", required=True)
    freeze_parser.add_argument("--dev-jsonl", required=True)
    freeze_parser.add_argument("--output", required=True)
    freeze_parser.add_argument("--source-layer", type=int, default=0)
    freeze_parser.add_argument("--target-layer", type=int, default=1)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--freeze", required=True)
    run_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_inputs(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

"""Audit published fixed-tape RTL benchmarks and test predictor generalization.

This is an offline analysis: it never runs inference or consumes held-out quality
targets. Output is JSON on stdout so callers can retain an immutable report.
"""
import hashlib
import json
import statistics
from pathlib import Path

from malleable.llm.experiments import validate_benchmark_result
from malleable.llm.optimization import Predictor
from malleable.records import identity


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/evidence/benchmarks-20261003-14of20"
DELTA = ROOT / "docs/evidence/benchmarks-20261003-complete"
REPORTS = DELTA / "complete-performance-reports-20261003"
FILES = {
    "qwen3": (BASE / "benchmark-window-20261001-01/qwen3/manifest.json", REPORTS / "qwen3-performance.json"),
    "qwen35": (BASE / "benchmark-window-20261001-01/qwen35/manifest.json", REPORTS / "qwen35-performance.json"),
    "lfm": (BASE / "lfm2-performance-manifest.json", BASE / "lfm2-01/store/research/objects/8486833b35eeba0ef1ccef4098df5e07360b6ccdb594dcd7eedceae59eae8f52.json"),
}


def load_verified(name):
    manifest_path, report_path = FILES[name]
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    if name == "lfm":
        if identity(report) != report_path.stem:
            raise ValueError("LFM report content address mismatch")
    else:
        inventory = json.loads((REPORTS / "inventory.json").read_text())[name]
        if hashlib.sha256(report_path.read_bytes()).hexdigest() != inventory["sha256"]:
            raise ValueError(f"{name} report export checksum mismatch")
    if report.get("completed") != 10 or report.get("failed") != 0 or len(report.get("runs", [])) != 10:
        raise ValueError(f"{name} has an incomplete report")
    if report.get("manifest_id") != identity(manifest):
        raise ValueError(f"{name} manifest identity mismatch")
    for index, row in enumerate(report["runs"]):
        validate_benchmark_result(manifest, index, row)
    return report["runs"]


def summarize(rows, predictor):
    samples = []
    for row in rows:
        measured = sum(step["cycles"] for step in row["counters"])
        predicted = predictor.predict(row)["cycles"]
        samples.append({"index": row["benchmark_run_index"], "personality": row["personality"],
                        "scenario": row["memory_scenario"], "measured_cycles": measured,
                        "predicted_cycles": predicted,
                        "absolute_percentage_error": abs(predicted - measured) / measured * 100})
    return {"count": len(samples), "mean_absolute_percentage_error": statistics.mean(
        sample["absolute_percentage_error"] for sample in samples),
        "worst_absolute_percentage_error": max(sample["absolute_percentage_error"] for sample in samples),
        "samples": samples}


def main():
    rows = {name: load_verified(name) for name in FILES}
    approvals = {}
    for name, (manifest_path, _) in FILES.items():
        manifest = json.loads(manifest_path.read_text())
        approvals[name] = sorted({e["record"]["personality"] for e in
                                  manifest.get("quality_evidence", [])
                                  if e.get("record", {}).get("split") == "validation"})
    predictor = Predictor().fit(rows["qwen3"] + rows["qwen35"])
    report = {"schema_version": 1, "provenance": "offline-prediction-versus-published-rtl-cycles",
              "training_models": ["qwen3", "qwen35"], "held_out_model": "lfm",
              "training_rmse_cycles": predictor.rmse,
              "manifest_quality_approved_personalities": approvals,
              "by_model": {name: summarize(data, predictor) for name, data in rows.items()},
              "limitations": ["Fixed-tape runs have no generated-token timing or sequential switching evidence.",
                              "Four-personality quality approval is absent from the exported manifests.",
                              "Physical FPGA timing, power, and runtime changeover costs are unavailable.",
                              "No model-specific held-out policy promotion is performed."]}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

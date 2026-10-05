# Offline controller and predictor readiness audit

Run `PYTHONPATH=. python3 tools/evaluate_exported_predictor.py` from the repository root to regenerate `predictor-report.json`. The command reads only published benchmark artifacts. It checks the two exported report checksums, the LFM content address, each manifest identity, and all 30 indexed benchmark results using the repository's validator. It does not run model inference or consume held-out quality examples.

The current ridge predictor was fitted on the 20 Qwen3/Qwen3.5 rows and frozen before checking the ten LFM rows. Mean absolute percentage error was 25.58% on Qwen3, 19.21% on Qwen3.5, and **109.85% on LFM**; worst LFM error was 161.75%. These errors are on aggregate fixed-tape RTL cycles. The current predictor does not generalize adequately to the held-out model, so its estimates cannot support a claimed controller improvement or automatic promotion.

The exported performance manifests contain no personality-specific validation-quality approvals. Exact DRAM/TMEM checks in the 30 performance runs establish execution correctness for those tapes; they do not grant all four personalities model-quality approval. The published data also lack sequential request episodes and measured program/reload/re-prefill costs. Controller training or promotion through the repository's gated release path therefore remains unrun here. A simulation using invented zero switching costs would not close this gap.

Next, obtain the already recorded quality/model-store evidence from the local release machine, run the four-personality quality checks required by the controller contract if they have not been completed, and export verified episodes with explicit switching scenarios. Do not repeat consumed held-out evaluations or the 30 completed benchmark runs. Evaluate predictor alternatives on the same frozen Qwen3/Qwen3.5 to LFM split, then compare fixed, heuristic, predictor-only, budget-matched random, and five seeded Double DQN policies. Retain the deterministic controller if the non-regression gate fails.

No physical timing, power, or deployed FPGA behavior is inferred from this audit.

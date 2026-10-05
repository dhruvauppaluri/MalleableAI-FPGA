# ISA personality validation

Nine new context-128 INT8/head-INT8 ISA quality evaluations used the unchanged frozen validation suites and model checkpoints downloaded at PR #8's pinned revisions. Qwen3 used the checksum-verified calibrated candidate rebuilt by the handoff script. The 30 completed RTL benchmarks and three consumed held-out evaluations were not rerun.

Gate: next-token agreement at least 90% and NLL degradation no more than 5% versus the original floating-point model, over 1,024 targets per trial. Results are validation evidence only (`selectable: false`); held-out release approval and measured switching costs remain outstanding. Automatic program switching stays gated.

| Model | Personality | Agreement | NLL change | Gate | Matching benchmark indices |
| --- | --- | ---: | ---: | --- | --- |
| LFM2.5-230M | compact | 93.75% | +0.0565% | pass | 0, 4 |
| LFM2.5-230M | compute | 93.75% | +0.0565% | pass | 2, 6, 9 |
| LFM2.5-230M | buffered | 93.75% | +0.0565% | pass | 3, 7 |
| Qwen3-0.6B | compact | 94.04% | +0.0044% | pass | 0, 4 |
| Qwen3-0.6B | compute | 94.04% | +0.0044% | pass | 2, 6, 9 |
| Qwen3-0.6B | buffered | 94.04% | +0.0044% | pass | 3, 7 |
| Qwen3.5-0.8B | compact | 96.78% | -0.1507% | pass | 0, 4 |
| Qwen3.5-0.8B | compute | 96.78% | -0.1507% | pass | 2, 6, 9 |
| Qwen3.5-0.8B | buffered | 96.78% | -0.1507% | pass | 3, 7 |

Each raw record is stored alongside this report. `summary.json` lists its SHA-256 and exact configuration ID. The repository's `quality_matches` check accepted every record against all published benchmark rows for that model and personality.

Run command shape: `python -m malleable.llm.cli quality --model build/models/MODEL --suite build/cloud-quality/MODEL/quality-suite.json --split validation --context 128 --personality PERSONALITY --wformat int8 --max-host-gib 16 --store build/cloud-quality/MODEL/PERSONALITY/store`. For Qwen3, the command was `quality-candidate-validate` with `--candidate-case build/cloud-handoff/qwen3-candidate/case-00`. The same frozen suite file was reused across the three personalities of each model; floating references were cached after the first pass.

The cloud host used Python 3.12.14, PyTorch 2.11.0+cpu, Transformers 5.17.0, and a four-core CPU quota. The raw records include their own toolchain and floating-reference identities. No GPU or physical FPGA timing was measured.

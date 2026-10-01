# Standalone acceptance — 2026-09-30

**PASS: standalone schema v3**, all three required models. This is local full-RTL
simulation evidence, not physical FPGA deployment or the full platform release.

| Model | Validation agreement | Held-out agreement | Held-out NLL change | RTL output tokens | Checked steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 94.04% | 94.14% | -0.207% | 8 | 12 |
| Qwen3.5-0.8B | 96.78% | 95.51% | +0.068% | 8 | 15 |
| LFM2.5-230M | 93.75% | 95.41% | +0.338% | 8 | 16 |

Each validation and held-out split contains 1,024 targets. Quality and generation
use context 128, balanced personality, INT8 matrices and INT8 head. Qwen3 uses its
approved calibrated candidate; Qwen3.5 and LFM use raw INT8. Gates remain agreement
>=90% and NLL degradation <=5%. All model steps passed bit-exact DRAM/TMEM checks.
The synthetic run adds 100 checked tokens. Audit verified 143 original trace.txt
hashes and recorded profile hashes, canonical result IDs, frozen suite identity,
configuration, current checkpoint/tokenizer identity and persistent held-out claims.

## Reproduce the checker without inference

From /home/dhruv/projects/MalleableAI-FPGA:

```sh
.venv/bin/python -m malleable.llm.cli release-check --manifest build/zephyrus-jobs/release/20260928T223413Z-b463e746/standalone-v3-20260930-01/standalone-v3.json
```

Manifest SHA256: `23f51698475d75850bdfcd8222fe87c866b55e76986b7ec3b782b30b752a2118`.
The directory includes `release-check.json`, `summary.json`, `inventory.json`,
`source-compatibility.json`, `trace-audit.json`, byte-identical model records,
frozen suites and applicable candidate freezes. Original traces/profiles remain
at their recorded absolute locations in the release root. This JSON package is
not a portable copy of all trace files. Never remove original evidence folders.

`tools/assemble_standalone_release.py --output NEW_DIRECTORY` reproduces the audit
from these preserved inputs; it refuses to overwrite an existing package.
Package source: `d5207ad48f99c1b793346a115e073f53106aa2cf`. Recorded inference source identities
remain historical and unchanged. Execution code under third_party/opentpu, rtl,
runtime.py, backend.py and models.py is unchanged since the reviewed Qwen3 freeze
baseline 0df744eb8f7dd906c434b6f5340dcd39ee000e16. Qwen3/LFM quality arithmetic and
reference identity remain unchanged; safeguards became stricter. Qwen3.5 used
the repaired reference v2 on source 608fea7. See the compatibility record.

## Qwen3.5 completion

The earlier 78.125% score used incorrectly initialized non-persistent rotary
buffers in the FP32 comparison model. Source ba9a988 rebuilds these buffers from
the unchanged configuration and versions Qwen3.5 reference caches. It changes no
checkpoint, suite, acceptance threshold, ISA or RTL arithmetic. Historical failures
remain preserved. Corrected validation and held-out each ran once to completion.

`continuation-20260930/qwen35-full-acceptance-01` contains quality evidence and the
failed initial RTL dispatch (Verilator absent from the service PATH).
`continuation-20260930/qwen35-rtl-eight-02` contains the successful RTL-only retry:
eight output tokens, 15 checked steps, 3,647.32 seconds. Its result SHA256 is
`a0f14b6a2119d7e9479f308a7beeaf56d1ffcc86095f91f35c49fd470f77f02a`.
Keep /home/dhruv/.local/bin in future RTL service PATH values.

## Remaining release scope

Standalone schema v3 is complete. Full-release schema v2 remains unfinished:
remaining Qwen3/Qwen3.5 indexed benchmarks, predictor/controller reports with
disjoint LFM evaluation, current-source workbench/browser acceptance, official
CUDA and matched hybrid evidence, final-source verification, full-release-check,
and PR review/current-tip CI evidence. LFM's ten benchmarks remain reusable
(7,492 seconds, approximately 2.08 hours). No merging or new heavy job is authorized
by this evidence-packaging step. Choose the next run mode in chat.

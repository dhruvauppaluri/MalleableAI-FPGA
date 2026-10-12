# AMD Radeon shared-matrix diagnostic, attempt 01

The user ran `tools/run_amd_shared_matrix_diagnostic.py` once in Radeon Cloud and
pasted the complete JSON report from `/workspace/amd-shared-matrix-20261012-01.json`.
The exact checkout commit was not printed in that terminal record. This is a
synthetic model diagnostic, not a release quality result or FPGA measurement.
The seed set in the report is consumed; do not rerun this attempt or reuse its
validation data.

| Measure | Ordinary student | Shared student | Shared minus ordinary |
| --- | ---: | ---: | ---: |
| Validation NLL | 0.494456 | 0.497551 | +0.626% |
| Next-token accuracy | 86.4163% | 86.5171% | +0.1008 percentage points |
| Teacher agreement | 87.3488% | 88.2308% | +0.8821 percentage points |
| Host ISA image bytes | 428,544 | 365,056 | -14.8148% |

The shared ISA layout produced bit-exact outputs in the report. The frozen
diagnostic gate **failed** because shared teacher agreement was below its 90%
absolute threshold. Sharing did not lower teacher agreement in this comparison.
The teacher's own next-token accuracy on the generated task was 84.6270%; both
students had higher task accuracy. Because training combined target cross entropy
with teacher distillation, an absolute agreement score may penalize some correct
departures from the teacher. The report does not include a per-token disagreement
breakdown, so that explanation remains a hypothesis.

The earlier CPU diagnostic in `docs/evidence/model-hardware-codesign-20261008`
also missed the same 90% rule while its shared student exceeded its ordinary
student on teacher agreement. Its validation seed had been scored in an earlier
failed attempt and was already marked diagnostic only. The Radeon attempt used a
new seed set; neither result establishes pretrained-model quality or physical
FPGA timing. Keep the original gate outcomes unchanged.

Before another paid experiment, freeze a new evaluation on fresh, unconsumed
data that compares shared and ordinary students under the same training budget,
reports task quality alongside teacher agreement, and separates disagreements
where the teacher was correct from those where it was wrong. Treat any new rule
as prospective. Automatic program switching remains gated.

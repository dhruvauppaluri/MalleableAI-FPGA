# Screen-selected 12→13 real-text development attempt 04

The operator ran a fresh, exclusive paired Qwen3-0.6B attempt on the healthy
Radeon Cloud notebook after the completed adjacent-pair development screen
selected layers 12→13 by lowest zero-step NLL. The uploaded archive's freeze
SHA-256 matches the claim and result. The runner source, model, corpus and
token hashes match the frozen identities. The two projection hashes in
`projection-sha256.txt` match `result.json`; projection binaries remain
outside the repository. The run finished all 40 matched training steps and
32 development sequences (4,096 next-token targets) in 15.903 seconds.

| Model | Development NLL | Next-token accuracy | Teacher agreement |
| --- | ---: | ---: | ---: |
| Unchanged teacher | 3.724158 | 38.9404% | 100% |
| Ordinary trained control | 3.591713 | 39.1846% | 81.8115% |
| Shared trained student | 3.618630 | 38.9404% | 81.8604% |

The shared student's NLL was 0.749% above the ordinary control and its
accuracy was 0.244 percentage points lower, so the frozen matched-control
check passed. Shared NLL was 2.834% **below** the untouched teacher, meeting
the teacher NLL limit. Teacher next-token agreement was only 81.8604%, below
the predeclared 90% minimum, so the teacher anchor and combined development
gate failed. The ordinary control's teacher agreement was similarly low at
81.8115%. For the shared student, agreement was 92.3511% where the teacher
was correct and 75.1699% where it was wrong. These conditional numbers
clarify that a small aggregate NLL and task-accuracy difference does not
imply preservation of the teacher's top-token decisions.

This is a development result on the same corpus used to select pair 12→13.
It cannot be promoted to an independent quality result. Attempts 01–04 and
the zero-step screen are consumed; do not rerun them. Do not open a release
held-out suite for this failed candidate. A future prospective development
recipe could investigate an initialization or objective that preserves
teacher predictions while maintaining real-text NLL, under a fresh claim.
Selective-pair compiler/RTL evidence, F2 timing and switching costs remain
unmeasured. Automatic program switching remains gated.

# Zero-step adjacent-pair development screen 01

The operator ran `tools/screen_shared_matrix_pairs.py` on the fresh Radeon
Cloud notebook under a six-minute external cap. The source script SHA-256 in
`claim.json` matches the published script. The claim references the completed
attempt-03 freeze SHA-256
`5c6846e15fcae9c317b1a4eacc34bc31108d62e134eca3b0c067f50f3e8d611b`.
The uploaded archive's claim SHA-256 matches `result.json`. All 27 claimed
adjacent pairs have one ordered row in `pair-results.jsonl` and the same row
in `result.json`; the last progress event is `completed`. The scan took
33.844 seconds. No training or release held-out test occurred.

The unchanged model scored 3.724158 NLL and 38.9404% next-token accuracy
on the 32 frozen WikiText development sequences (4,096 targets), exactly
matching attempt 03's teacher baseline. Copying layer 0's gate/up weights
into layer 1 **before training** raised NLL to 13.227332 (+255.18%) and cut
accuracy to 2.1973% (-36.743 points). The prior ordinary and shared students
recovered partially during 40 training steps, but both remained far below the
unchanged teacher. This directly confirms that the initial 0→1 copy is a
major source of their quality loss.

The lowest-NLL pair was 12→13: 3.824099 NLL (+2.684%), 37.2070% accuracy
(-1.733 points), and 82.3975% agreement with the unchanged model. Pair
11→12 had the highest accuracy at 37.5000% (-1.440 points) and NLL 3.833239
(+2.929%). No adjacent pair stayed within both 5% NLL and one percentage
point of the unchanged model in this zero-step development screen. These are
descriptive comparisons, not a retroactive quality gate. The best pair is
selected on this same development set, so it cannot serve as an independent
quality claim.

The next training recipe should replace the damaging 0→1 starting point,
predeclare the selected pair and adaptation budget, and use a fresh exclusive
attempt. Its development score must be compared with an untouched pretrained
baseline as well as a matched ordinary control. Only a later frozen, new
independent held-out suite can support release quality review. The selective
pair compiler layout, FPGA timing and switching costs remain unmeasured;
automatic switching remains gated.

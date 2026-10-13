# Qwen3 adjacent-pair sharing development attempt 03

The operator completed this fresh attempt on a new Radeon Cloud notebook after
the default ROCm transfer and Qwen3 load/forward probes passed. The earlier
attempts 01 and 02 remain consumed; this attempt used a distinct freeze and
claim directory. The uploaded evidence archive was extracted without path
changes. The freeze SHA-256 is
`5c6846e15fcae9c317b1a4eacc34bc31108d62e134eca3b0c067f50f3e8d611b`;
it matches both `claim.json` and `result.json`. The result's source script,
model, corpus and token hashes match the frozen identities. Projection file
digests in `projection-sha256.txt` match `result.json`; the binary projection
files remain outside the repository.

The run trained the ordinary and layer-0/1 gate/up-tied students for all 40
matched steps, then scored 32 WikiText development sequences (4,096 next-token
targets). It completed in 16.393 seconds with peak PyTorch GPU allocation of
9,554,226,688 bytes. The installed stack was PyTorch 2.9.1+gitff65f5b,
HIP 7.2.53211-e1a6bc5663, Transformers 4.57.6, with BF16 autocast.

| Model | Development NLL | Next-token accuracy | Teacher agreement |
| --- | ---: | ---: | ---: |
| Unmodified teacher | 3.724158 | 38.9404% | 100% |
| Ordinary control after layer-1 copy and training | 6.732002 | 15.4785% | 25.7080% |
| Shared student after layer-1 copy and training | 7.375570 | 12.2559% | 21.3135% |

The shared student's NLL is 9.56% higher than the ordinary control, exceeding
the frozen 5% limit; accuracy is 3.223 percentage points lower, exceeding the
1-point limit. The frozen development gate failed. Both students also scored
far below the untouched teacher: the ordinary student's NLL was 80.77% higher
and its accuracy 23.462 points lower. The recipe copied pretrained layer 0
gate/up weights into layer 1 in **both** students before training. This large
common regression indicates that the initial copy is a plausible major
problem, but no immediate post-copy, pretraining score was recorded; the
experiment does not isolate the copy's exact effect. The shared student had
half as many trainable parameters (6,291,456 versus 12,582,912) and its
training loss was usually higher; the results do not establish whether more
steps, a different layer pair, or a different adaptation method would recover
quality.

Next development work should measure a zero-step ordinary/copy baseline and
screen candidate layer pairs on development data before another tied training
attempt. Preserve this result and its failed gate. Do not use it as a release
quality pass, FPGA benefit, or reason to enable automatic switching. No
release held-out suite or RTL benchmark was run.

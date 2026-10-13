# Fresh Radeon notebook transfer and Qwen3 load checks

The operator ran these checks on a new Radeon Cloud notebook with container
hostname `j-thi5yonmij7y2mykjr5qgbhl3k`. This is a different instance from the
failed transfer notebook documented in `../amd-qwen-load-20261012-probe01/`.

With `HSA_ENABLE_SDMA` unset, `tools/probe_amd_transfer.py` (SHA-256
`00c330bb2251d00fe3a0ff69097cf1a4b683da71c6b7688cb0d8dfae360c7704`)
completed a four-byte CPU-to-GPU copy at 1.588 seconds and a 64 MiB copy at
1.604 seconds. The bounded command exited 0. The 64 MiB copy itself took about
0.010 seconds after its start event. The operator reported a pinned
Qwen3-0.6B `model.safetensors` SHA-256 of
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

The operator then ran the inline Python check in `qwen-load-check.txt` under a
two-minute external timeout. CPU model load completed at 1.47 seconds, full
FP32 model transfer to GPU at 1.70 seconds, and a four-token forward pass at
2.72 seconds. The logits shape was `(1, 4, 151936)` and exit status was 0.

This establishes basic transfer and Qwen3 inference health on this notebook.
It does not establish paired training throughput, shared-matrix quality, a
release held-out result, FPGA timing, or automatic switching approval. No
quality or held-out suite was opened by these checks.

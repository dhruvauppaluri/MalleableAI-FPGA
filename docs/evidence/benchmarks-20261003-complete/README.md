# Complete benchmark evidence

All ten Qwen3 and ten Qwen3.5 indexed runs verify. Together with the ten preserved LFM runs, all 30 performance runs are complete. This delta extends ../benchmarks-20261003-14of20; unchanged records remain there. Complete canonical reports are in complete-performance-reports-20261003/. inventory.json binds copied records and lossless compressed profiles to original SHA256 hashes. Raw trace text and model checkpoints remain local. No quality evaluation was repeated.

Baseline compact-to-compute cycle reduction is approximately 1% for Qwen3 and 3% for Qwen3.5. Under the combined latency/stall/bandwidth stress scenario, balanced cycle counts rise about 86% and 81%, respectively; compute improves over balanced by less than 0.1%. This supports memory-delivery sensitivity, not an isolated bandwidth diagnosis. These are fixed-tape RTL measurements, not physical FPGA speed or power.

Standalone quality acceptance remains passed. Full-release acceptance still requires controller quality/learning, browser, matched CUDA/hybrid, final-source verification and review.

# Benchmark evidence snapshot

14 of 20 Qwen3/Qwen3.5 indexed runs verified (indices 0-6 each), plus the preserved ten LFM runs and four completed four-bit validation screens. This is an incomplete performance release, not a full-suite acceptance claim.

Structured JSON/JSONL records are copied byte-for-byte; per-step profiles are losslessly gzip-compressed. inventory.json lists original and exported SHA256 hashes. Absolute paths identify the original local artifacts; model checkpoints and raw simulator trace text are not included. Trace hashes in result records identify the original local traces. This export supports data analysis but is not a self-contained hardware replay package. Historical interrupted attempts are preserved and must not be counted as successful runs.

Source of measured runs: e919d558960162d0caae12f50946e2aa20025274. Cycles and traffic are simulated; host timing is simulator execution time. Physical timing and power are unavailable. Quality-screen failures do not invalidate prior INT8 standalone acceptance.

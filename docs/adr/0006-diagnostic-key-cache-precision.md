# ADR-0006: bounded key-cache precision investigation

Status: diagnostic experiment only; production arithmetic/storage is not adopted.

The frozen 128-target panel reaches 91.41% agreement when key-cache quantization
is bypassed, versus 83.59% for baseline INT8. The user authorized an INT8 pilot
followed, on failure, by ten different precision cases. Both quality thresholds
remain unchanged. No held-out data is used.

Diagnostic policy v2 adds explicit overrides at key-store and query-quantization
sites. Float32/float16 cases round values to that storage type then compute in
the existing independent float64 emulation. INT16 uses signed symmetric max-abs
scaling to [-32767,32767], ties-to-even rounding and declared 128/64 blocks;
INT8 cases retain [-127,127] with 64/32/16 blocks at key-store only. These are
fully declared emulation experiments, not FPGA memory layouts or cycle estimates.
Other quantization points, weights, output head and overall D=128 are retained.

The fixed ten-case schedule is:

1. Key float32.
2. Key float16.
3. Key INT16, block 128.
4. Key INT8, block 64.
5. Key INT8, block 32.
6. Key INT8, block 16.
7. Key and query float32.
8. Key and query float16.
9. Key and query INT16, block 128.
10. Key and query INT16, block 64.

All cases use the existing frozen 128-target spread panel and original FP32
reference. Selection requires agreement >=90% and NLL degradation <=5%; record
failures as well as passes. Candidates remain unselectable release evidence.
Production adoption requires a further design decision specifying compiler,
ISA, DRAM/TMEM/cache layout, arithmetic, partial blocks, and matching RTL tests.
The dense INT8/INT32 numeric boundary is unchanged.

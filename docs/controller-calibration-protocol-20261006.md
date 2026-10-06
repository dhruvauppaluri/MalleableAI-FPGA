# Calibration protocol for the independent controller study

This document defines a small preparation tape for the October controller study.
It is written before candidate scoring and is not an accuracy claim. Its purpose
is to give each local tokenizer a fixed sample from which the quality machinery
can establish reference statistics without observing either of the later tapes.

The preparation stage follows four rules. First, the checkpoint and tokenizer
must match the immutable download inventory. Second, the program shape, numeric
format, and context limit must be named before execution. Third, every derived
artifact must be content addressed and traceable to its unchanged parent.
Fourth, the scoring process must stop when any recorded identity changes.

Calibration cannot approve a hardware personality. It cannot select a program,
alter a threshold, or justify a deployment. It is deliberately separated from
validation and final review. Measurements produced here may initialize cached
references, but they may not expose target outcomes from either later partition.

The operator should retain the command, source revision, start time, model file
inventory, and output digest. Missing metadata is a failure, not permission to
infer a value. If execution ends unexpectedly, its files remain historical and
a new uniquely named attempt must be used. This preparation tape is frozen once,
hashed once, and read only by the campaign tools after publication.

## Accumulator -> scratchpad move-out (mvout_spad)

This hardware supports `mvout_spad`: `st_acc_i8_act_spad` /
`st_acc_i8_act_spad_v2` (and the config-free `do_st_acc_i8_act_spad`) in
`gemmini` move accumulator rows out with the same scale / clamp / ReLU as a DRAM
mvout (`st_acc_i8_act`), but write the i8 rows into the scratchpad, in one
`gemmini_extended_mvout_spad`. The next layer can then read its input straight
from the scratchpad, with no DRAM round trip between the layers. You may use
it wherever it lowers cycles; it is optional, and a schedule without it is
just as valid.

Constraints:
- One mvout_spad writes `n <= 16` contiguous destination rows (stride 1): the
  reservation station tracks its destination as `n` consecutive rows, so a
  wider stride would hide the later rows from its dependency checks.
- It writes whole 16-column rows (the hardware ignores the column mask).
- Exo treats two fixed-address buffers as unrelated, so the consumer's
  scratchpad input and accumulator must not overlap the producer's.
- No fence is needed between a producer and a consumer connected this way.

A derivation that works without unsafe options (conv2 -> conv3): lower both
layers to low-level instrs, leave the producer's store and the consumer's input
staging as plain loops, inline both layers into one proc, cut the store loop
into pieces and duplicate each (`add_loop` then `unroll_loop` on the idempotent
store) as many times as the staging reads it, `fuse` each copy with the staging
piece that reads it, `inline_assign` the store, `delete_buffer` the
intermediate once nothing uses it, then `join_loops` and `replace` with the
instr.

Measured: that conv2 -> conv3 fusion cost about 350 cycles (48,464 against
48,106 for the same schedule without it), because conv3 stages three
kcol-shifted copies of its input (42 five-row mvout_spads instead of 14 mvouts
and 21 mvins). It pays off when the consumer reads one contiguous copy of the
producer's output, so look for producer/consumer pairs like that rather than
repeating this one.

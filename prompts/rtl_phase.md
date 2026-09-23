## This step: a microarchitecture (RTL) change

In this step, change how Gemmini executes commands by editing its RTL -- the
Chisel sources under `generators/gemmini/src/main/scala/gemmini/` (e.g.
`ExecuteController.scala`, `Scratchpad.scala`, `LoopConv.scala`,
`LoopMatmul.scala`). Do NOT tune `Configs.scala` parameters here; a later step
does that. Make one focused change per step.

What is known about where the time goes (hardware counters from a native-C
BraggNN run on this Gemmini; the Exo schedules the loop runs may differ):
- About half of all cycles, and 70-80% of every conv / NLB layer, the mesh is
  starved: it is ready for an operand that the scratchpad has not delivered
  yet (`SCRATCHPAD_A/B_WAIT_CYCLE` in `ExecuteController.scala`). Making the
  SRAMs dual-ported and adding accumulator banks removed only ~2,000 of
  ~51,000 cycles, so port or bank conflicts are not the main cause. Likely
  causes are the scratchpad read latency not being hidden across the many
  short commands BraggNN issues (its layers are 11, 9, 7 and 5 wide, so each
  compute moves only a few rows), and preload / config overhead per command.
- About 13% is the softmax in `LoopMatmul`: its multi-pass stats-then-output
  protocol, not a stall.
- The theoretical floor for this network on a 16x16 array is about 11,500
  cycles; the best schedule today runs at about 40,000.

How your change is judged: the loop builds a Verilator simulator of your
design (instead of Hammer) and runs fixed, known-good schedules (the seed, the
best one found so far, a fusion reference) on the first few patches. The
change is accepted only if the ISA files are unchanged, every schedule's
predictions are bit-identical to the seed's, and the mean cycles per patch go
down. A rejected change is reverted automatically before your next step, and
you get the reason. Verilator cycles are only comparable with each other, not
with FireSim numbers.

Your goal is to reduce single-batch BraggNN inference latency to 1,500 clock cycles measured cycle-accurate simulators. Current `braggnn.c` takes about 56,000 clock cycles, so you need to make 38x performance improvement without overfitting to the BraggNN workload.

You have direct shell access to the chipyard checkout via your bash tool. Use it now to actually read and edit the files yourself (e.g. `sed`/`python3 -c` in-place, or writing full file contents) until you are done. Do not just describe a plan, print a diff, or hand back a script for someone else to run -- if you don't call the tool to edit the files, no change happens at all.

Your session has a time limit, and a session that ends without an edit is wasted. Nothing can be built in this shell: sbt, make and the simulators do not run here (everything outside `generators/gemmini` is read-only and there is no build cache), so do not try to compile or elaborate. The loop elaborates your design with Hammer right after you reply and sends any error back to you. Read only what you need (the File Map below tells you where things are), make your edit early, and finish with a short explanation of what you changed. `generators/gemmini.bak` is an unrelated old fork; ignore it.

Shell tool rules: every command is killed after 300 seconds and its output is lost, so keep commands small. Never search recursively over `vlsi/`, `sims/`, `.conda-env/` or the whole checkout -- they hold gigabytes of build output and the search will time out; search `generators/gemmini/src` instead. A command that timed out does not mean the shell is stuck: just run your next command. Never kill processes (`kill`, `pkill`, `killall`): nothing you start outlives its command, and killing processes can only break the tool you are using.

## Target Application Spec

Your target application is BraggNN, a light-weight CNN model for Bragg peak localization.
It has three conv, non-local attention block (NLB), and four fc layers. The input size is arbitrary but you can assume 11x11 as default, and the output is x-y dimention's pixel with sub-pixel precision.

## Restriction

Don't change RoCC interface and evaluation infrastructer such as FireSim, Verilator, and Chipyard.

## Co-Design Loop

You are the hardware half of a hardware/software co-design loop, not the
whole loop. Every time you finish a change, it is built and, separately, an
automated search (Exo + AlphaEvolve) re-tunes the BraggNN kernel's
scheduling -- tiling, loop order, which ops go through Gemmini vs the host
CPU, etc. -- against your new hardware for several iterations, entirely on
its own, before you are asked to make another change. You will not see or
control that search directly; you only see its outcome (best cycle count
and accuracy) fed back to you afterward, plus a Hammer PPA (Genus synthesis,
sky130 + SRAM macros) area/slack/power summary for the same hardware when synthesis
succeeds.

Given that division of labor:
- Spend your effort on changes that only a hardware edit can produce --
  `Configs.scala` parameters (mesh size, dataflow, scratchpad/accumulator
  capacity, reservation-station depth, DMA width, datatypes). The
  automated SW search already covers scheduling/tiling decisions within
  whatever hardware you give it, so re-deriving a tiling strategy by hand
  in `gemmini.h`/`gemmini_nn.h` is redundant work that the next SW search
  pass will redo anyway.
- Read the feedback you're given (cycle count, accuracy, or a build/run
  failure) as the result of your *previous* hardware paired with the best
  schedule the SW search could find for it -- not as a report on a fixed
  kernel. A high cycle count means the SW search could not find a good
  enough schedule for that hardware, which is itself a hint about what to
  change (e.g. too few scratchpad banks to keep the mesh fed, an
  under-provisioned reservation station serializing independent ops).
- The C API's internal implementation (tiling/blocking logic in
  `gemmini.h`/`gemmini_nn.h`) is out of scope for you under normal
  circumstances; edit it only if a hardware change you made requires a
  matching C-side change to stay correct or to compile at all -- not as a
  way to hand-tune performance yourself.

## File Map

All paths below are relative to `generators/gemmini/` (the only directory
you may write to). Use this instead of blindly grepping the whole tree.

`src/main/scala/gemmini/` (hardware, Chisel):
- `Configs.scala` -- `GemminiArrayConfig` case class and the named configs
  (`defaultConfig`, etc.) that set all the hardware parameters in the
  Tuning Strategy section below. This is your main entry point for
  parameter changes.
- `Mesh.scala`, `MeshWithDelays.scala`, `Tile.scala`, `PE.scala` --
  the systolic-array datapath: a `Tile` is a block of `PE`s (single MAC
  units), `Mesh` arranges `Tile`s into the full `meshRows` x `meshColumns`
  grid, `MeshWithDelays` adds the pipeline delay/skew logic around it.
- `ExecuteController.scala` -- drives the mesh: sequences load-weight /
  compute / drain phases for `MVIN`/`MVOUT`/`COMPUTE`/`PRELOAD` instructions.
- `LoadController.scala`, `StoreController.scala` -- handle `MVIN`/`MVOUT`
  DMA instructions between scratchpad/accumulator and main memory.
- `Controller.scala` -- top-level RoCC command decode/dispatch into the
  load/store/execute controllers.
- `ReservationStation.scala` -- out-of-order instruction issue/dependency
  tracking across the ld/st/ex pipelines (`reservation_station_entries_*`,
  `*_queue_length` params live here).
- `Scratchpad.scala`, `AccumulatorMem.scala` -- the on-chip SRAM scratchpad
  and accumulator banks (`sp_capacity`/`sp_banks`, `acc_capacity`/
  `acc_banks`).
- `DMA.scala`, `BeatMerger.scala`, `XactTracker.scala` -- memory-side DMA
  engine feeding the scratchpad/accumulator (`dma_maxbytes`,
  `dma_buswidth`, `max_in_flight_mem_reqs`).
- `LoopMatmul.scala`, `LoopConv.scala`, `LoopUnroller.scala` -- hardware
  sequencers that implement `LOOP_WS`/`LOOP_CONV_WS` RoCC instructions
  (the hardware-side counterpart of `tiled_matmul_auto`/`tiled_conv_auto`'s
  looping in software).
- `Im2Col.scala` -- im2col unit used by the conv loop for weight-stationary
  convolution.
- `Normalizer.scala`, `AccumulatorScale.scala`, `Activation.scala` --
  post-matmul scaling/normalization/activation pipeline.
- `Dataflow.scala` -- the `OS`/`WS`/`BOTH` dataflow enum used by
  `dataflow` in `Configs.scala`.
- `LocalAddr.scala`, `GemminiISA.scala` -- scratchpad/accumulator address
  encoding and the RoCC instruction encoding/opcodes; rarely need editing.

`software/gemmini-rocc-tests/include/` (C API, called from `braggnn.c`):
- `gemmini.h` -- low-level C API: `tiled_matmul_auto`, raw
  `MVIN`/`MVOUT`/`COMPUTE` wrappers, and the tiling/blocking logic that
  decides how a matmul is split across scratchpad-sized tiles.
- `gemmini_nn.h` -- higher-level NN ops built on `gemmini.h`
  (`tiled_conv_auto` and similar), used directly by `braggnn.c`'s conv
  layers.
- `gemmini_params.h` -- `#define`s mirroring `Configs.scala`'s dimension
  parameters (`DIM`, `BANK_NUM`, `BANK_ROWS`, `ACC_ROWS`, `MAX_BYTES`,
  etc.). Generated by Gemmini when the design is elaborated, and the SW
  search compiles against that generated copy -- do not edit it by hand.

## Tuning Strategy

Do not change the Gemmini C API's interface (the function signatures
declared in `gemmini.h` / `gemmini_nn.h`, e.g. `tiled_matmul_auto`,
`tiled_conv_auto`, and friends). `braggnn.c`'s `gemmini_inference()` and the
rest of the pipeline depend on that interface staying fixed -- changing a
signature will break compilation or silently break correctness elsewhere in
the pipeline, which you cannot see or fix from this step.

Your primary lever is **Gemmini hardware parameters** -- edit
`generators/gemmini/src/main/scala/gemmini/Configs.scala`'s
`GemminiArrayConfig` (e.g. `defaultConfig`, or add a new config and wire it
into the build). Parameters worth tuning include:
- `meshRows` / `meshColumns` -- systolic array (PE grid) size (product with tile size fixed at 16, see below)
- `tileRows` / `tileColumns` -- PE tiling within the array
- `dataflow` -- `Dataflow.OS` / `WS` / `BOTH`
- `sp_capacity` / `acc_capacity` -- scratchpad / accumulator size
- `sp_banks` / `acc_banks` -- scratchpad / accumulator bank count
- `reservation_station_entries_ld/st/ex`, `ld_queue_length`,
  `st_queue_length`, `ex_queue_length` -- instruction-level parallelism
- `dma_maxbytes`, `dma_buswidth`, `max_in_flight_mem_reqs` -- DMA
  bandwidth/concurrency
- `inputType` / `weightType` / `accType` -- datatypes/precision (fixed by the SW toolchain, see below)

`gemmini_params.h` is regenerated from your `Configs.scala` every time the
design is elaborated (right after your edit), and every SW candidate is
compiled against that regenerated header, so the C side follows your
hardware automatically. Any hand edit to it is overwritten.

**Hard constraint from the SW toolchain:** the Exo instruction library the
SW search uses is written for a 16-wide array with int8 inputs/weights and
int32 accumulators. Keep `meshRows * tileRows == meshColumns * tileColumns
== 16` (i.e. `DIM` 16) and `inputType`/`weightType` = `SInt(8.W)`,
`accType` = `SInt(32.W)`. A design outside this is rejected right after
elaboration -- no bitstream is built and no cycles are measured -- and you
get that as failure feedback. Scratchpad/accumulator capacity and banking,
dataflow, queue depths and DMA parameters are all fair game. Note that
`acc_singleported = true` does not elaborate in this Gemmini as is:
`AccumulatorMem.scala` calls `io.ext_mem.get` in that branch, and
`ext_mem` is `None` unless `use_shared_ext_mem` is set. Fix that call as
part of the same edit if you want a single-ported accumulator.

**Hard constraint from the FPGA build:** every design is built into a FireSim
bitstream for one Xilinx Alveo U250, and a build that is not finished after
8 hours is killed and counts as a failure (the unedited design takes about
3.6 hours). Growth in the generated RTL costs far more than proportionally:
one attempt that switched the scratchpad to 16 dual-ported banks
(`sp_singleported = false`, `sp_banks = 16`), doubled the accumulator and set
`num_scale_units = -1` doubled the FPGA RTL (27 MB -> 54 MB), and Vivado's
RTL elaboration alone went from 2 minutes to over 2 hours, so the build never
finished. Grow the design in small steps, change a few parameters at a time,
and prefer changes that do not multiply memory ports or replicate large
datapaths.

The loop measures the generated FPGA RTL about 10 minutes into the build
and cancels any design over **50 MB** (it is then reverted to the last design
that built). Measured so far: 27 MB (unedited) built in 3.6 h, 45.2 MB in
4.5 h, and 53-54 MB designs never got through Vivado. The designs you start
from are already close to the limit, so budget the RTL: reservation-station
entries and queue lengths grow it steeply (doubling only the store
reservation station and queue added about 8 MB), and so do extra accumulator
or scratchpad banks, sub-banks, dual ports, a different tile shape and more
scale units. To afford a change, remove hardware the software does not use
(next section).

## What the software uses

The Exo library the SW search compiles against (`gemmini.py`) uses only
these Gemmini features. Keep all of them; removing or breaking any one makes
every candidate fail to build or compute wrong results:
- the weight-stationary dataflow (every `config_ex` is `WEIGHT_STATIONARY`);
- the matmul loop unroller `gemmini_loop_ws` (fc layers, the NLB matmuls,
  softmax and the residual add) and the conv loop unroller
  `gemmini_loop_conv_ws` (conv1-3 and the NLB 1x1 convs; `has_loop_conv`);
- first-layer optimizations (`has_first_layer_optimizations`): conv1 has one
  input channel and uses `max_pixels_per_row` = 3;
- normalization and softmax (`has_normalizations`, the non-linear activation
  path: `gemmini_config_norm` with the `SOFTMAX` activation, in the NLB);
- RELU on store;
- transposing B in a matmul (the NLB's theta x phi^T);
- mvin scaling (the residual add loads scaled inputs) and accumulator output
  scaling on every store (`acc_scale_args`);
- moving data into the accumulator (bias loads) and `fence`.

The SW search may also write its own Gemmini instrs, and a hand-tuned
schedule (`exo/braggnn_schedule_fast.py`, 34k cycles vs 45k) already uses
these, so keep them working too:
- `gemmini_loop_ws` with a bias D row of stride 0 and with A = NULL, which
  reuses the input the previous loop left in the scratchpad (LoopMatmul
  skips its A loads when the DRAM address is 0);
- `gemmini_loop_ws`'s skip bits (rs2 bits 3..7: LoopMatmul's
  `ld{a,b}_started` handling) and a resadd loop with K = 0 whose store goes
  through the SOFTMAX activation.

It never uses these, so they can be removed to save area and RTL:
- the output-stationary dataflow (`dataflow` can be `Dataflow.WS` instead of
  `BOTH`);
- training convolutions (`has_training_convs`: all transpose/rotation
  arguments of `gemmini_loop_conv_ws` are 0);
- max pooling (`has_max_pool`: pool size and stride are 1 and no store uses
  pooling);
- depthwise convolutions (`has_dw_convs`).

Leave `gemmini.h` / `gemmini_nn.h`'s internal tiling/blocking/scheduling
logic alone otherwise, per the Co-Design Loop section above -- that is the
automated SW search's job, re-run fresh after every hardware change you
make.

## Current Results
Here is current evaluation results including hardware counters, accuracy, and cycle counts, in case using initial braggnn.c and default Gemmini configuration.

```
==============================================
BraggNN Gemmini through native C
==============================================
Flushing Gemmini TLB

Input shape: [1, 1, 11, 11]
Running 11 inferences (1 warmup + 10 measured)

--- Warmup (patch 0) ---
Warmup done
cycles: 56350
dram<->spad DMA cycles: rdma=10540, wdma=4671
compute stall waiting on spad: A=30954, B=6126, D=9381
compute stall waiting on acc:  A=6103, B=30980, D=21614
pred: (4.6772, 5.1102), actual: (5.4543, 5.0766), error: (-0.7772, 0.0336)

--- Inference 1/10 ---
cycles: 51366
dram<->spad DMA cycles: rdma=9046, wdma=3974
compute stall waiting on spad: A=32055, B=0, D=13306
compute stall waiting on acc:  A=0, B=32020, D=18731
pred: (5.0236, 6.0630), actual: (5.3680, 5.4268), error: (-0.3444, 0.6362)

--- Inference 2/10 ---
cycles: 51332
dram<->spad DMA cycles: rdma=9028, wdma=3975
compute stall waiting on spad: A=32022, B=0, D=13318
compute stall waiting on acc:  A=0, B=32035, D=18733
pred: (5.8898, 5.1969), actual: (5.6984, 5.1974), error: (0.1914, -0.0005)

--- Inference 3/10 ---
cycles: 51322
dram<->spad DMA cycles: rdma=9007, wdma=3962
compute stall waiting on spad: A=32041, B=0, D=13352
compute stall waiting on acc:  A=0, B=32038, D=18702
pred: (6.2362, 5.8898), actual: (5.7705, 5.9003), error: (0.4657, -0.0105)

--- Inference 4/10 ---
cycles: 51274
dram<->spad DMA cycles: rdma=9012, wdma=3974
compute stall waiting on spad: A=32058, B=0, D=13324
compute stall waiting on acc:  A=0, B=32022, D=18715
pred: (5.8898, 6.0630), actual: (5.8680, 5.7095), error: (0.0217, 0.3535)

--- Inference 5/10 ---
cycles: 51253
dram<->spad DMA cycles: rdma=8997, wdma=3974
compute stall waiting on spad: A=32012, B=0, D=13304
compute stall waiting on acc:  A=0, B=31981, D=18694
pred: (6.2362, 5.7165), actual: (5.9607, 5.9356), error: (0.2755, -0.2191)

--- Inference 6/10 ---
cycles: 51193
dram<->spad DMA cycles: rdma=8960, wdma=3963
compute stall waiting on spad: A=31896, B=0, D=13195
compute stall waiting on acc:  A=0, B=31882, D=18703
pred: (4.6772, 5.1102), actual: (5.4543, 5.0766), error: (-0.7772, 0.0336)

--- Inference 7/10 ---
cycles: 51269
dram<->spad DMA cycles: rdma=8988, wdma=3979
compute stall waiting on spad: A=32001, B=0, D=13260
compute stall waiting on acc:  A=0, B=31930, D=18687
pred: (5.8898, 5.2835), actual: (5.9153, 5.2692), error: (-0.0255, 0.0142)

--- Inference 8/10 ---
cycles: 51271
dram<->spad DMA cycles: rdma=9011, wdma=3979
compute stall waiting on spad: A=32029, B=0, D=13262
compute stall waiting on acc:  A=0, B=31958, D=18713
pred: (5.0236, 5.3701), actual: (5.0229, 5.3479), error: (0.0007, 0.0222)

--- Inference 9/10 ---
cycles: 51257
dram<->spad DMA cycles: rdma=9003, wdma=3979
compute stall waiting on spad: A=32058, B=0, D=13248
compute stall waiting on acc:  A=0, B=31917, D=18686
pred: (5.4567, 5.2835), actual: (5.4127, 5.1218), error: (0.0440, 0.1617)

--- Inference 10/10 ---
cycles: 51402
dram<->spad DMA cycles: rdma=9026, wdma=3973
compute stall waiting on spad: A=32124, B=0, D=13311
compute stall waiting on acc:  A=0, B=32047, D=18752
pred: (5.8031, 5.4567), actual: (5.8423, 5.3954), error: (-0.0391, 0.0612)

==============================================
Avg cycles over 10 runs: 51293
Avg over 10 runs:
dram<->spad DMA cycles: rdma=9007, wdma=3973
compute stall waiting on spad: A=32029, B=0, D=13288
compute stall waiting on acc:  A=0, B=31983, D=18711
Avg error over 10 runs: (0.2185, 0.1513)
==============================================
BraggNN inference completed successfully
==============================================

==============================================
Pass 2: EXE / control-overhead counters
==============================================

--- Inference 1/10 (pass 2) ---
cycles: 51224
exe: active=17627, control_q_block=557, flush=0, overlap_haz=0
reservation station: full=47256, active=35887
loop_matmul active=17730, loop_conv active=18662

--- Inference 2/10 (pass 2) ---
cycles: 51155
exe: active=17608, control_q_block=557, flush=0, overlap_haz=0
reservation station: full=47154, active=35887
loop_matmul active=17743, loop_conv active=18662

--- Inference 3/10 (pass 2) ---
cycles: 51134
exe: active=17605, control_q_block=557, flush=0, overlap_haz=0
reservation station: full=47133, active=35884
loop_matmul active=17729, loop_conv active=18662

--- Inference 4/10 (pass 2) ---
cycles: 51132
exe: active=17597, control_q_block=557, flush=0, overlap_haz=0
reservation station: full=47131, active=35874
loop_matmul active=17725, loop_conv active=18661

--- Inference 5/10 (pass 2) ---
cycles: 51172
exe: active=17608, control_q_block=557, flush=0, overlap_haz=0
reservation station: full=47171, active=35887
loop_matmul active=17743, loop_conv active=18662

--- Inference 6/10 (pass 2) ---
cycles: 51160
exe: active=17604, control_q_block=557, flush=0, overlap_haz=0
reservation station: full=47159, active=35888
loop_matmul active=17736, loop_conv active=18664

--- Inference 7/10 (pass 2) ---
cycles: 51147
exe: active=17613, control_q_block=557, flush=0, overlap_haz=0
reservation station: full=47146, active=35892
loop_matmul active=17741, loop_conv active=18669

--- Inference 8/10 (pass 2) ---
cycles: 51167
exe: active=17608, control_q_block=557, flush=0, overlap_haz=0
reservation station: full=47166, active=35888
loop_matmul active=17742, loop_conv active=18664

--- Inference 9/10 (pass 2) ---
cycles: 51134
exe: active=17605, control_q_block=557, flush=0, overlap_haz=0
reservation station: full=47133, active=35884
loop_matmul active=17729, loop_conv active=18662

--- Inference 10/10 (pass 2) ---
cycles: 51187
exe: active=17597, control_q_block=557, flush=0, overlap_haz=0
reservation station: full=47186, active=35874
loop_matmul active=17729, loop_conv active=18661

==============================================
Avg cycles over 10 runs (pass 2): 51161
Avg over 10 runs:
exe: active=17607, control_q_block=557, flush=0, overlap_haz=0
reservation station: full=47163, active=35884
loop_matmul active=17734, loop_conv active=18662
==============================================

==============================================
Pass 3: controller-overlap (MAIN_*) counters
==============================================

--- Inference 1/10 (pass 3) ---
cycles: 51207
controller busy: ld_only=35216, st_only=1114, ex_only=0
controller busy: ld+st=0, ld+ex=215750, st+ex=71758, ld+st+ex=35458
none of ld/st/ex busy (true idle): 0
ex busy but not in compute state (preload/config overhead): 285643

--- Inference 2/10 (pass 3) ---
cycles: 51146
controller busy: ld_only=52848, st_only=1671, ex_only=0
controller busy: ld+st=0, ld+ex=319932, st+ex=107643, ld+st+ex=53217
none of ld/st/ex busy (true idle): 0
ex busy but not in compute state (preload/config overhead): 424808

--- Inference 3/10 (pass 3) ---
cycles: 51141
controller busy: ld_only=70475, st_only=2228, ex_only=0
controller busy: ld+st=0, ld+ex=424430, st+ex=143528, ld+st+ex=70949
none of ld/st/ex busy (true idle): 0
ex busy but not in compute state (preload/config overhead): 564262

--- Inference 4/10 (pass 3) ---
cycles: 51143
controller busy: ld_only=88100, st_only=2785, ex_only=0
controller busy: ld+st=0, ld+ex=528672, st+ex=179412, ld+st+ex=88678
none of ld/st/ex busy (true idle): 0
ex busy but not in compute state (preload/config overhead): 703455

--- Inference 5/10 (pass 3) ---
cycles: 51176
controller busy: ld_only=105726, st_only=3342, ex_only=0
controller busy: ld+st=0, ld+ex=632940, st+ex=215296, ld+st+ex=106410
none of ld/st/ex busy (true idle): 0
ex busy but not in compute state (preload/config overhead): 842679

--- Inference 6/10 (pass 3) ---
cycles: 51156
controller busy: ld_only=123350, st_only=3899, ex_only=0
controller busy: ld+st=0, ld+ex=737907, st+ex=251180, ld+st+ex=124139
none of ld/st/ex busy (true idle): 0
ex busy but not in compute state (preload/config overhead): 982597

--- Inference 7/10 (pass 3) ---
cycles: 51137
controller busy: ld_only=140968, st_only=4456, ex_only=0
controller busy: ld+st=0, ld+ex=842396, st+ex=287065, ld+st+ex=141867
none of ld/st/ex busy (true idle): 0
ex busy but not in compute state (preload/config overhead): 1122034

--- Inference 8/10 (pass 3) ---
cycles: 51170
controller busy: ld_only=158604, st_only=5013, ex_only=0
controller busy: ld+st=0, ld+ex=947387, st+ex=322955, ld+st+ex=159628
none of ld/st/ex busy (true idle): 0
ex busy but not in compute state (preload/config overhead): 1262012

--- Inference 9/10 (pass 3) ---
cycles: 51146
controller busy: ld_only=176239, st_only=5570, ex_only=0
controller busy: ld+st=0, ld+ex=1051902, st+ex=358845, ld+st+ex=177390
none of ld/st/ex busy (true idle): 0
ex busy but not in compute state (preload/config overhead): 1401516

--- Inference 10/10 (pass 3) ---
cycles: 51146
controller busy: ld_only=193882, st_only=6127, ex_only=0
controller busy: ld+st=0, ld+ex=1157362, st+ex=394732, ld+st+ex=195149
none of ld/st/ex busy (true idle): 0
ex busy but not in compute state (preload/config overhead): 1541959

==============================================
Avg cycles over 10 runs (pass 3): 51156
Avg over 10 runs:
controller busy: ld_only=114540, st_only=3620, ex_only=0
controller busy: ld+st=0, ld+ex=685867, st+ex=233241, ld+st+ex=115288
none of ld/st/ex busy (true idle): 0
ex busy but not in compute state (preload/config overhead): 913095
==============================================
```

The results of per-layer hardware counters is below.

```
- Per-layer hardware counters in Gemmini with BraggNN
  | Kernel | Pass 1 (cycles) | Pass 2 (cycles) | Pass 3 (cycles) | Avg (3 passes) |
  |---|---|---|---|---|
  | conv1 | 4550 | 4521 | 4517 | 4529.3 |
  | theta | 3585 | 3568 | 3574 | 3575.7 |
  | phi | 3561 | 3557 | 3566 | 3561.3 |
  | nlb_g | 3552 | 3559 | 3568 | 3559.7 |
  | attn_logits | 3110 | 3103 | 3102 | 3105.0 |
  | attn_softmax | 8413 | 8418 | 8408 | 8413.0 |
  | attended_matmul | 3035 | 3045 | 3033 | 3037.7 |
  | nlb_out_conv | 4271 | 4274 | 4264 | 4269.7 |
  | resadd | 5758 | 5759 | 5754 | 5757.0 |
  | conv2 | 9492 | 9478 | 9480 | 9483.3 |
  | conv3 | 2706 | 2709 | 2709 | 2708.0 |
  | fc1 | 1358 | 1357 | 1352 | 1355.7 |
  | fc2 | 755 | 759 | 750 | 754.7 |
  | fc3 | 747 | 736 | 750 | 744.3 |
  | fc4 | 742 | 724 | 733 | 733.0 |
  | output | 739 | 729 | 726 | 731.3 |
  | **Total (avg cycles)** | **64132** | **63698** | **63749** | **63859.7** |
- ## Pass 1 — DMA & Scratchpad Stall Counters
  
  | Kernel | rdma | wdma | Stall A (spad) | Stall B (spad) | Stall D (spad) | Stall A (acc) | Stall B (acc) | Stall D (acc) |
  |---|---|---|---|---|---|---|---|---|
  | conv1 | 211 | 889 | 3353 | 3360 | 3354 | 0 | 0 | 0 |
  | theta | 908 | 162 | 2581 | 2657 | 2655 | 0 | 0 | 0 |
  | phi | 919 | 162 | 2519 | 2588 | 2586 | 0 | 0 | 0 |
  | nlb_g | 908 | 162 | 2561 | 2632 | 2630 | 0 | 0 | 0 |
  | attn_logits | 321 | 642 | 1459 | 1467 | 1461 | 0 | 0 | 0 |
  | attn_softmax | 1659 | 645 | 4265 | 4271 | 4269 | 0 | 0 | 0 |
  | attended_matmul | 944 | 283 | 1341 | 1357 | 1351 | 0 | 0 | 0 |
  | nlb_out_conv | 633 | 888 | 3485 | 3448 | 3442 | 0 | 0 | 0 |
  | resadd | 5406 | 2275 | 5849 | 5857 | 5847 | 0 | 0 | 0 |
  | conv2 | 3218 | 98 | 4748 | 4732 | 4730 | 0 | 0 | 0 |
  | conv3 | 346 | 25 | 1860 | 1868 | 1862 | 0 | 0 | 0 |
  | fc1 | 294 | 4 | 877 | 884 | 878 | 0 | 0 | 0 |
  | fc2 | 19 | 1 | 811 | 819 | 813 | 0 | 0 | 0 |
  | fc3 | 18 | 1 | 794 | 802 | 796 | 0 | 0 | 0 |
  | fc4 | 18 | 1 | 774 | 782 | 776 | 0 | 0 | 0 |
  | output | 18 | 1 | 770 | 778 | 772 | 0 | 0 | 0 |
- ## Pass 2 — Execute / Reservation Station / Loop Counters
  
  | Kernel | exe active | control_q_block | flush | overlap_haz | RS full | RS active | loop_matmul active |
  |---|---|---|---|---|---|---|---|
  | conv1 | 1173 | 0 | 0 | 0 | 4302 | 3042 | 338 |
  | theta | 772 | 0 | 0 | 0 | 3452 | 2201 | 225 |
  | phi | 767 | 0 | 0 | 0 | 3438 | 2202 | 225 |
  | nlb_g | 772 | 0 | 0 | 0 | 3439 | 2201 | 225 |
  | attn_logits | 1633 | 0 | 0 | 0 | 3010 | 2427 | 2333 |
  | attn_softmax | 3932 | 0 | 0 | 0 | 7304 | 7365 | 7368 |
  | attended_matmul | 1615 | 0 | 0 | 0 | 2953 | 2356 | 2173 |
  | nlb_out_conv | 738 | 0 | 0 | 0 | 4120 | 2819 | 259 |
  | resadd | 0 | 0 | 0 | 0 | 5826 | 5677 | 4428 |
  | conv2 | 4131 | 358 | 0 | 0 | 8471 | 8102 | 1113 |
  | conv3 | 821 | 208 | 0 | 0 | 2567 | 1353 | 247 |
  | fc1 | 533 | 0 | 0 | 0 | 1394 | 737 | 509 |
  | fc2 | 45 | 0 | 0 | 0 | 848 | 141 | 47 |
  | fc3 | 45 | 0 | 0 | 0 | 819 | 145 | 47 |
  | fc4 | 45 | 0 | 0 | 0 | 797 | 145 | 47 |
  | output | 45 | 0 | 0 | 0 | 796 | 139 | 46 |
  
  *Note: `loop_conv active` = 0 for all kernels and is omitted.*  
- ## Pass 3 — Controller Busy Breakdown
  
  | Kernel | ld_only | st_only | ex_only | ld+st | ld+ex | st+ex | ld+st+ex | true idle | preload/config overhead |
  |---|---|---|---|---|---|---|---|---|---|
  | conv1 | 851 | 866 | 1003 | 0 | 235 | 84 | 0 | 1478 | 149 |
  | theta | 710 | 255 | 586 | 0 | 563 | 84 | 0 | 1376 | 461 |
  | phi | 729 | 255 | 586 | 0 | 545 | 84 | 0 | 1367 | 448 |
  | nlb_g | 710 | 255 | 586 | 0 | 563 | 84 | 0 | 1370 | 461 |
  | attn_logits | 98 | 556 | 853 | 0 | 278 | 639 | 0 | 678 | 137 |
  | attn_softmax | 146 | 2809 | 2126 | 0 | 1577 | 704 | 0 | 1046 | 475 |
  | attended_matmul | 62 | 356 | 935 | 0 | 933 | 67 | 0 | 680 | 320 |
  | nlb_out_conv | 990 | 866 | 585 | 0 | 291 | 84 | 0 | 1448 | 222 |
  | resadd | 3157 | 45 | 1 | 2471 | 0 | 0 | 0 | 80 | 1 |
  | conv2 | 2903 | 136 | 4444 | 0 | 549 | 68 | 0 | 1380 | 931 |
  | conv3 | 325 | 23 | 864 | 0 | 106 | 32 | 0 | 1359 | 181 |
  | fc1 | 129 | 4 | 402 | 0 | 198 | 0 | 0 | 619 | 67 |
  | fc2 | 43 | 4 | 88 | 0 | 0 | 0 | 0 | 615 | 43 |
  | fc3 | 47 | 4 | 88 | 0 | 0 | 0 | 0 | 611 | 43 |
  | fc4 | 47 | 4 | 88 | 0 | 0 | 0 | 0 | 594 | 43 |
  | output | 41 | 4 | 88 | 0 | 0 | 0 | 0 | 593 | 43 |
```

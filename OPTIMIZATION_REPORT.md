# BraggNN on Gemmini: hand-tuned C + RTL optimization report

Status as of 2026-09-25. BraggNN inference (one 11x11 patch -> (x, y) peak
position, int8) on Gemmini (`GemminiRocketConfig`, 16x16 weight-stationary
array, 256 KB scratchpad, 64 KB accumulator) with a Rocket core.

**Result: 45,170 cycles/patch (Exo seed, FireSim) -> 34,028 (hand-tuned C,
FireSim, stock RTL) -> 23,481 (C + RTL changes, Verilator).** Every step keeps
the predictions identical to the seed's (bit-exact per-patch errors).

| Stage | Where measured | cycles/patch | vs. previous |
|---|---|---:|---:|
| Exo seed schedule (`braggnn_schedule.py`) | FireSim, stock bitstream | 45,170 | |
| AlphaEvolve best schedule | FireSim, stock bitstream | 39,950 | -11.6% |
| Hand-tuned C (`exo/braggnn_tune_opus.c`) | FireSim, stock bitstream | 34,028 | -14.8% |
| Same C, unmodified RTL (771d30f) | Verilator | 33,814 | (reference for the RTL work) |
| C + RTL changes A-K, M + later C changes | Verilator | **23,481** | -30.6% vs. 33,814 |

The last row uses a narrower timing definition than the others (input
quantization and a one-time weight preload are outside the timed region);
see [Measurement](#how-it-was-measured). Under the original definition the
same build is roughly 23,481 + ~1,000 (quantization) cycles, plus the weight
loads that residency removes.

---

## How it was measured

### Timing
- `braggnn_eval` reads `rdcycle` at the start and end of every patch
  (`cycles[p] = end - begin`); `braggnn_main.c` prints them. The final
  int8 -> px dequantization runs in `braggnn_main.c`, outside that window.
- From `OPT_QUANT_OUTSIDE` on (agreed with the user), the fp32 -> int8 input
  quantization also runs before the patch loop. From `OPT_WEIGHTS_RESIDENT`
  on, weights are loaded into the scratchpad once before the patch loop (like
  the weight staging copies, which were always outside). Earlier rows and all
  seed / AlphaEvolve / `OPT_BASELINE` numbers include the quantization.
- **Warm average**: the first patch is cold (caches, TLB); Verilator numbers
  are the average of patches 1..n-1 (`-DEVAL_PATCHES=2` or `4`), FireSim
  numbers the average over 10 patches.

### Correctness
- Each run prints every patch's (x, y) prediction error. A build is accepted
  only if every patch's errors equal the seed's (`SEED_PATCH_ERRORS` in
  `src/verilator_eval.py`); predictions are int8 steps of 0.087 px, so any
  changed prediction shows. No per-layer comparison (the user's criterion is
  the final prediction).
- RTL changes additionally: no change to the ISA files (`GemminiISA.scala`,
  `LocalAddr.scala`, `gemmini.h`) and the generated `gemmini_params.h`
  unchanged; `src/verilator_eval.py` (Exo seed + fusion schedule) passes on the
  final RTL (seed 38,484, fusion 42,784 warm cycles/patch, both bit-exact).
- `OPT_BASELINE` (the Exo build's Gemmini calls) is run as a second check on
  each RTL step (41,439 unmodified -> 35,897 with A-M, bit-exact).

### Tools
| What | How |
|---|---|
| Build a C variant | `NAME=x scripts/build_c_tune.sh exo/braggnn_tune_opus.c [-DOPT_...] [-DEVAL_PATCHES=n]` (chia-exo container) |
| FireSim run | `python src/run_elf.py <ELFs>` (Ray job on the FPGA node) |
| Verilator run | `python src/verilator_elf.py <ELFs>` (Ray job; builds the simulator from the node's gemmini tree, runs the ELFs in parallel) |
| Per-layer profile | `-DCHIA_LAYER_MARKS`: a fence + cycle mark after every layer (adds a few hundred cycles in total, so marks builds read higher) |
| RTL cycle traces | diagnostic printf patches (`rtl_patches/D_trace_diag`, `H_trace_conv_rs`) + `verilator_elf.py --trace TRC` (runs with `+verbose`, keeps `TRC` lines): per-cycle reservation-station issue/occupancy, loop-unit stalls, store-controller command accepts, Normalizer inputs |
| RTL edits on the node | `python src/node_rtl.py status / diff / apply / revert` |
| Other workloads | `python src/rocc_tests.py` (gemmini-rocc-tests bare-metal tests on Verilator) |

Verilator runs are deterministic for a given ELF and RTL (repeat runs gave
identical per-patch cycles); Verilator and FireSim differ by ~1-4% on the
same build (e.g. 33,814 vs 34,028), so numbers are compared within one
platform only.

---

## Phase 1: hand-tuned C on the stock bitstream (FireSim)

Starting point: the Exo seed's Gemmini calls re-expressed in C (`OPT_BASELINE`).

| Step | cycles/patch | Bottleneck it removed |
|---|---:|---|
| Exo seed (Exo-generated C) | 45,170 | |
| `OPT_BASELINE` on the harness's arrays | 44,803 | |
| weights/biases staged into 64-byte-aligned buffers | 39,953 | unaligned DMA (~4.8k) |
| + softmax input loaded straight into the accumulator (no 81x81 identity matmul), fc1 fed conv3's output with permuted weights (no CPU flatten), qkv as loop_ws matmuls reusing their shared input (commit add29c8) | 34,939 | identity matmul, CPU copies, loop_conv overhead for 1x1 convs, triple input loads |
| + softmax loads its input once (skip-ldB bit) | 34,092 | duplicate 81x81 load |
| + NLB output conv as a loop_ws matmul | **34,028** | loop_conv overhead |

Per layer (FireSim, marks build): conv1 4,530, qkv 4,502, theta*phi^T 2,772,
softmax 5,164, attention*g 2,377, NLB out 2,152, resadd 3,093, conv2 6,588,
conv3 1,585, fc1 860, fc2..output 832.

Tried and rejected: conv1 as CPU im2col + matmul (+2.2k), fc tail on the CPU
(slower), scratchpad chaining with mvout_spad (did not write on the deployed
bitstream), conv2 split into several loops (`OPT_CONV2_SPLIT`, wrong
predictions even on the stock bitstream).

---

## Phase 2: Gemmini RTL (Verilator, same C)

Found with the RTL cycle traces. Patches in `rtl_patches/`, accepted set in
`rtl_patches/combined_A-M.patch`.

| Step | Change | Bottleneck | cycles |
|---|---|---|---:|
| - | unmodified 771d30f | | 33,814 |
| A | Normalizer stat IDs 2 -> 4 | only 2 softmax rows in flight | 33,456 |
| B | one 1/sum(exp) divider per stat ID | one shared iterative divider | 33,365 |
| C | store reservation-station entries 4 -> 8 | stores throttled by RS entries | 32,572 |
| E | stores wait only for unissued earlier stores (mvout_spad still for completion) | every store waited for all earlier stores to *complete*: softmax's 729 commands ran ~4 cycles apart | 31,839 |
| F, A2 | a MAX input no longer blocks the Normalizer for a cycle; stat IDs 8 | 2 cycles per block in the MAX pass | 31,357 |
| G | a row's division starts when its own lane inputs are done | division waited for all rows' sums | 31,037 |
| J | mvin scaler 4 units -> one per column | scaled mvins (resadd) at 4 cycles/row: resadd 3,080 -> 1,408 | 29,569 |
| K | `max_in_flight_mem_reqs` 16 -> 64 | Load/StoreController had only 2 commands in flight (latency-bound small loads/stores) | 27,841* |
| L | reservation station keeps mvout_spad's accumulator source (bug fix) | RS wiped it, so compute could overwrite accumulator rows a mvout_spad was reading | unchanged |
| M | DMA writer: 2-row queue between block gathering and TileLink writes | writer took the next row's blocks only when idle | -353 (see phase 3) |

\* K was measured after the C change `OPT_ATT_STRIDE` (28,830).
Dropped: load RS entries 8 -> 16 (no change), load+store 16 (-28).

---

## Phase 3: C changes on the modified RTL (Verilator)

| Step | Change | Bottleneck | cycles |
|---|---|---|---:|
| `OPT_ATT_STRIDE=128` | 81x81 buffers with 128-byte rows | unaligned 81-byte-stride stores/loads | 28,830 |
| (K, RTL) | | | 27,841 |
| 8-wide quantization | 8 independent fmul/fcvt chains | serial latency on the in-order core (14 cycles/element) | 27,039 |
| `OPT_WEIGHTS_RESIDENT` | loop_ws weights (qkv, NLB out, fc) loaded once, B = NULL per patch | per-patch weight reloads | 26,489 |
| `OPT_QUANT_OUTSIDE` | input quantization before the patch loop | (timing definition) | 25,595 |
| `OPT_CONV2_RESIDENT` | conv2 weights resident (b_spad_id = 2 fixes the region) | 18 KB reload, not overlapped in conv loops | 24,483 |
| `OPT_CONV1_WINDOW` | conv1 as a matmul over sliding 32-byte windows of the input (stride-1 mvins, zero-padded 32x64 weights) | loop_conv overhead for a 1-channel conv | 24,056 |
| `OPT_CONV3_RESIDENT` | conv3 weights resident in the freed id-1 conv region | weight reload | 23,834 |
| (M, RTL) | pipelined DMA writer | store tails | **23,481** |

Tried: the fc tail on the CPU again with parallel accumulators and fcvt
rounding (2,127 vs 848 cycles, slower); qkv scratchpad chaining with L
(bit-exact but slower, 26,164 vs 25,780: mvout_spad stores complete one at a
time and the three loops share one accumulator region); softmax chaining
(hangs, and the scratchpad store unit has no softmax sequence).

---

## Where the time goes now (trace of the 23,481 build, one patch)

| Segment | cycles | Limited by |
|---|---:|---|
| conv1 input mvins | 392 | 81 unaligned 32-byte reads |
| conv1 matmul | 1,574 | startup loads, then compute |
| qkv (3 matmuls) | 3,261 | the array (ex queue full) |
| theta*phi^T | 2,348 | compute, then a ~400-cycle store tail |
| softmax | 2,456 | stores: DRAM beats of 81-byte rows + store command rate |
| attention*g | 2,066 | compute, partly loads |
| NLB out | 1,790 | compute / loads |
| resadd | 1,233 | loads and stores |
| conv2 | 5,142 | the array: 504 7-row computes, 72 weight tiles (~4.2k) + input loads |
| conv3 | 1,078 | compute |
| fc1 + fc2..output (5 small loops) | ~1,000 | fixed per-loop latency |
| layer boundaries (x12) | ~900 | fence, CPU issuing configs, loop startup |

### Remaining bottlenecks
1. **The 16x16 array** (qkv, conv2, much of theta*phi^T and attention*g).
   Weight preloads already overlap computes; conv2 is costly because only 49
   output pixels share each weight tile. Beyond this needs a bigger array
   (changes DIM, i.e. the ISA).
2. **The 128-bit system bus** for every layer's activations (store, fence,
   reload); avoiding the round trip needs scratchpad chaining, which is slower
   today (see phase 3).
3. **Store path for odd-width rows** (softmax, theta*phi^T): 6 beats per
   81-byte row.
4. **Fixed overheads**: layer boundaries (~900) and the tiny fc tail (~660).

Not bottlenecks: scratchpad/accumulator capacity (well under a quarter used)
and scratchpad-to-array bandwidth (16-row computes issue exactly every 16
cycles).

---

## Open items
- **Area/power not measured.** A synthesis flow exists (`src/hammer_ppa.py`,
  sky130 + SRAM22/OpenRAM, needs the VLSI node). Estimated largest adds: J
  (12 extra fp32 scale pipelines) and B (7 extra fp32 dividers for only -91
  cycles; the first to drop if area matters).
- **FireSim confirmation** of the RTL changes needs a new bitstream (~5 h).
- **Other workloads**: gemmini-rocc-tests on the A-M RTL: 42 / 51 pass,
  matmul_spad fails as on the unmodified RTL, 8 large conv / tiled tests
  unfinished after 3 hours on Verilator (see below).
- Commits: branch `add-gcp-cluster` (not pushed); the firesim node's shared
  `~/gemmini4xraymodels` carries A-M in place.

## Other-workload validation (gemmini-rocc-tests)

`python src/rocc_tests.py` builds gemmini-rocc-tests' bare-metal tests on the
firesim node against the `gemmini_params.h` generated with the A-M simulator
(DIM 16, 4 banks, HAS_NORMALIZATIONS, NORM_STAT_IDS 2) and runs each on
Verilator; a test passes when its program exits 0 (each compares Gemmini's
results with a CPU reference). The `*_perf`, `gemmini_counter` and `template`
programs are left out.

First run, A-M RTL, 30-minute timeout per test: **29 / 51 passed, 1 failed,
21 timed out.**

- Passed: mvin_mvout (plain, zeros, stride, block_stride), mvin_mvout_acc
  (plain, zero_stride, full, full_stride), mvin_mvout_spad, raw_hazard,
  aligned, padded, mvin_scale, conv_stride, conv_rect, conv_first_layer,
  conv_dw, tiled_matmul_ws_At / _Bt / _full_C / _low_D / _igelu /
  _layernorm / _softmax, transpose, matrix_add, resadd, resadd_stride,
  global_average.
- **matmul_spad** fails with a store-controller assertion
  (`DMACommandTracker: bytes_left >= bytes_read`). **It fails identically on
  the unmodified 771d30f** (same assertion at the same simulation time), so it
  is a pre-existing bug in the mvout_spad store path, not caused by A-M.
- 21 tests hit the 30-minute limit before printing anything (their CPU
  reference computations are slow in Verilator; a Gemmini hang would instead
  trip the reservation station's 10,000-cycle stall assertion). Rerun with a
  3-hour limit: **13 more passed** (matmul, matmul_os, matmul_ws,
  mvin_mvout_acc_stride, conv_rect_pool, conv_with_pool,
  conv_trans_output_1203, conv_trans_weight_1203, conv_trans_weight_0132,
  conv_trans_input_3120, tiled_matmul_os, tiled_matmul_ws, tiled_matmul_cpu).
- **8 still unfinished after 3 hours** (no verdict): conv, conv_with_rot180,
  conv_with_kernel_dilation, conv_with_input_dilation,
  conv_with_input_dilation_and_rot180,
  conv_with_input_dilation_and_neg_padding,
  conv_trans_input_3120_with_kernel_dilation, tiled_matmul_option. They are
  the largest CPU-reference convs; not run on the unmodified RTL for
  comparison. A FireSim bitstream of A-M would run them in seconds.

**Total: 42 / 51 pass, 1 pre-existing failure (matmul_spad), 8 without a
verdict.**

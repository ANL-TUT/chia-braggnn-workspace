# Gemmini RTL patches for braggnn_tune_opus.c

Patches against gemmini4xraymodels 771d30f (`generators/gemmini`). No ISA file
(GemminiISA.scala, LocalAddr.scala, gemmini.h) changes, and the generated
gemmini_params.h is unchanged. Every accepted step printed exactly the seed's
per-patch prediction errors on Verilator (`src/verilator_elf.py`).

`combined_A-K.patch` is the accepted set in one diff (what the firesim node's
tree carries); the single-step files are relative to the step before them.

Verilator, current-best C (warm cycles/patch; 2-patch runs up to G, 4-patch after):

| step | patch | change | cycles |
|---|---|---|---|
| - | - | 771d30f unmodified | 33,814 |
| A | A_norm_stats4 | Normalizer stat ids 2 -> 4 (LoopMatmulStC `NORM_STAT_IDS` + `num_stats`) | 33,456 |
| B | B_per_stat_divider | one 1/sum-exp divider per stat id; one scale multiply in flight | 33,365 |
| C | C_rs_st8 | `reservation_station_entries_st` 4 -> 8 | 32,572 |
| E | E_rs_st_pipelined | stores wait only for unissued earlier stores (mvout_spad still waits for completion) | 31,839 |
| F, A2 | F_norm_max_nodrain, A2_norm_stats8 | a MAX input no longer blocks the Normalizer input a cycle; stat ids 8 | 31,357 |
| G | G_per_stat_lane_drain | a row's division starts once its own lane inputs are out | 31,037 |
| J | J_mvin_scale_parallel | mvin scaler `num_scale_units` 4 -> -1 (a scaled mvin row per cycle) | 29,569 |
| (SW) | - | `OPT_ATT_STRIDE=128` in the C file (aligned 81x81 rows) | 28,830 |
| K | K_inflight64 | `max_in_flight_mem_reqs` 16 -> 64 (Load/StoreController commands in flight 2 -> 5) | 27,841 |

Tried and dropped: I_rs_ld16 (loads 8 -> 16, no change), IC2 (ld/st 16, -28).
D_trace_diag / H_trace_conv_rs are printf-only diagnostics (run with
`src/verilator_elf.py --trace TRC`), not part of the set.

Scope: C and E help any workload with many stores; J any scaled mvin; K any
latency-bound load/store stream; A/B/F/G only softmax (and A layernorm).

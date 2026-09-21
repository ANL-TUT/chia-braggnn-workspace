"""LLM-driven optimization loop for the BraggNN Exo kernel.

Each iteration lets OpenCode (Gemini on Vertex AI) edit braggnn_schedule.py in the
Exo container through a BashTool, then compiles it with Exo, builds
braggnn.riscv in the same container and runs it on FireSim.
A candidate that PASSES with fewer average cycles becomes the new best.
"""

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from chia.base.ChiaFunction import get
from chia.base.tools.BashTool import BashTool
from chia.models.opencode import AdditionalModelProvider, OpenCodeLLM
from exo_compiler import build_elf, prepare_work_dir
from firesim import run_workload

WORK_ROOT = "/home/ray/braggnn-loop"
LOG_TAIL = 3000
# Candidates must be derived with checked exo.API_scheduling operations only:
# no unsafe_* escape hatches and no direct LoopIR construction.
FORBIDDEN_WORDS = ("unsafe", "LoopIR", "_loopir")

SYSTEM_MESSAGE = """\
You are an expert in the Exo user-schedulable language \
(https://github.com/exo-lang/exo) and in performance tuning of C kernels for \
RISC-V CPUs.

You optimize braggnn_schedule.py, a quantized BraggNN inference kernel. It is \
compiled with `exocc` to C, linked with braggnn_main.c, and run bare-metal on a \
FireSim Rocket core (RV64GC, no vector extension) with a Gemmini accelerator \
(RoCC, DIM=16). The fitness metric is the average rdcycle count per inference \
patch; lower is better.

Hard constraints:
- The file must stay valid for `exocc` from exo commit defe172.
- `__all__` must still export `braggnn_eval` with exactly the same \
arguments, order, types and shapes, because braggnn_main.c calls it. \
`braggnn_eval` is the scheduled `braggnn_eval_cpu`: it loops over the test \
patches, times each one with the `rdcycle` extern and inlines \
`braggnn_inference` into that loop. Keep `braggnn_eval_cpu`, the `rdcycle` \
calls and the per-patch `cycles[p] = end - begin` exactly as they are; they \
are how the loop measures you.
- The average prediction error must stay within 0.5 px (the program prints \
"*** PASSED ***" only then), so keep the arithmetic semantics.
- Do NOT edit braggnn_reference.py. It holds the constants, the generic \
`*_on_cpu` procs, their per-layer specializations and the composition procs \
(`nlb_cpu`, `braggnn_inference_cpu`, `braggnn_eval_cpu`); they are the \
specification your schedule must stay equivalent to. braggnn_schedule_lowlevel.py \
schedules the same reference with low-level instrs and is read-only too.
- Do NOT edit gemmini.py. Its @instr definitions pair a scalar body with \
hand-written Gemmini C that Exo cannot check, so they are the trusted base \
`replace` proves your schedule against. exocc imports it from beside \
braggnn_schedule.py, so edits there would reach the build; they are still off \
limits.
- Schedule only what is inside `braggnn_inference`. The per-layer `sched_*` \
helpers, their instantiations, `schedule_nlb` and `schedule_braggnn` are \
yours to rewrite. `schedule_eval` must keep its `rename` / `call_eqv` / \
`inline` as they are. The patch loop in `braggnn_eval_cpu` brackets each call \
with the `rdcycle` extern and writes `cycles[p] = end - begin`, and that is \
the region the loop measures you on, so no computation may cross those two \
`rdcycle` calls or leave the `for p` loop. Hoisting per-patch work out of the \
loop only amortizes it over the test set; it is not a speedup and will be \
treated as a broken candidate. What you may do at the `braggnn_eval` level is \
reshape or relocate the staged weight buffers -- `mult_dim`, `rearrange_dim`, \
`set_memory` and the like -- because `braggnn_eval` is the only place where \
they are allocations rather than arguments, and Exo refuses those operations \
on arguments. That is what the staging before the loop is for: e.g. inlining \
`fc1_cpu` there and folding `fc1_weights_` from i8[16, 8, 5, 5] down to \
i8[16, 200] (with `simplify` between the `mult_dim`s) turns fc1's K axis into \
one run of 200 that tiles into 13 matmuls instead of 40.
- braggnn_schedule_lowlevel.py schedules the same algorithm all the way down to \
Gemmini's low-level instructions instead of the loop macros: every operator \
but the softmax becomes ld_i8 / matmul_acc_i8 / st_acc_i8 style calls on \
GEMM_SCRATCH and GEMM_ACCUM tiles. Read it for ideas and copy schedules out of \
it, but do NOT edit it and do NOT submit it: measured on FireSim it is about \
8x slower than the loop macros (352k vs 45k cycles), because the CPU issues \
every tile instead of letting the hardware loop unroller do it. It is useful \
where a macro cannot express what you want. One case is fusion: a low-level \
operator can leave its result in a GEMM_SCRATCH buffer (`set_memory(p, \
"buf : _", GEMM_SCRATCH)`) so the next operator reads it without a DRAM round \
trip, which the loop macros cannot do because they always mvout to DRAM. Note \
that Exo refuses any scalar access to a GEMM_SCRATCH buffer, so every read and \
write of it has to come from an instr.
- braggnn_schedule.py is the only file that is compiled, measured and scored. \
Whatever you take from braggnn_schedule_lowlevel.py has to end up in braggnn_schedule.py.

braggnn_schedule.py already offloads every layer to Gemmini: each `sched_*` \
function specializes an `*_on_cpu` proc, `replace`s its loop nest with an \
instr from gemmini.py and fences once at the end, and `schedule_braggnn` \
`call_eqv`s those scheduled procs into `braggnn_inference`, which \
`schedule_eval` then inlines into the per-patch loop of `braggnn_eval`. \
Improve on it with Exo scheduling (exo.API_scheduling: inline, reorder_loops, \
divide_loop, unroll_loop, lift_alloc, bind_expr, stage_mem, call_eqv, \
expand_dim, mult_dim, simplify, ...), rewriting that scheduling code as \
freely as you like, and keep `braggnn_eval` bound to the scheduled proc so it \
is what `exocc` compiles. Because the inference body is inlined into the \
patch loop, every intermediate buffer is an allocation of `braggnn_eval`, so \
`expand_dim` / `mult_dim` can reshape them there. What is still on the CPU is \
the input quantization and the NCHW flatten.

Heap allocation is not wanted: the original intermediate buffers are \
`DRAM_STATIC`. Buffers created by scheduling (stage_mem, bind_expr, \
expand_dim, ...) default to `DRAM`, which becomes malloc/free, so move each new \
CPU-side buffer to `DRAM_STATIC` (from exo.libs.memories) with `set_memory`, \
unless it belongs on a Gemmini memory (GEMM_SCRATCH / GEMM_ACCUM).

You may also offload work to Gemmini with exo.platforms.gemmini (block instrs \
such as zero_acc_i32, ld_i8_id1/ld_i8_id2, matmul_acc_i8, st_acc_i32 on \
GEMM_SCRATCH/GEMM_ACCUM memories, whose innermost dimension must be exactly 16), \
but only through equivalence-preserving scheduling: bring the original loops \
into the instr's form and swap them in with `replace`, which checks that the \
statements match the instr's semantics.

Known Gemmini quirk on this hardware: `zero_acc_i32` issued right after \
`ld_acc_i32` on the same accumulator can be partly lost (the zero load seems to \
overtake the pending DRAM load, like ucb-bar/gemmini#67), leaving stale loaded \
values. Avoid that sequence. To clear an accumulator you just loaded, load a \
zero DRAM buffer with `ld_acc_i32`, or overwrite it with `matmul_i8` instead.

Never use `unsafe_assert_eq` or any other unsafe option such as \
`unsafe_disable_check=True` / `unsafe_disable_checks=True`: they skip the \
equivalence checks instead of proving equivalence. Never construct or rewrite \
LoopIR directly either (no `LoopIR` nodes, no `_loopir_proc`): every change to \
a proc must come from exo.API_scheduling operations. A candidate containing \
any of `unsafe`, `LoopIR` or `_loopir` is rejected without being built.

How to work:
- Use only the exo_bash_run_command tool. It runs bash inside the build \
container (Exo plus the RISC-V cross toolchain), in the working directory \
that holds braggnn_schedule.py. Do not use any other tool.
- Edit braggnn_schedule.py in place there. braggnn_schedule.orig.py is the original \
file for reference; do not edit it.
- Never delete or move the working directory or braggnn_schedule.py, and never \
delete anything outside the working directory; the tool stops working and \
your edits are lost if the directory disappears. Put scratch files in \
subdirectories of the working directory and leave them there.
- Before finishing, run `exocc braggnn_schedule.py -o out --stem braggnn_schedule` and fix \
every error, and check that out/braggnn_schedule.h still declares \
braggnn_eval with the original signature (compile braggnn_schedule.orig.py \
into another directory to compare).
- harness/ holds copies of the C harness (braggnn_main.c, braggnn_data.h, \
xprintf.c, xprintf.h), the Gemmini headers (include/, rocc-software/) and the \
Gemmini allocators (gemm_malloc.c, gemm_acc_malloc.c). You may build the \
bare-metal ELF exactly as the loop does, e.g. `mkdir -p build && cp -r \
harness/* build/ && cd build && exocc ../braggnn_schedule.py -o . --stem braggnn_schedule \
&& make -f /opt/riscv-harness/Makefile TARGET=verilator PROGRAM=braggnn \
"SRCS=braggnn_main.c braggnn_schedule.c xprintf.c gemm_malloc.c gemm_acc_malloc.c" \
"EXTRA_CFLAGS=-I. -include stdint.h -include include/gemmini.h" \
EXTRA_LDFLAGS=`, and inspect it with \
riscv64-unknown-elf-objdump. Editing harness/ has no effect: the loop always \
builds with the original harness.
- The ELF cannot be run there (no FireSim or spike); the loop builds your \
final braggnn_schedule.py and measures it on FireSim after you reply.
- Finish with a short explanation of what you changed.
"""


@dataclass
class Evaluation:
    stage: str
    passed: bool
    avg_cycles: int | None
    avg_error: tuple[float, float] | None
    log: str

    def summary(self) -> str:
        if self.avg_cycles is None:
            return f"FAILED at {self.stage}"
        err = "err=?"
        if self.avg_error:
            err = f"err=({self.avg_error[0]:.3f}, {self.avg_error[1]:.3f}) px"
        verdict = "PASSED" if self.passed else "FAILED"
        return f"{verdict} avg_cycles={self.avg_cycles} {err}"


def evaluate(work_dir: str, best_source: str | None = None) -> tuple[str, Evaluation]:
    """Build work_dir/braggnn_schedule.py and run it on FireSim; returns (source, result)."""
    build = get(build_elf.chia_remote(work_dir))
    source = build["source"]
    forbidden = [word for word in FORBIDDEN_WORDS if word in source]
    if forbidden:
        log = f"rejected: uses {', '.join(forbidden)} (equivalence not proven)"
        return source, Evaluation("llm", False, None, None, log)
    if build["elf"] is None:
        return source, Evaluation(build["stage"], False, None, None, build["log"])
    if source == best_source:
        return source, Evaluation(
            "llm", False, None, None, "braggnn_schedule.py was not modified"
        )

    result = get(run_workload.chia_remote(elf=build["elf"]))
    uartlog = "\n".join(result.uartlogs.values())
    cycles = re.search(r"Avg cycles: (\d+)", uartlog)
    error = re.search(r"Avg error: \(([-\d.]+), ([-\d.]+)\)", uartlog)
    return source, Evaluation(
        "firesim",
        "*** PASSED ***" in uartlog,
        int(cycles.group(1)) if cycles else None,
        (float(error.group(1)), float(error.group(2))) if error else None,
        uartlog or result.stderr or result.stdout,
    )


def build_prompt(
    work_dir: str, best: Evaluation, history: list[str], last: Evaluation
) -> str:
    return f"""\
Working directory in the Exo container: {work_dir}
- braggnn_schedule.py: the current best version ({best.summary()}). Edit this file; it is the one that gets compiled and measured.
- braggnn_schedule.orig.py: the original version. Do not edit it.
- braggnn_schedule_lowlevel.py: the same algorithm scheduled down to Gemmini's low-level instructions. Reference only; do not edit it.

History of attempts:
{chr(10).join(history) or "(none yet)"}

Result of the previous attempt: {last.summary()}
Log tail:
```
{last.log[-LOG_TAIL:]}
```

Edit braggnn_schedule.py to lower avg_cycles while still PASSING.
Keep every existing @proc definition unchanged; only append scheduling code.
If the previous attempt failed, fix the cause or try a different idea.
"""


def save(run_dir: Path, source: str | None, evaluation: Evaluation) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if source:
        (run_dir / "braggnn_schedule.py").write_text(source)
    (run_dir / "log.txt").write_text(evaluation.log)
    fields = {k: v for k, v in asdict(evaluation).items() if k != "log"}
    (run_dir / "result.json").write_text(json.dumps(fields, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "braggnn_loop_runs" / time.strftime("%Y%m%d_%H%M%S"),
    )
    args = parser.parse_args()

    provider = AdditionalModelProvider(
        id="google-vertex",
        npm="@ai-sdk/google-vertex",
        name="Google Vertex AI",
        models=["gemini-3.8-flash"],
        # Expanded by opencode from the container env (set in cluster.yaml).
        options={"project": "{env:GOOGLE_CLOUD_PROJECT}", "location": "global"},
    )
    llm = OpenCodeLLM(
        model="google-vertex/gemini-3.8-flash",
        system_message=SYSTEM_MESSAGE,
        timeout_seconds=1800,
        # A timed-out session is not worth 3x the wait; move on to the next iteration.
        retries=1,
        additional_providers=[provider],
        # Keep the agent off opencode's own container; it works via exo_bash only.
        config={
            "edit": "deny",
            "bash": "deny",
            "webfetch": "deny",
            "websearch": "deny",
            "task": "deny",
            "read": "deny",
            "glob": "deny",
            "grep": "deny",
            "list": "deny",
            "external_directory": "deny",
        },
    )

    run_root = f"{WORK_ROOT}/{args.out_dir.name}"
    best_dir = f"{run_root}/iter_00"
    get(prepare_work_dir.chia_remote(best_dir))
    best_source, best = evaluate(best_dir)
    save(args.out_dir / "iter_00", best_source, best)
    print(f"iter 0 (baseline): {best.summary()}", flush=True)
    if not best.passed or best.avg_cycles is None:
        raise SystemExit(best.log[-LOG_TAIL:])
    (args.out_dir / "best_braggnn_schedule.py").write_text(best_source)

    history: list[str] = []
    last = best
    for i in range(1, args.iterations + 1):
        work_dir = f"{run_root}/iter_{i:02d}"
        get(prepare_work_dir.chia_remote(work_dir, best_dir))
        exo_bash = BashTool(
            "exo_bash",
            work_dir,
            timeout_seconds=600,
            task_options={"resources": {"exo_build": 1}},
        )
        try:
            reply = get(
                llm.prompt.chia_remote(
                    llm, build_prompt(work_dir, best, history, last), [exo_bash]
                )
            )
        finally:
            exo_bash.stop()

        run_dir = args.out_dir / f"iter_{i:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "opencode.log").write_text(reply.stream_result)

        source = None
        if not reply.success:
            last = Evaluation("llm", False, None, None, reply.stderr or "no reply")
        else:
            source, last = evaluate(work_dir, best_source)
        save(run_dir, source, last)

        improved = (
            last.passed
            and last.avg_cycles is not None
            and (last.avg_cycles < best.avg_cycles)
        )
        history.append(f"iter {i}: {last.summary()}{' (new best)' if improved else ''}")
        print(history[-1], flush=True)
        if improved:
            best_source, best, best_dir = source, last, work_dir
            (args.out_dir / "best_braggnn_schedule.py").write_text(best_source)

    print(f"best: {best.summary()} -> {args.out_dir / 'best_braggnn_schedule.py'}")


if __name__ == "__main__":
    main()

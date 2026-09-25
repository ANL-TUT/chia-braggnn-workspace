import contextvars
import hashlib
import itertools
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

import ray
from ray import ObjectRef

import ae_backend_patch  # noqa: F401 -- passes evolution_settings to AlphaEvolve
from chia.base.ChiaFunction import get
from result_mapper import map_braggnn_results
from skydiscover.evaluation.chia_evaluator import ChiaEvaluator
from skydiscover.evaluation.evaluation_result import EvaluationResult

from exo_compiler import build_candidate_elf
from firesim import run_workload

logger = logging.getLogger(__name__)

_eval_binary: contextvars.ContextVar[Optional[bytes]] = contextvars.ContextVar(
    "_eval_binary", default=None,
)


@dataclass
class BraggnnBuildResult:
    success: bool
    binary: bytes  # the ELF
    stage: str  # "exo" | "build" | "llm" (build_elf's failing stage)
    stdout_tail: str  # the build log


@dataclass
class BraggnnRunResult:
    success: bool
    cycles: Optional[int]
    subpixel_error: Optional[float]  # worst of the two axes' average error
    stdout: str
    stderr: str
    uartlog: str = ""  # workload's own printf output
    # braggnn_main.c's profiled pass: avg cycles per layer (mark name -> cycles)
    layer_cycles: Optional[dict] = None
    # patches whose int8 prediction differs from the seed's (None: not printed)
    mismatches: Optional[int] = None


# braggnn_main.c's summary lines.
_CYCLES_RE = re.compile(r"Avg cycles:\s*(\d+)")
_ERROR_RE = re.compile(r"Avg error:\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")
_LAYER_RE = re.compile(r"^Layer (\w+): (\d+)", re.MULTILINE)
_MISMATCH_RE = re.compile(r"Prediction mismatches:\s*(\d+)")

MEASUREMENTS_FILE = "measurements.jsonl"


def source_sha256(program_solution: str) -> str:
    return hashlib.sha256(program_solution.strip().encode()).hexdigest()


_FORBIDDEN_EXO = re.compile(r"\bunsafe_[A-Za-z_]+|\b_loopir_proc\b|\bLoopIR\b")


def build_braggnn_binary(program_solution: str, gemmini_params_h: Optional[str] = None) -> Any:
    banned = _FORBIDDEN_EXO.search(program_solution)
    if banned:
        return ray.put(BraggnnBuildResult(
            success=False, binary=b"", stage="llm",
            stdout_tail=(
                f"rejected before building: `{banned.group(0)}` bypasses Exo's "
                "equivalence checks (unsafe_* operations and direct LoopIR "
                "construction are not allowed)"
            ),
        ))
    build = get(build_candidate_elf.chia_remote(program_solution, gemmini_params_h))
    return ray.put(BraggnnBuildResult(
        success=build["elf"] is not None,
        binary=build["elf"] or b"",
        stage=build["stage"],
        stdout_tail=build["log"],
    ))


class BraggnnEvaluator(ChiaEvaluator):
    def __init__(
        self,
        firesim_ready: ObjectRef,
        output_dir: str,
        fake_run: bool = False,
        gemmini_params_h: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._firesim_ready = firesim_ready
        # This attempt's elaborated gemmini_params.h for every candidate build
        # (None = the header shipped in exo/include).
        self._gemmini_params_h = gemmini_params_h
        # --sw-only test runs: no FireSim node, so "run" = the ELF's size in bytes.
        self._fake_run = fake_run
        # itertools.count().next() is a single GIL-protected C call, so it's
        # safe to share across the evaluator threads AlphaEvolve's shim spins
        # up per candidate (see bridge.py's _SHIM_TEMPLATE) without a lock.
        self._candidate_counter = itertools.count(1)
        # sha256(ELF) -> successful BraggnnRunResult. Identical generated code
        # links to an identical binary and, on deterministic hardware, gives
        # identical cycles, so re-running it on the FPGA teaches nothing.
        self._run_cache: dict = {}
        super().__init__(
            build_fn=self._build,
            run_fn=self._run,
            result_mapper_fn=map_braggnn_results,
            workloads=["braggnn"],
            output_dir=output_dir,
            **kwargs,
        )

    def _log_evaluation(self, *args: Any, **kwargs: Any) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        super()._log_evaluation(*args, **kwargs)

    async def evaluate_program(
        self, program_solution: str, program_id: str = "",
    ) -> EvaluationResult:
        result = await super().evaluate_program(program_solution, program_id)
        self._record_measurement(program_solution, result)
        return result

    def _record_measurement(self, program_solution: str, result: EvaluationResult) -> None:
        """Append the candidate's measured cycles / error, keyed by its source.
        Only the score reaches the AlphaEvolve server, so this is where the
        driver looks up the real numbers of the program it reports as best.
        One short append per line, so the evaluator threads need no lock."""
        artifacts = result.artifacts or {}
        record = {
            "sha256": source_sha256(program_solution),
            "combined_score": result.metrics.get("combined_score", 0.0),
            "cycles": artifacts.get("cycles"),
            "subpixel_error": artifacts.get("subpixel_error"),
            "failure_stage": artifacts.get("failure_stage"),
        }
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, MEASUREMENTS_FILE), "a") as f:
            f.write(json.dumps(record) + "\n")

    def _build(self, program_solution: str) -> Any:
        idx = next(self._candidate_counter)
        self._save_candidate_source(program_solution, idx)
        return build_braggnn_binary(program_solution, self._gemmini_params_h)

    def _save_candidate_source(self, program_solution: str, idx: int) -> None:
        """Persist every candidate AlphaEvolve generates (not just the best
        one per attempt) so failed/rejected candidates' code is inspectable
        after the run, alongside chia_eval_log.jsonl's per-candidate entries."""
        candidates_dir = os.path.join(self.output_dir, "candidates")
        os.makedirs(candidates_dir, exist_ok=True)
        path = os.path.join(candidates_dir, f"candidate_{idx:04d}.py")
        with open(path, "w", errors="replace") as f:
            f.write(program_solution or "")

    def _run(self, *, workload: str) -> Any:
        binary_content = _eval_binary.get()
        if binary_content is None:
            raise RuntimeError("No binary available -- build must succeed first")

        if self._fake_run:
            return ray.put(BraggnnRunResult(
                success=True, cycles=len(binary_content), subpixel_error=0.0,
                stdout="[sw-only] fake run: cycles = ELF size in bytes", stderr="",
                mismatches=0,
            ))

        firesim_ok, firesim_out, firesim_err = get(self._firesim_ready)
        if not firesim_ok:
            return ray.put(BraggnnRunResult(
                success=False, cycles=None, subpixel_error=None,
                stdout="", stderr=f"FireSim bitstream not ready:\n{firesim_out}\n{firesim_err}",
            ))

        run = get(run_workload.chia_remote(binary_content))
        uartlog = "\n".join(run.uartlogs.values())
        cycles = _CYCLES_RE.search(uartlog)
        error = _ERROR_RE.search(uartlog)
        mismatches = _MISMATCH_RE.search(uartlog)
        return ray.put(BraggnnRunResult(
            success=run.returncode == 0 and cycles is not None,
            cycles=int(cycles.group(1)) if cycles else None,
            subpixel_error=(
                max(float(error.group(1)), float(error.group(2))) if error else None
            ),
            stdout=run.stdout,
            stderr=run.stderr,
            uartlog=uartlog,
            layer_cycles={name: int(c) for name, c in _LAYER_RE.findall(uartlog)} or None,
            mismatches=int(mismatches.group(1)) if mismatches else None,
        ))

    async def _dispatch_runs(self, label: str) -> Any:
        binary = _eval_binary.get()
        key = hashlib.sha256(binary).hexdigest() if binary else None
        if key is not None and key in self._run_cache:
            logger.info(
                "Candidate%s: identical binary already evaluated; reusing its result "
                "(no FireSim run)", label,
            )
            return [self._run_cache[key]]
        results = await super()._dispatch_runs(label)
        if (
            key is not None
            and isinstance(results, list)
            and results
            and getattr(results[0], "success", False)
            and getattr(results[0], "cycles", None) is not None
        ):
            self._run_cache[key] = results[0]
        return results

    async def _dispatch_build(
        self,
        program_solution: str,
        label: str,
    ) -> Union[Any, EvaluationResult]:
        _eval_binary.set(None)
        logger.info("Candidate%s: build started (exocc -> cross-compile)", label)
        t0 = time.time()
        result = await super()._dispatch_build(program_solution, label)
        logger.info(
            "Candidate%s: build finished in %.0fs (success=%s)",
            label, time.time() - t0, getattr(result, "success", None),
        )

        if isinstance(result, EvaluationResult):
            return result
        if result is None:
            return None

        if not getattr(result, "success", False):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "score": 0.0},
                artifacts={
                    "failure_stage": "build",
                    "error_type": "BuildFailure",
                    "build_stage": result.stage,
                    "stderr": result.stdout_tail,
                },
            )

        _eval_binary.set(result.binary)
        return result

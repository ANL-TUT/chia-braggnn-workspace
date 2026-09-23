"""Result mapper: BraggnnRunResult list -> EvaluationResult.

combined_score = SEED_CYCLES / cycles, the speedup over the seed. AlphaEvolve
maximizes the score and registers the seed itself at 0.0, so a working candidate
must score above 0 and every failure (build, run, accuracy gate) scores exactly
0: with -cycles a broken candidate outscored every working one. The accuracy gate
(see below) keeps a fast-but-wrong candidate at 0 rather than winning on latency.

Every key in ``metrics`` is sent to the AlphaEvolve server as a score, i.e. a
target it optimizes, so metrics holds only the score. ``score`` duplicates
``combined_score`` because skydiscover picks the best program with
``order_by="score desc"`` and the server sorts by metric name; the pipeline
itself reads ``combined_score``. Cycles and error go to artifacts, which stay
in chia_eval_log.jsonl and are never sent to the server.
"""

from typing import Any, Dict, List

from skydiscover.evaluation.evaluation_result import EvaluationResult

# Measured avg cycles of exo/braggnn_schedule.py (the seed) on the deployed
# bitstream (results/20260922_171212/iter_00), so the score reads as a speedup.
SEED_CYCLES = 45170

# Target once the hardware numerics are fixed: reject candidates whose
# predictions are off from ground truth by more than this many sub-pixels.
MAX_ACCEPTABLE_SUBPIXEL_ERROR = 0.5

# Absolute gate: the fixed harness (braggnn_main.c) fails a run whose average
# error exceeds 0.5 px, so use the same limit. Set a float here for a relative
# gate (reference error + tolerance) instead.
ACCURACY_REFERENCE_ERROR = None
ACCURACY_TOLERANCE = 0.05


def _accuracy_limit() -> float:
    if ACCURACY_REFERENCE_ERROR is None:
        return MAX_ACCEPTABLE_SUBPIXEL_ERROR
    return ACCURACY_REFERENCE_ERROR + ACCURACY_TOLERANCE


def _score(value: float) -> Dict[str, float]:
    return {"combined_score": value, "score": value}


def map_braggnn_results(run_results: List[Any]) -> EvaluationResult:
    run = run_results[0]

    if not getattr(run, "success", False) or not run.cycles:
        return EvaluationResult(
            metrics=_score(0.0),
            artifacts={
                "failure_stage": "run",
                "stdout": run.stdout,
                "stderr": run.stderr,
                # The program's own UART output: without it a crash or a wrong
                # binary is indistinguishable from an infrastructure failure.
                "uartlog": (getattr(run, "uartlog", "") or "")[-3000:],
            },
        )

    # Unknown accuracy counts as a failure: it cannot be shown to match the seed.
    subpixel_error = run.subpixel_error if run.subpixel_error is not None else float("inf")
    if subpixel_error > _accuracy_limit():
        return EvaluationResult(
            metrics=_score(0.0),
            artifacts={
                "failure_stage": "accuracy_gate",
                "cycles": float(run.cycles),
                "subpixel_error": subpixel_error,
                "detail": (
                    f"sub-pixel error {subpixel_error:.4f} exceeds the limit "
                    f"{_accuracy_limit():.4f}; the candidate computes a "
                    "different result than the seed"
                ),
            },
        )

    return EvaluationResult(
        metrics=_score(SEED_CYCLES / float(run.cycles)),
        artifacts={"cycles": float(run.cycles), "subpixel_error": subpixel_error},
    )

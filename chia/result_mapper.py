"""Result mapper: BraggnnRunResult list -> EvaluationResult.

combined_score = -cycles (AlphaEvolve maximizes score; latency should be
minimized), gated by sub-pixel error (see below): a fast-but-wrong candidate scores 0
rather than winning on latency alone.
"""

from typing import Any, Dict, List

from skydiscover.evaluation.evaluation_result import EvaluationResult

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


def map_braggnn_results(run_results: List[Any]) -> EvaluationResult:
    run = run_results[0]

    if not getattr(run, "success", False) or run.cycles is None:
        return EvaluationResult(
            metrics={"error": 0.0, "combined_score": 0.0},
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
            metrics={
                "combined_score": 0.0,
                "cycles": float(run.cycles),
                "subpixel_error": subpixel_error,
            },
            artifacts={
                "failure_stage": "accuracy_gate",
                "detail": (
                    f"sub-pixel error {subpixel_error:.4f} exceeds the limit "
                    f"{_accuracy_limit():.4f}; the candidate computes a "
                    "different result than the seed"
                ),
            },
        )

    metrics: Dict[str, float] = {
        "combined_score": -float(run.cycles),
        "cycles": float(run.cycles),
    }
    metrics["subpixel_error"] = subpixel_error
    return EvaluationResult(metrics=metrics, artifacts={})

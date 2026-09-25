"""Result mapper: BraggnnRunResult list -> EvaluationResult.

combined_score = SEED_CYCLES / cycles, the speedup over the seed. AlphaEvolve
maximizes the score and registers the seed itself at 0.0, so a working candidate
must score above 0 and every failure (build, run, accuracy gate) scores exactly
0: with -cycles a broken candidate outscored every working one. The accuracy gate
(see below) keeps a fast-but-wrong candidate at 0 rather than winning on latency.

Every key in ``metrics`` is sent to the AlphaEvolve server as a score, i.e. a
target it optimizes. ``score`` duplicates ``combined_score`` because
skydiscover picks the best program with ``order_by="score desc"`` and the
server sorts by metric name; the pipeline itself reads ``combined_score``.
Once SEED_LAYER_CYCLES is filled in, metrics also carry one speedup per layer
group (``speedup_<group>``), which pareto sampling
(config_alphaevolve.yaml) uses to keep candidates that improved one part of
the network even when the total did not improve yet. Cycles, error and the
per-layer profile go to artifacts, which skydiscover's AlphaEvolve backend
sends to the server as insights: later generations see them, but they are
not optimized.
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

# braggnn_main.c's per-layer marks, grouped into the parts of the network a
# schedule change usually moves together.
LAYER_GROUPS = {
    "frontend": ("input_quantize", "conv1"),
    "nlb": (
        "nlb_qkv", "nlb_theta_phi", "nlb_softmax", "nlb_attention_g",
        "nlb_out_conv", "nlb_resadd",
    ),
    "conv2_conv3": ("conv2", "conv3"),
    "fc": ("flatten_fc1", "fc2_to_output"),
}

# The seed's cycles per layer group in the profiled pass (on the deployed
# bitstream), so speedup_<group> reads like combined_score. Left empty, the
# first run that reports every group sets it: skydiscover evaluates the seed
# on its own before the search starts, so that is the seed on the hardware at
# hand (each search runs in a fresh EvolverNode process).
SEED_LAYER_CYCLES: Dict[str, float] = {}


def _accuracy_limit() -> float:
    if ACCURACY_REFERENCE_ERROR is None:
        return MAX_ACCEPTABLE_SUBPIXEL_ERROR
    return ACCURACY_REFERENCE_ERROR + ACCURACY_TOLERANCE


def _score(value: float) -> Dict[str, float]:
    return {"combined_score": value, "score": value}


def _group_cycles(layer_cycles: Dict[str, int]) -> Dict[str, float]:
    groups = {}
    for group, names in LAYER_GROUPS.items():
        if all(name in layer_cycles for name in names):
            groups[group] = float(sum(layer_cycles[name] for name in names))
    return groups


def _layer_profile(layer_cycles: Dict[str, int]) -> str:
    total = sum(layer_cycles.values()) or 1
    lines = [
        (
            "Average cycles per patch and layer, from a second pass in which "
            "every layer boundary fences Gemmini (the scored pass has no such "
            "fences, so the sum is higher than the score's cycles):"
        )
    ]
    for name, cycles in layer_cycles.items():
        lines.append(f"  {name:16s} {cycles:7d}  ({100.0 * cycles / total:4.1f}%)")
    return "\n".join(lines)


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

    # Predictions must equal the seed's bit for bit: a wrong hand-written instr
    # C string or a dropped fence can stay under the error limit.
    mismatches = getattr(run, "mismatches", None)
    if mismatches is None or mismatches > 0:
        return EvaluationResult(
            metrics=_score(0.0),
            artifacts={
                "failure_stage": "exactness_gate",
                "cycles": float(run.cycles),
                "subpixel_error": subpixel_error,
                "detail": (
                    f"{mismatches if mismatches is not None else 'unknown'} of the "
                    "test patches predict a different int8 result than the seed; "
                    "every schedule must compute exactly the seed's arithmetic"
                ),
                "uartlog": (getattr(run, "uartlog", "") or "")[-3000:],
            },
        )

    metrics = _score(SEED_CYCLES / float(run.cycles))
    artifacts: Dict[str, Any] = {
        "cycles": float(run.cycles),
        "subpixel_error": subpixel_error,
    }
    layer_cycles = getattr(run, "layer_cycles", None)
    if layer_cycles:
        artifacts["layer_profile"] = _layer_profile(layer_cycles)
        groups = _group_cycles(layer_cycles)
        if not SEED_LAYER_CYCLES and len(groups) == len(LAYER_GROUPS):
            SEED_LAYER_CYCLES.update(groups)
        for group, cycles in groups.items():
            if SEED_LAYER_CYCLES.get(group) and cycles > 0:
                metrics[f"speedup_{group}"] = SEED_LAYER_CYCLES[group] / cycles
    return EvaluationResult(metrics=metrics, artifacts=artifacts)

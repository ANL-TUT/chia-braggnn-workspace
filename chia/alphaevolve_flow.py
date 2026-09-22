import logging
import os
from typing import Optional

import ray
import yaml
from ray import ObjectRef

from chia.base.ChiaFunction import ChiaFunction, get

from braggnn_evaluator import BraggnnEvaluator
from evolve_flows.evolver.node import EvolverNode
from evolve_flows.evolver.types import EvolverInput
from firesim import read_gemmini_params_h

from constants import DEFAULT_OUTPUT_BASE, PACKAGE_DIR
from prompts import _EXO_SEED, _GEMMINI_PY

logger = logging.getLogger(__name__)


@ChiaFunction(resources={"evolver": 0.01})
def _read_eval_log(output_dir: str) -> str:
    path = os.path.join(output_dir, "chia_eval_log.jsonl")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


@ChiaFunction(resources={"evolver": 0.01})
def _read_candidates(output_dir: str) -> dict:
    """Every candidate BraggnnEvaluator._save_candidate_source wrote on the
    evolver node during this attempt -- name -> source, in generation order."""
    candidates_dir = os.path.join(output_dir, "candidates")
    if not os.path.isdir(candidates_dir):
        return {}
    result = {}
    for name in sorted(os.listdir(candidates_dir)):
        path = os.path.join(candidates_dir, name)
        if os.path.isfile(path):
            with open(path, errors="replace") as f:
                result[name] = f.read()
    return result


DEFAULT_CONFIG = str(PACKAGE_DIR.parent / "config_alphaevolve.yaml")  # workspace root
EVOLVER_ACTOR_NAME = "braggnn-evolver"
EVOLVER_NAMESPACE = "chia-gemmini-braggnn"
SEED_PROGRAM = _EXO_SEED


# ── Callable entry point (invoked once per outer HW attempt) ─────────

def run_alphaevolve_search(
    dump,
    attempt: int,
    initial_program: str,
    firesim_ready: ObjectRef,
    config_path: str = DEFAULT_CONFIG,
    eval_dir: Optional[str] = None,
    hw_change_summary: str = "",
    hw_context: Optional[str] = None,
    fake_run: bool = False,
):
    # *eval_dir* is resolved in the evolver container, where the evaluator runs;
    # the driver's own artifacts go through *dump* instead.
    if eval_dir is None:
        eval_dir = DEFAULT_OUTPUT_BASE

    actor_name = f"{EVOLVER_ACTOR_NAME}-attempt{attempt}"

    try:
        stale = ray.get_actor(actor_name, namespace=EVOLVER_NAMESPACE)
        logger.warning(
            "Found stale evolver actor '%s' from a previous run -- killing it",
            actor_name,
        )
        ray.kill(stale)
    except ValueError:
        pass

    evaluator = BraggnnEvaluator(
        firesim_ready=firesim_ready,
        output_dir=eval_dir,
        fake_run=fake_run,
        timeout=3600.0,
        max_retries=1,
    )

    logger.info("Creating EvolverNode actor '%s'", actor_name)
    evolver = EvolverNode.options(
        name=actor_name,
        namespace=EVOLVER_NAMESPACE,
        lifetime="detached",
        resources={"evolver": 1.0},
    ).remote()

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    ae_config = config_dict.setdefault("alphaevolve", {})
    if ae_config.get("project_id") == "GOOGLE_CLOUD_PROJECT":
        ae_config["project_id"] = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if ae_config.get("engine_id") == "GE_APP_ID":
        ae_config["engine_id"] = os.environ.get("GE_APP_ID", "")

    # Tell the SW search which deployed hardware it is scheduling against.
    if hw_context is None:
        hw_context = get(
            read_gemmini_params_h.options(resources={"manager": 0.05}).chia_remote()
        )
    if hw_change_summary:
        hardware_heading = "## Current Gemmini hardware (this co-design attempt)"
        hardware_description = (
            "gemmini_params.h, the C-level hardware constants for this build:"
        )
    else:
        hardware_heading = "## Current fixed Gemmini hardware"
        hardware_description = (
            "gemmini_params.h, the C-level hardware constants for the deployed "
            "FireSim bitstream:"
        )
    base_problem_description = ae_config.get("problem_description", "")
    ae_config["problem_description"] = (
        f"{base_problem_description}\n\n"
        f"{hardware_heading}\n\n"
        f"{hardware_description}\n\n"
        f"```c\n{hw_context}\n```\n\n"
        "## The `gemmini` helper module (read-only)\n\n"
        "gemmini.py, imported by the program as `from gemmini import ...`. It is "
        "the hardware instruction library: written next to your program before "
        "it is compiled, trusted and fixed, and NOT part of what you edit. Each "
        "make_loop_* function builds an Exo `@instr` whose body is a plain loop "
        "nest (the semantics `replace` matches against) and whose C string is "
        "one Gemmini hardware-loop instruction sequence:\n\n"
        f"```python\n{_GEMMINI_PY}\n```"
        + (
            "\n\n## What the hardware-tuning process changed this attempt, "
            "in its own words\n\n"
            "This is the hardware LLM's own explanation of the Chisel change "
            "behind the numbers above -- use it to understand structural "
            "changes (e.g. cascaded/restructured systolic arrays) that plain "
            f"parameter values don't capture:\n\n{hw_change_summary}"
            if hw_change_summary else ""
        )
    )

    config_content = yaml.safe_dump(config_dict, sort_keys=False)

    evolver_input = EvolverInput(
        config_path=os.path.basename(config_path),
        initial_program=initial_program,
        config_content=config_content,
    )

    logger.info("Launching AlphaEvolve search for attempt %d", attempt)
    result_ref = evolver.run_search.remote(
        evolver_input,
        build_fn=evaluator._build,
        run_fn=evaluator._run,
        result_mapper_fn=evaluator.result_mapper_fn,
        evaluator=evaluator,
    )
    result = ray.get(result_ref)

    logger.info(
        "AlphaEvolve search finished (attempt %d): terminal_status=%s, iterations=%d",
        attempt, result.terminal_status, result.iteration_count,
    )
    if result.best_program:
        dump.text(f"best_exo_attempt{attempt}.py", result.best_program)

    evaluator.close()
    ray.kill(evolver)

    eval_log = get(_read_eval_log.options(resources={"evolver": 0.01}).chia_remote(eval_dir))
    if eval_log:
        dump.text(f"chia_eval_log_attempt{attempt}.jsonl", eval_log)

    candidates = get(
        _read_candidates.options(resources={"evolver": 0.01}).chia_remote(eval_dir)
    )
    for name, content in candidates.items():
        dump.text(f"attempt{attempt}_{name}", content)
    if candidates:
        logger.info(
            "Saved %d candidate program(s) for attempt %d", len(candidates), attempt,
        )

    return result

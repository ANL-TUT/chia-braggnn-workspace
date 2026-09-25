import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import ray
import yaml
from ray import ObjectRef

from chia.base.ChiaFunction import ChiaFunction, get

from braggnn_evaluator import MEASUREMENTS_FILE, BraggnnEvaluator, source_sha256
from evolve_flows.evolver.node import EvolverNode
from evolve_flows.evolver.types import EvolverInput
from firesim import read_gemmini_params_h

from constants import DEFAULT_OUTPUT_BASE, PACKAGE_DIR
from prompts import _EXO_SEED, _GEMMINI_H_EXCERPT, _GEMMINI_PY

logger = logging.getLogger(__name__)


@ChiaFunction(resources={"evolver": 0.01})
def _read_evolver_env(names: list) -> dict:
    """Env vars as the evolver container sees them. cluster.yaml passes
    GOOGLE_CLOUD_PROJECT / GE_APP_ID into that container; the job driver does
    not have them, because .env is not shipped with the job."""
    return {name: os.environ.get(name, "") for name in names}


@ChiaFunction(resources={"evolver": 0.01})
def _read_eval_log(output_dir: str) -> str:
    path = os.path.join(output_dir, "chia_eval_log.jsonl")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


@ChiaFunction(resources={"evolver": 0.01})
def _read_measurements(output_dir: str) -> str:
    path = os.path.join(output_dir, MEASUREMENTS_FILE)
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


@ChiaFunction(resources={"evolver": 0.01})
def _read_candidates(output_dir: str, skip: int = 0) -> dict:
    """Every candidate BraggnnEvaluator._save_candidate_source wrote on the
    evolver node during this attempt -- name -> source, in generation order --
    except the first *skip*, which the caller already has."""
    candidates_dir = os.path.join(output_dir, "candidates")
    if not os.path.isdir(candidates_dir):
        return {}
    result = {}
    for name in sorted(os.listdir(candidates_dir))[skip:]:
        path = os.path.join(candidates_dir, name)
        if os.path.isfile(path):
            with open(path, errors="replace") as f:
                result[name] = f.read()
    return result


DEFAULT_CONFIG = str(PACKAGE_DIR.parent / "config_alphaevolve.yaml")  # workspace root
# How often the driver copies the evolver's candidates / measurements back
# while the search runs, so a search stopped midway keeps its results.
PROGRESS_POLL_SECONDS = 60
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
    gemmini_params_h: Optional[str] = None,
    max_iterations: Optional[int] = None,
):
    # *eval_dir* is resolved in the evolver container, where the evaluator runs;
    # the driver's own artifacts go through *dump* instead.
    if eval_dir is None:
        eval_dir = DEFAULT_OUTPUT_BASE
    # One subdirectory per attempt: each attempt's evaluator numbers its
    # candidates from 1 again and appends to its own measurements file, so a
    # shared directory would mix attempts (and overwrite candidate files).
    eval_dir = os.path.join(eval_dir, f"attempt{attempt}")

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

    # Must run before the EvolverNode actor below takes the whole evolver resource.
    evolver_env = get(
        _read_evolver_env.options(resources={"evolver": 0.01}).chia_remote(
            ["GOOGLE_CLOUD_PROJECT", "GE_APP_ID"]
        )
    )

    evaluator = BraggnnEvaluator(
        firesim_ready=firesim_ready,
        output_dir=eval_dir,
        fake_run=fake_run,
        gemmini_params_h=gemmini_params_h,
        timeout=3600.0,
        max_retries=1,
    )

    logger.info("Creating EvolverNode actor '%s'", actor_name)
    # 0.9, not 1.0: the actor holds this for its lifetime, and _LiveProgress
    # reads the evaluator's files with {"evolver": 0.01} tasks while it runs.
    # Two actors still cannot share the node.
    evolver = EvolverNode.options(
        name=actor_name,
        namespace=EVOLVER_NAMESPACE,
        lifetime="detached",
        resources={"evolver": 0.9},
    ).remote()

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    if max_iterations is not None:
        config_dict["max_iterations"] = max_iterations
    ae_config = config_dict.setdefault("alphaevolve", {})
    # The server's max_programs includes the seed ("The initial program counts
    # towards this limit"), while the client stops after max_iterations *new*
    # candidates. With equal values the server finishes one short and the client
    # idles until idle_timeout_s, so derive it here rather than in the YAML.
    ae_config["max_programs"] = config_dict["max_iterations"] + 1
    if ae_config.get("project_id") == "GOOGLE_CLOUD_PROJECT":
        ae_config["project_id"] = evolver_env["GOOGLE_CLOUD_PROJECT"]
    if ae_config.get("engine_id") == "GE_APP_ID":
        ae_config["engine_id"] = evolver_env["GE_APP_ID"]

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
        f"```python\n{_GEMMINI_PY}\n```\n\n"
        "## gemmini.h (excerpt, read-only)\n\n"
        "The C header every instr's C string compiles against (included "
        "before the generated code). For writing your own instrs: the RoCC "
        "command codes, the command macros including the LOOP_WS hardware "
        "loop, and tiled_matmul_outer, which shows how gemmini.h itself issues "
        "LOOP_WS:\n\n"
        f"```c\n{_GEMMINI_H_EXCERPT}\n```"
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
    progress = _LiveProgress(dump.out_dir, attempt, eval_dir)
    logger.info("Live results: %s", progress.dir)
    # The evolver is a detached actor: it outlives this driver, so kill it on
    # every exit path (incl. `ray job stop`, see main's SIGTERM handler) rather
    # than leave it running candidates on the FPGA.
    try:
        result_ref = evolver.run_search.remote(
            evolver_input,
            build_fn=evaluator._build,
            run_fn=evaluator._run,
            result_mapper_fn=evaluator.result_mapper_fn,
            evaluator=evaluator,
        )
        # Short waits so a SIGTERM is handled within the few seconds `ray job
        # stop` allows before SIGKILL.
        last_sync = time.monotonic()
        while not ray.wait([result_ref], timeout=5)[0]:
            if time.monotonic() - last_sync >= PROGRESS_POLL_SECONDS:
                progress.sync()
                last_sync = time.monotonic()
        result = ray.get(result_ref)
        progress.sync()
    finally:
        ray.kill(evolver)

    logger.info(
        "AlphaEvolve search finished (attempt %d): terminal_status=%s, iterations=%d",
        attempt, result.terminal_status, result.iteration_count,
    )
    if result.best_program:
        dump.text(f"best_exo_attempt{attempt}.py", result.best_program)

    evaluator.close()

    eval_log = get(_read_eval_log.options(resources={"evolver": 0.01}).chia_remote(eval_dir))
    if eval_log:
        dump.text(f"chia_eval_log_attempt{attempt}.jsonl", eval_log)

    measurements = get(
        _read_measurements.options(resources={"evolver": 0.01}).chia_remote(eval_dir)
    )
    if measurements:
        dump.text(f"{MEASUREMENTS_FILE[:-6]}_attempt{attempt}.jsonl", measurements)
    _attach_measurement(result, measurements)

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


class _LiveProgress:
    """Mirrors the search onto the driver while it runs, into
    <out_dir>/attempt<N>_live/: candidates/, measurements.jsonl, progress.txt
    (one line per measured candidate) and, whenever a candidate beats the
    previous best, best_braggnn_schedule.py + best.json. The files exist only
    in the evolver container until copied here, so this is what is left if the
    job is stopped before the search ends."""

    def __init__(self, out_dir: Path, attempt: int, eval_dir: str):
        self.dir = Path(out_dir) / f"attempt{attempt}_live"
        (self.dir / "candidates").mkdir(parents=True, exist_ok=True)
        self.eval_dir = eval_dir
        self.sources: dict[str, tuple[str, str]] = {}  # sha256 -> (name, source)
        self.n_candidates = 0
        self.n_measured = 0
        self.best: Optional[dict] = None

    def sync(self) -> None:
        try:
            new = get(_read_candidates.options(resources={"evolver": 0.01})
                      .chia_remote(self.eval_dir, self.n_candidates))
            measurements = get(_read_measurements.options(resources={"evolver": 0.01})
                               .chia_remote(self.eval_dir))
        except Exception as e:  # noqa: BLE001 -- progress is best-effort
            logger.warning("Progress sync failed: %s", e)
            return
        for name, source in new.items():
            (self.dir / "candidates" / name).write_text(source, errors="replace")
            self.sources[source_sha256(source)] = (name, source)
        self.n_candidates += len(new)

        lines = measurements.splitlines()
        (self.dir / MEASUREMENTS_FILE).write_text(measurements)
        with open(self.dir / "progress.txt", "a") as progress:
            for line in lines[self.n_measured:]:
                record = json.loads(line)
                name, source = self.sources.get(record["sha256"], ("?", None))
                improved = (
                    record.get("cycles") is not None
                    and not record.get("failure_stage")
                    and record.get("combined_score", 0.0) > 0.0
                    and (self.best is None
                         or record["combined_score"] > self.best["combined_score"])
                )
                if record.get("failure_stage"):
                    verdict = f"FAILED at {record['failure_stage']}"
                else:
                    verdict = "PASSED"
                entry = (
                    f"{name}: {verdict} cycles={record.get('cycles')} "
                    f"subpixel_error={record.get('subpixel_error')} "
                    f"score={record.get('combined_score', 0.0):.4f}"
                    + (" (new best)" if improved else "")
                )
                progress.write(entry + "\n")
                logger.info("Candidate %s", entry)
                if improved and source is not None:
                    self.best = {**record, "candidate": name}
                    (self.dir / "best_braggnn_schedule.py").write_text(source)
                    (self.dir / "best.json").write_text(json.dumps(self.best, indent=2))
        self.n_measured = len(lines)


def _attach_measurement(result, measurements: str) -> None:
    """Put the best program's measured cycles / sub-pixel error into
    best_metrics. The server only knows the score, so best_metrics holds just
    that until it is matched here, by source, to what the evaluator measured."""
    if not result.best_program or result.best_metrics is None:
        return
    key = source_sha256(result.best_program)
    for line in measurements.splitlines():
        record = json.loads(line)
        if record["sha256"] == key and record.get("cycles") is not None:
            result.best_metrics["cycles"] = record["cycles"]
            result.best_metrics["subpixel_error"] = record["subpixel_error"]
            return

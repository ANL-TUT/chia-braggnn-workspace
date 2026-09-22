import logging
import uuid
from dataclasses import dataclass, field

from chia.base.ChiaFunction import get
from chia.chipyard.chipyard_hammer import ChipyardHammerNode

from constants import (
    BUILD_CONFIG, FIRESIM_CHIPYARD_PATH, VLSI_BUILDFILE_TIMEOUT_SECONDS,
    VLSI_INPUT_CONFS, VLSI_OBJ_DIR_ROOT, VLSI_SYN_TIMEOUT_SECONDS, VLSI_TOP,
)
from dumper import Dumper

logger = logging.getLogger(__name__)

MAKE_VARS = {"tech_name": "sky130", "VLSI_TOP": VLSI_TOP,
             "INPUT_CONFS": " ".join(VLSI_INPUT_CONFS)}


@dataclass
class PpaResult:
    success: bool
    stage: str                # "buildfile" | "syn"
    obj_dir: str               # on the vlsi worker
    final_area_rpt: str = ""   # syn-rundir/reports/final_area.rpt, if collected
    reports: dict[str, str] = field(default_factory=dict)
    stderr_tail: str = ""


def run_ppa_synthesis(dump: Dumper, attempt: int, config: str = BUILD_CONFIG) -> PpaResult:
    """Synthesize `config` (a chipyard Config name -- BUILD_CONFIG by default,
    the same one FireSim/Verilator just elaborated) with Genus/sky130/SRAM22
    and save its area/timing reports. Best-effort: never raises. A failure
    here does not by itself say anything about the attempt's FireSim/cycles
    result -- callers decide whether/how to fold this into their own
    feedback."""
    obj_dir = f"{VLSI_OBJ_DIR_ROOT}/attempt{attempt}-{uuid.uuid4().hex[:8]}"
    logger.info("Hammer PPA: CONFIG=%s OBJ_DIR=%s (attempt %d)", config, obj_dir, attempt)

    try:
        with ChipyardHammerNode() as node:
            bf = get(node.make.chia_remote(
                FIRESIM_CHIPYARD_PATH, "buildfile", config=config, obj_dir=obj_dir,
                make_vars=MAKE_VARS, timeout_seconds=VLSI_BUILDFILE_TIMEOUT_SECONDS,
            ))
            dump.text(f"hammer_ppa_attempt{attempt}_buildfile.stdout.txt", bf.stdout)
            dump.text(f"hammer_ppa_attempt{attempt}_buildfile.stderr.txt", bf.stderr)
            if not bf.success:
                logger.warning("Hammer PPA buildfile failed (attempt %d, rc=%s)",
                                attempt, bf.returncode)
                return PpaResult(False, "buildfile", obj_dir, stderr_tail=bf.stderr[-2000:])

            syn = get(node.make.chia_remote(
                FIRESIM_CHIPYARD_PATH, "syn", config=config, obj_dir=obj_dir,
                make_vars=MAKE_VARS, timeout_seconds=VLSI_SYN_TIMEOUT_SECONDS,
            ))
            dump.text(f"hammer_ppa_attempt{attempt}_syn.stdout.txt", syn.stdout)
            dump.text(f"hammer_ppa_attempt{attempt}_syn.stderr.txt", syn.stderr)

            rpts = get(node.collect.chia_remote(
                obj_dir, ["syn-rundir/reports/**", "syn-rundir/*.log"],
                max_bytes_per_file=2_000_000,
            ))
    except Exception as e:  # noqa: BLE001 -- PPA is best-effort, never break the loop
        logger.warning("Hammer PPA failed (attempt %d): %s", attempt, e)
        return PpaResult(False, "syn", obj_dir, stderr_tail=str(e))

    dump.json(f"hammer_ppa_attempt{attempt}_reports.json", rpts.files)
    area = rpts.files.get("syn-rundir/reports/final_area.rpt", "")
    if area:
        dump.text(f"hammer_ppa_attempt{attempt}_final_area.rpt", area)

    logger.info("Hammer PPA %s (attempt %d): %d report(s)",
                "OK" if syn.success else "FAILED", attempt, len(rpts.files))
    return PpaResult(
        success=syn.success, stage="syn", obj_dir=obj_dir,
        final_area_rpt=area, reports=rpts.files,
        stderr_tail="" if syn.success else syn.stderr[-2000:],
    )

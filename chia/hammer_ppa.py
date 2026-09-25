import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import ray
from chia.base.ChiaFunction import chia_cancel, get
from chia.chipyard.chipyard_hammer import ChipyardHammerNode

from constants import (
    BUILD_CONFIG, CHIPYARD_PATH, VLSI_BUILDFILE_TIMEOUT_SECONDS,
    VLSI_INPUT_CONFS, VLSI_OBJ_DIR_ROOT, VLSI_PG_READY_TIMEOUT_SECONDS,
    VLSI_SYN_TIMEOUT_SECONDS, VLSI_TOP,
)
from dumper import Dumper

logger = logging.getLogger(__name__)

PPA_FLOW_LABEL = "sky130+SRAM22/OpenRAM"
# Caveat on the macro numbers for the LLM (gemmini-sky130-sram22's
# docs/openram-1r1w.md "Caveats").
SRAM_CAVEAT = (
    "single-port memories map to SRAM22 macros, dual-port ones to OpenRAM "
    "1rw1r macros (at most 2 KiB each, ~2.5x less dense than SRAM22, so a "
    "dual-ported memory costs far more area than a single-ported one); SRAM "
    "macro libs are typical-corner only (OpenRAM's analytically), so macro "
    "timing is optimistic, and macro leakage reads as 0"
)

# generated_src_name: MacroCompiler's output under vlsi/generated-src
# depends on the plugin's sram-cache.json, which make does not track, so keep
# this flow's elaborations out of the default directory, where an older
# SRAM22-only mapping of the same config could otherwise be reused as-is.
MAKE_VARS = {"tech_name": "sky130", "VLSI_TOP": VLSI_TOP,
             "INPUT_CONFS": " ".join(VLSI_INPUT_CONFS),
             "generated_src_name": "generated-src-chia-ppa"}

# make syn stops right after the syn_map reports: syn_opt, without retiming,
# spends hours on a path it cannot fix (norm_unit -> acc_scale_unit) and did
# not finish in 6 h, while syn_map is done in ~3.5 h, before the bitstream.
# genus_report_map is the hook gemmini-sky130-sram22's example-vlsi-sky130.patch
# adds after syn_map; it must be registered after genus_syn_opt there so it
# runs first (each post-insertion hook goes directly after its target).
SYN_STOP_AFTER_STEP = "genus_report_map"
SYN_MAKE_VARS = {**MAKE_VARS, "HAMMER_EXTRA_ARGS": f"--stop_after_step {SYN_STOP_AFTER_STEP}"}

# Report sets by the step that wrote them, most complete first; result()
# uses the first one it finds. final_* come from a full run (generate_reports;
# final_power.rpt from the patch's genus_extra_reports hook), map_* / generic_*
# from the patch's per-stage hooks. Power exists only once the design is mapped.
_RPT = "syn-rundir/reports"
REPORT_SETS = [
    ("final", f"{_RPT}/final_area.rpt",
     re.compile(rf"^{_RPT}/final_time_.*\.setup_view\.rpt$"), f"{_RPT}/final_power.rpt"),
    ("syn_map", f"{_RPT}/map_area.rpt",
     re.compile(rf"^{_RPT}/map_timing\.rpt$"), f"{_RPT}/map_power.rpt"),
    ("syn_generic", f"{_RPT}/generic_area.rpt",
     re.compile(rf"^{_RPT}/generic_timing\.rpt$"), None),
]
# Per-file cap for collect. The untrimmed ChipTop area report (~7.4k
# hierarchical instances) and 50-path timing report are estimated at ~1-2MB
# each, so 2MB risks silently skipping the files we parse.
REPORT_MAX_BYTES = 8_000_000


# ---------------------------------------------------------------------------
# Report parsers, ported from chia-gemmini-braggnn examples
# (common/common_nodes.py: _parse_area_from_final_area_rpt,
#  timing_opt/db.py: parse_worst_slack).
# ---------------------------------------------------------------------------

def _parse_number_with_commas(s: str) -> float:
    """Parse a number string that may contain commas as thousands separators."""
    return float(s.replace(",", ""))


def _parse_area_from_final_area_rpt(content: str) -> float | None:
    """Parse Total Area from Genus final_area.rpt table format.

    Returns the Total Area value (last numeric column on the first data line,
    i.e. the top module). Total Area = Cell Area + Net Area, which includes
    the estimated routing overhead from wire-load models.
    """
    # Match the data line after the dashes separator: instance name followed by numbers
    data_re = re.compile(
        r'^(\S+)\s+'              # Instance name
        r'(?:\S+\s+)?'           # Optional Module name
        r'(\d[\d,]*)\s+'         # Cell Count
        r'([\d,]+\.?\d*)\s+'     # Cell Area
        r'([\d,]+\.?\d*)\s+'     # Net Area
        r'([\d,]+\.?\d*)',        # Total Area
        re.MULTILINE,
    )
    match = data_re.search(content)
    if match:
        try:
            return _parse_number_with_commas(match.group(5))  # Total Area column
        except ValueError:
            pass
    return None


_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
# Genus per-path header, e.g. "Path 1: VIOLATED (-12054 ps) Setup Check ..."
# Path 1 is the worst path by construction (Genus sorts paths by slack).
_PATH_HEAD_RE = re.compile(
    r"^Path\s+\d+:\s+(VIOLATED|MET)\s+\(\s*([-+]?\d+(?:\.\d+)?)\s*(ps|ns)\s*\)",
    re.IGNORECASE,
)


def parse_worst_slack(report_text: str) -> tuple[float | None, int | None, str | None]:
    """Best-effort extract (slack_ns, met, raw_line) from a Genus timing report."""
    for line in report_text.splitlines():
        m = _PATH_HEAD_RE.match(line.lstrip())
        if m:
            marker = m.group(1).upper()
            value = float(m.group(2))
            unit = m.group(3).lower()
            slack_ns = value / 1000.0 if unit == "ps" else value
            return slack_ns, (1 if marker == "MET" else 0), line.strip()
    for line in report_text.splitlines():
        low = line.lower()
        if "slack" in low and ("(violated)" in low or "(met)" in low):
            stripped = line.strip()
            met = 1 if "(met)" in low else 0
            m = _FLOAT_RE.search(stripped)
            ns = float(m.group()) if m else None
            return ns, met, stripped
    return None, None, None


_POWER_UNIT_RE = re.compile(r"^Power Unit:\s*(\S+)", re.MULTILINE)
_POWER_UNIT_SCALE = {"W": 1.0, "mW": 1e-3, "uW": 1e-6, "nW": 1e-9, "pW": 1e-12}


def parse_total_power(content: str) -> float | None:
    """Total power (W) of the top instance (Lvl 0) from a Genus
    ``report_power -by_hierarchy`` table, or None if it can't be found."""
    m = _POWER_UNIT_RE.search(content)
    scale = _POWER_UNIT_SCALE.get(m.group(1), None) if m else 1.0
    if scale is None:
        return None
    total_col = lvl_col = None
    for line in content.splitlines():
        tokens = line.split()
        if total_col is None:
            if "Total" in tokens and "Lvl" in tokens:
                total_col, lvl_col = tokens.index("Total"), tokens.index("Lvl")
            continue
        if len(tokens) > max(total_col, lvl_col) and tokens[lvl_col] == "0":
            try:
                return float(tokens[total_col]) * scale
            except ValueError:
                return None
    return None


def parse_reports(files: dict[str, str]):
    """(report_stage, area_um2, slack_ns, met, power_w) from the first
    REPORT_SETS entry that has an area or timing report among *files*
    (relpath -> text); all None if there is none."""
    for report_stage, area_rel, timing_re, power_rel in REPORT_SETS:
        area_rpt = files.get(area_rel, "")
        timing_rpt = next((text for rel, text in sorted(files.items())
                           if timing_re.match(rel)), "")
        if not area_rpt and not timing_rpt:
            continue
        area = _parse_area_from_final_area_rpt(area_rpt) if area_rpt else None
        slack_ns, met, _line = parse_worst_slack(timing_rpt)
        power_rpt = files.get(power_rel, "") if power_rel else ""
        power = parse_total_power(power_rpt) if power_rpt else None
        return report_stage, area, slack_ns, met, power
    return None, None, None, None, None


@dataclass
class PpaResult:
    success: bool
    stage: str                # "buildfile" | "syn" | "collect" | "cancelled"
    obj_dir: str               # on the vlsi worker (NFS, under VLSI_OBJ_DIR_ROOT)
    total_area_um2: float | None = None   # ChipTop Total Area (cell + net), final_area.rpt
    worst_slack_ns: float | None = None   # path 1 of final_time_*.setup_view.rpt
    slack_met: bool | None = None
    total_power_w: float | None = None    # ChipTop Total, final_power.rpt (vectorless)
    report_stage: str | None = None       # which REPORT_SETS entry the numbers came from
    reports: dict[str, str] = field(default_factory=dict)
    stderr_tail: str = ""

    def summary(self) -> str:
        """One-paragraph PPA summary for LLM feedback ("n/a" for unparsed fields)."""
        if self.total_area_um2 is not None:
            area = f"{self.total_area_um2:,.0f} um^2 ({self.total_area_um2 / 1e6:.3f} mm^2)"
        else:
            area = "n/a"
        if self.worst_slack_ns is not None:
            slack = f"{self.worst_slack_ns:.3f} ns ({'MET' if self.slack_met else 'VIOLATED'})"
        else:
            slack = "n/a"
        if self.total_power_w is not None:
            power = (f"{self.total_power_w * 1e3:.1f} mW (vectorless estimate with "
                     "default switching activity, not the BraggNN workload)")
        else:
            power = "n/a"
        stage = ""
        if self.report_stage and self.report_stage != "final":
            stage = (f"  measured after Genus {self.report_stage} (synthesis stops there): "
                     "compare with other attempts, which are measured at the same point\n")
        return (f"{stage}"
                f"  total area (ChipTop, cell + net, incl. SRAM macros) = {area}\n"
                f"  worst setup slack = {slack}\n"
                f"  total power = {power}\n"
                f"  note: {SRAM_CAVEAT}")


class PpaJob:
    """One attempt's Hammer PPA run (Genus/sky130/SRAM22+OpenRAM), split so synthesis
    overlaps the FireSim bitstream build and the AlphaEvolve search:

      1. :meth:`elaborate` -- ``make buildfile``, synchronous. It elaborates
         the design, so it is also the loop's check that the LLM's Chisel
         edit builds; run it BEFORE the bitstream build starts, since both
         run sbt against the same (NFS) chipyard checkout.
      2. :meth:`start_syn` -- ``make syn`` in the background. It reads the
         RTL that step 1 generated (vlsi/generated-src) and runs no sbt, but
         it is not isolated from later Chisel edits -- collect with
         :meth:`result` before the LLM's next edit.
      3. :meth:`result` (blocks for syn, collects, parses) or :meth:`cancel`.

    Everything lands in the run dir's ``hammer_ppa_attempt<N>/``: make logs,
    the collected reports/logs mirrored at their obj_dir relpaths, and a
    summary.json (incl. obj_dir) written however the job ends. Apart from
    elaborate()'s return value, nothing here raises -- PPA is best-effort.
    """

    def __init__(self, dump: Dumper, attempt: int, config: str = BUILD_CONFIG):
        self.attempt = attempt
        self.config = config
        self.obj_dir = f"{VLSI_OBJ_DIR_ROOT}/attempt{attempt}-{uuid.uuid4().hex[:8]}"
        self.out = dump.dir(f"hammer_ppa_attempt{attempt}")
        self._node: ChipyardHammerNode | None = None
        self._syn_ref = None
        self._done: PpaResult | None = None

    def _make(self, target: str, timeout_seconds: int):
        return self._node.make.chia_remote(
            CHIPYARD_PATH, target, config=self.config, obj_dir=self.obj_dir,
            make_vars=SYN_MAKE_VARS if target == "syn" else MAKE_VARS,
            timeout_seconds=timeout_seconds,
        )

    def _finish(self, result: PpaResult, skipped: dict[str, int] | None = None) -> PpaResult:
        """Release the placement group and write summary.json (once)."""
        if self._node is not None:
            self._node.close()
            self._node = None
        summary = {k: v for k, v in asdict(result).items() if k != "reports"}
        summary.update(config=self.config, skipped=skipped or {})
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2))
        self._done = result
        return result

    def elaborate(self):
        """Run ``make buildfile``. Returns its ChipyardHammerResult -- check
        ``.success``: False means the design did not elaborate, i.e. the
        Chisel edit is broken -- or None if the vlsi worker could not be used
        at all (infrastructure; the loop stops, see _run_hw_flow)."""
        logger.info("Hammer PPA: CONFIG=%s OBJ_DIR=%s (attempt %d)",
                    self.config, self.obj_dir, self.attempt)
        try:
            # Wait for the PG ourselves so a missing/busy vlsi worker times
            # out instead of hanging the loop, and the pending PG is removed.
            self._node = ChipyardHammerNode(wait_for_pg=False)
            ray.get(self._node.placement_group.ready(),
                    timeout=VLSI_PG_READY_TIMEOUT_SECONDS)
            bf = get(self._make("buildfile", VLSI_BUILDFILE_TIMEOUT_SECONDS))
        except Exception as e:  # noqa: BLE001 -- infra failure: reported as None
            logger.warning("Hammer PPA unavailable (attempt %d): %s", self.attempt, e)
            self._finish(PpaResult(False, "buildfile", self.obj_dir, stderr_tail=str(e)))
            return None
        _write(self.out / "buildfile.stdout.txt", bf.stdout)
        _write(self.out / "buildfile.stderr.txt", bf.stderr)
        if not bf.success:
            logger.warning("Hammer buildfile failed (attempt %d, rc=%s)",
                           self.attempt, bf.returncode)
            self._finish(PpaResult(False, "buildfile", self.obj_dir,
                                   stderr_tail=bf.stderr[-2000:]))
        return bf

    def start_syn(self) -> None:
        """Dispatch ``make syn`` without waiting (no-op if elaborate failed)."""
        if self._done is None and self._syn_ref is None:
            self._syn_ref = self._make("syn", VLSI_SYN_TIMEOUT_SECONDS)

    def cancel(self) -> None:
        """Kill a running syn (e.g. the attempt failed, so its PPA is moot)."""
        if self._done is not None:
            return
        if self._syn_ref is not None:
            try:
                chia_cancel(self._syn_ref, force=True)
            except Exception as e:  # noqa: BLE001
                logger.warning("Hammer PPA cancel failed (attempt %d): %s", self.attempt, e)
        self._finish(PpaResult(False, "cancelled", self.obj_dir))

    def result(self) -> PpaResult:
        """Wait for syn, collect and parse its reports."""
        if self._done is not None:
            return self._done
        if self._syn_ref is None:
            return self._finish(PpaResult(False, "syn", self.obj_dir,
                                          stderr_tail="syn was never started"))
        stage = "syn"
        try:
            syn = get(self._syn_ref)
            _write(self.out / "syn.stdout.txt", syn.stdout)
            _write(self.out / "syn.stderr.txt", syn.stderr)
            stage = "collect"
            rpts = get(self._node.collect.chia_remote(
                self.obj_dir, ["syn-rundir/reports/**", "syn-rundir/*.log"],
                max_bytes_per_file=REPORT_MAX_BYTES,
            ))
        except Exception as e:  # noqa: BLE001 -- PPA is best-effort, never break the loop
            logger.warning("Hammer PPA failed at %s (attempt %d): %s", stage, self.attempt, e)
            return self._finish(PpaResult(False, stage, self.obj_dir, stderr_tail=str(e)))

        for rel, text in rpts.files.items():
            _write(self.out / rel, text)

        report_stage, area, slack_ns, met, power = parse_reports(rpts.files)

        # Genus errors don't reliably fail `make syn` (the hammer Genus plugin
        # does not check its exit status), so success is judged by the numbers.
        # They are also usable when make syn itself failed or timed out: each
        # report set is written by one step that either completed or left no
        # file (a fresh obj_dir per attempt). Power is optional.
        success = area is not None and slack_ns is not None
        logger.info("Hammer PPA %s (attempt %d, %s reports): %d report(s), area=%s, "
                    "slack=%s, power=%s -> %s", "OK" if success else "FAILED",
                    self.attempt, report_stage, len(rpts.files), area, slack_ns,
                    power, self.out)
        return self._finish(PpaResult(
            success=success, stage="syn", obj_dir=self.obj_dir,
            total_area_um2=area, worst_slack_ns=slack_ns,
            slack_met=None if met is None else bool(met), total_power_w=power,
            report_stage=report_stage, reports=rpts.files,
            stderr_tail="" if syn.success else syn.stderr[-2000:],
        ), rpts.skipped)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", errors="replace") as f:
        f.write(text or "")

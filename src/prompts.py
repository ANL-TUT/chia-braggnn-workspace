from constants import EXO_DIR, PROMPTS_DIR


def _load_prompt(name: str, **subs: str) -> str:
    """Read prompts/<name> and replace ${KEY} placeholders with subs values.

    ``${KEY}`` (rather than str.format's ``{KEY}``) so prompt text can contain
    literal braces (Chisel/Scala snippets) without escaping.
    """
    text = (PROMPTS_DIR / name).read_text()
    for key, val in subs.items():
        text = text.replace("${" + key + "}", val)
    return text


_GEMMINI_TUNE = _load_prompt("gemmini.md")
_DEBUGGER_PREAMBLE = _load_prompt("debugger.md")
_OPTIMIZER_PREAMBLE = _load_prompt("optimizer.md")
# AlphaEvolve problem_description section for runs on an mvout_spad-capable
# bitstream (--mvout-spad).
_MVOUT_SPAD_SECTION = _load_prompt("alphaevolve_mvout_spad.md")
# --rtl loop: what the HW LLM does in each phase (appended to gemmini.md).
_RTL_PHASE = _load_prompt("rtl_phase.md")
_PARAM_PHASE = _load_prompt("param_phase.md")


# ── SW search seed: the shipped Exo program ──────────────────────────
# Helper module imported by the seed (`from gemmini import ...`); written next
# to every candidate before exocc runs. Read-only for the search.
_GEMMINI_PY = (EXO_DIR / "gemmini.py").read_text()
# Also shown to the search, read-only: the *_cpu specs the seed schedules
# (loop nests and their names), and the two reference schedules whose ideas
# problem_description describes.
_REFERENCE_PY = (EXO_DIR / "braggnn_reference.py").read_text()
# Pre-checked conv lowering (lower_conv) a schedule may import and call.
_LOWERING_PY = (EXO_DIR / "braggnn_lowering.py").read_text()
_SCHEDULE_FUSION_PY = (EXO_DIR / "braggnn_schedule_fusion.py").read_text()
_SCHEDULE_LOWLEVEL_PY = (EXO_DIR / "braggnn_schedule_lowlevel.py").read_text()

# The EVOLVE-BLOCK is the scheduling section, from the tile constant through the
# `braggnn_eval = schedule_eval()` binding (schedule_eval is inside so the search
# can reshape the staged weight buffers, which are allocations only there).
# Only `__all__` stays fixed.
_EXO_BASE = (EXO_DIR / "braggnn_schedule.py").read_text()
_EVOLVE_START = _EXO_BASE.index("NLB_ROW_TILE = ")
_EVOLVE_END = _EXO_BASE.index("__all__ = ")
_EXO_SEED = (
    _EXO_BASE[:_EVOLVE_START]
    + "# EVOLVE-BLOCK-START\n"
    + _EXO_BASE[_EVOLVE_START:_EVOLVE_END].rstrip()
    + "\n# EVOLVE-BLOCK-END\n\n\n"
    + _EXO_BASE[_EVOLVE_END:]
)

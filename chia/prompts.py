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


# ── SW search seed: the shipped Exo program ──────────────────────────
# Helper module imported by the seed (`from gemmini import ...`); written next
# to every candidate before exocc runs. Read-only for the search.
_GEMMINI_PY = (EXO_DIR / "gemmini.py").read_text()


def _gemmini_h_excerpt() -> str:
    """The parts of gemmini.h a hand-written instr C string needs: the RoCC
    command codes, the command macros (config, mvin/mvout, the LOOP_WS
    hardware loop) and tiled_matmul_outer, which shows how gemmini.h itself
    drives LOOP_WS (bias rows, A reuse)."""
    lines = (EXO_DIR / "include" / "gemmini.h").read_text().splitlines()

    def index(prefix: str, start: int = 0) -> int:
        return next(i for i in range(start, len(lines)) if lines[i].startswith(prefix))

    codes = lines[index("#define k_CONFIG"):index("#define CONFIG_EX")]
    macros_from = index("#define ROCC_INSTRUCTION_RS1_RS2")
    macros_to = index("#define gemmini_loop_ws_spad", macros_from)
    while lines[macros_to].rstrip().endswith("\\") or lines[macros_to].strip() == "}":
        macros_to += 1
    macros = lines[macros_from:macros_to]
    outer_from = index("static void tiled_matmul_outer")
    outer_to = index("}", outer_from) + 1
    return "\n".join(
        codes + ["", "// ..."] + macros + ["", "// ..."] + lines[outer_from:outer_to]
    )


_GEMMINI_H_EXCERPT = _gemmini_h_excerpt()

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

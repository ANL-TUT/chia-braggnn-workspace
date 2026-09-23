import logging
import time

from chia.base.ChiaFunction import get
from chia.base.llm_call import QueryResult
from chia.base.tools.BashTool import BashTool
from chia.models.opencode import AdditionalModelProvider, OpenCodeLLM

from constants import LLM_SYSTEM_MESSAGE, LLM_TIMEOUT_SECONDS, OPENCODE_MODEL
from prompts import (
    _DEBUGGER_PREAMBLE, _GEMMINI_TUNE, _OPTIMIZER_PREAMBLE, _PARAM_PHASE, _RTL_PHASE,
)

logger = logging.getLogger(__name__)

# Newer Gemini Flash models reject a request whose last turn is the model's
# (HTTP 400), and OpenCode sometimes sends one when it retries internally
# after the model has thought, mostly while Vertex is overloaded
# (anomalyco/opencode#45359, open). chia classifies it as InvalidRequestError
# and does not retry; a fresh prompt is the known workaround, so retry those.
_MODEL_TURN_ERROR = "Requests ending with a model turn are not supported"
MODEL_TURN_RETRIES = 3
MODEL_TURN_BACKOFF_SECONDS = 60


def gemini_vertex_provider(model: str = OPENCODE_MODEL) -> AdditionalModelProvider:
    provider_id, _, model_id = model.partition("/")
    return AdditionalModelProvider(
        id=provider_id or "google-vertex", npm="@ai-sdk/google-vertex",
        name="Google Vertex AI",
        models={model_id: {"limit": {"context": 1000000, "output": 65536}}},
        # Expanded by opencode from the container env (set in cluster.yaml);
        # the job driver has no GOOGLE_CLOUD_PROJECT (.env is not shipped).
        options={"project": "{env:GOOGLE_CLOUD_PROJECT}", "location": "global"},
    )


def make_llm(chipyard_bash: BashTool):
    """Build the implement/debug LLM ONCE, to be reused across the whole loop.

    Reusing a single instance is what lets ``ClaudeCodeLLM``'s
    ``@_session_tracked`` wrapper thread the session automatically: each
    ``get()`` syncs the transcript *and* advances the call counter onto this
    instance, so every ``debug`` call ``--resume``s the ``implement``
    conversation with no manual session bookkeeping. ``AntigravityLLM`` shares
    the same wrapper and result fields, so ``--llm antigravity`` threads one agy
    conversation the same way. (Session persistence for OpenCode is in
    development — each OpenCode call is independent — so reuse is just a
    convenience there; the failure context is re-supplied inline regardless.)
    """
    return OpenCodeLLM(
        model=OPENCODE_MODEL,
        system_message=LLM_SYSTEM_MESSAGE,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
        logging_name="gemmini_tuner",
        additional_providers=[gemini_vertex_provider()],
        # Restrict opencode to ONLY the chipyard_bash MCP tool. Its built-in
        # write/edit/bash tools act on the opencode container's own FS, not
        # the chipyard container — deny all, allow just this MCP server.
        config={"*": "deny", f"{chipyard_bash.name}_*": "allow"},
    )


def _run_llm(llm, prompt: str, chipyard_bash: BashTool) -> QueryResult:
    """Dispatch *prompt* to *llm*'s worker (its backend's creds resource),
    retrying the Gemini "model turn" 400 (see _MODEL_TURN_ERROR) with backoff.
    A retry is a fresh OpenCode session on the same chipyard tree, so edits
    the failed session already made stay and the new one continues from them."""
    resources = {"opencode_creds": 1}
    for attempt in range(MODEL_TURN_RETRIES + 1):
        try:
            return get(
                llm.prompt.options(resources=resources).chia_remote(
                    llm, prompt, [chipyard_bash])
            )
        except Exception as e:  # noqa: BLE001 -- only the model-turn 400 is retried
            if _MODEL_TURN_ERROR not in str(e) or attempt == MODEL_TURN_RETRIES:
                raise
            wait = MODEL_TURN_BACKOFF_SECONDS * 2 ** attempt
            logger.warning(
                "OpenCode hit the Gemini model-turn 400 (try %d/%d); retrying "
                "in %ds", attempt + 1, MODEL_TURN_RETRIES + 1, wait,
            )
            time.sleep(wait)


def implement(llm, chipyard_bash: BashTool) -> QueryResult:
    return _run_llm(llm, _GEMMINI_TUNE, chipyard_bash)


def debug(llm, chipyard_bash: BashTool, feedback: str) -> QueryResult:
    """Feedback call: diagnose and fix *feedback*. For claude this ``--resume``s
    the shared implement session (the reused instance carries the transcript +
    counter); for opencode the full failure context is inline in *feedback*."""
    return _run_llm(llm, f"{_DEBUGGER_PREAMBLE}\n\n{feedback}", chipyard_bash)


def optimize(llm, chipyard_bash: BashTool, feedback: str) -> QueryResult:
    """Continuation call after a *successful* attempt: given the previous
    design's real-hardware FireSim results (cycle count, accuracy) in
    *feedback*, ask for further improvement rather than a bug fix -- unlike
    debug(), whose preamble is framed around diagnosing a failure. Session
    reuse works the same way as debug()."""
    return _run_llm(llm, f"{_OPTIMIZER_PREAMBLE}\n\n{feedback}", chipyard_bash)


def rtl_edit(llm, chipyard_bash: BashTool, context: str) -> QueryResult:
    """--rtl loop, RTL phase: one microarchitecture change. OpenCode calls are
    independent, so every call carries the full task plus *context* (the
    Verilator numbers so far and the result of the previous step)."""
    return _run_llm(llm, f"{_GEMMINI_TUNE}\n\n{_RTL_PHASE}\n\n{context}", chipyard_bash)


def param_tune(llm, chipyard_bash: BashTool, context: str) -> QueryResult:
    """--rtl loop, parameter phase: tune Configs.scala for the accepted RTL."""
    return _run_llm(llm, f"{_GEMMINI_TUNE}\n\n{_PARAM_PHASE}\n\n{context}", chipyard_bash)

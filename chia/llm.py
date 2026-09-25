from chia.base.ChiaFunction import get
from chia.base.llm_call import QueryResult
from chia.base.tools.BashTool import BashTool
from chia.models.opencode import AdditionalModelProvider, OpenCodeLLM

from constants import LLM_SYSTEM_MESSAGE, LLM_TIMEOUT_SECONDS, OPENCODE_MODEL
from prompts import _DEBUGGER_PREAMBLE, _GEMMINI_TUNE, _OPTIMIZER_PREAMBLE


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
    """Dispatch *prompt* to *llm*'s worker (its backend's creds resource)."""
    resources = {"opencode_creds": 1}
    return get(
        llm.prompt.options(resources=resources).chia_remote(llm, prompt, [chipyard_bash])
    )


def _with_brief(preamble: str, feedback: str) -> str:
    """The full task brief (gemmini.md) followed by this round's request.
    Each OpenCode call is a fresh session, so debug/optimize would otherwise
    get neither the file map nor the constraints the preambles refer to."""
    return f"{_GEMMINI_TUNE}\n\n## This round\n\n{preamble}\n\n{feedback}"


def implement(llm, chipyard_bash: BashTool) -> QueryResult:
    return _run_llm(llm, _GEMMINI_TUNE, chipyard_bash)


def debug(llm, chipyard_bash: BashTool, feedback: str) -> QueryResult:
    """Feedback call: diagnose and fix *feedback*. For claude this ``--resume``s
    the shared implement session (the reused instance carries the transcript +
    counter); for opencode the full failure context is inline in *feedback*."""
    return _run_llm(llm, _with_brief(_DEBUGGER_PREAMBLE, feedback), chipyard_bash)


def optimize(llm, chipyard_bash: BashTool, feedback: str) -> QueryResult:
    """Continuation call after a *successful* attempt: given the previous
    design's real-hardware FireSim results (cycle count, accuracy) in
    *feedback*, ask for further improvement rather than a bug fix -- unlike
    debug(), whose preamble is framed around diagnosing a failure. Session
    reuse works the same way as debug()."""
    return _run_llm(llm, _with_brief(_OPTIMIZER_PREAMBLE, feedback), chipyard_bash)

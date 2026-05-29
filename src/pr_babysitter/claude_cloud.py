from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import httpx


log = logging.getLogger(__name__)


# --- Anthropic Managed Agents API constants -------------------------------

ANTHROPIC_API = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
MANAGED_AGENTS_BETA = "managed-agents-2026-04-01"

# Model used by the babysitter agent. Managed Agents resolves "claude-opus-4-8"
# to the standard (non-1M-context) variant by default, which matches the prior
# Claude Cloud contract. The Managed Agents POST /v1/agents schema (see
# https://platform.claude.com/docs/en/managed-agents/agent-setup) only accepts
# {name, model, system, tools, mcp_servers, skills, multiagent, description,
# metadata}. There is no `thinking`, `output_config`, or `effort` field; passing
# any of them returns 400 "Extra inputs are not permitted". The only model-side
# knob is `speed: "fast"` (fast mode), which is the opposite of what we want.
# So we just pass the model name and let Anthropic pick the default behavior.
AGENT_MODEL = "claude-opus-4-8"

# Where the PR's repository is cloned inside the session container. The system
# prompt references this path so the user-message prompts don't have to.
REPO_MOUNT_PATH = "/workspace/repo"


# Backoff applied to ALL `POST /v1/sessions` calls after a 400 from that
# endpoint — the only 400s observed in the wild are Anthropic refusing a new
# session while a per-account concurrent-session quota is full. Set at the
# client level so the throttle covers every (PR, subsystem) trying to spawn
# at once. Cleared as soon as a spawn succeeds.
SPAWN_BACKOFF_SECONDS = 120.0


class SessionSpawnQuotaError(RuntimeError):
    """`POST /v1/sessions` returned 400. Treated as a transient quota/rate-
    limit signal: the caller should leave its respawn guard cleared and try
    again on a future tick, rather than burning a spawn budget or routing to
    a terminal ERROR state. The 400 response body is exposed as `.detail`."""

    def __init__(self, detail: str):
        super().__init__(detail or "session spawn rejected with 400")
        self.detail = detail


# --- JSON envelope parsing -------------------------------------------------


@dataclass
class SessionResult:
    """Status of a managed-agents session. `status` is one of:

    - `running`     — still working, or freshly created and not yet picked up
    - `completed`   — agent finished and emitted the expected JSON envelope
    - `needs_nudge` — agent went idle without emitting an envelope. The session
                      is resumable: per the Managed Agents docs, `idle` means
                      "waiting for input", not terminated. The caller should
                      send a follow-up `user.message` to push it to emit the
                      envelope. Common causes: backend `retries_exhausted`
                      (transient errors used up the retry budget mid-task), or
                      the agent simply forgot the envelope on `end_turn`.
    - `failed`      — session is `terminated`; not resumable.

    `response_text` carries whatever the agent did say (used for transcript
    dumps and envelope extraction). `error` carries the most recent
    session.error message when present. `stop_reason` carries the
    `session.status_idle.stop_reason.type` (e.g. `retries_exhausted`,
    `end_turn`, `requires_action`) for observability.
    """

    status: str
    response_text: str = ""
    error: str = ""
    stop_reason: str = ""
    # Raw API status from `GET /v1/sessions/{id}` (e.g. `idle`, `running`,
    # `rescheduling`, `terminated`). Observability-only — distinct from
    # `status` above, which is our orchestrator-level interpretation. A
    # session can be API-`idle` while we still return `running` (genuine
    # spin-up before any event lands).
    api_status: str = ""
    # `stats.active_seconds` from the session GET — total compute time the
    # agent has accrued. Compare against wall-clock elapsed to detect stuck
    # sessions: a healthy long-running session has active_seconds ≈ elapsed;
    # a corpse has active stalled while wall-clock grows.
    active_seconds: float = 0.0


def extract_envelope(text: str) -> Optional[dict]:
    """Find a JSON object in `text` matching our protocol.

    The babysitter prompts ask Claude to put either:
      {"error": false, "ignore": true,  "reason": "..."}
      {"error": false, "ignore": false}
      {"error": true,  "reason": "..."}
      {"error": false}
    somewhere in the response. We scan all `{` positions, try to JSON-decode
    a complete object starting there via `raw_decode` (handles nested braces
    and braces inside string values, which a regex can't), and return the
    LAST valid envelope - Claude may print intermediate JSON first.
    """
    if not text:
        return None
    # Tolerant of smart-quote " introduced by chat formatting.
    normalized = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    decoder = json.JSONDecoder()
    found: Optional[dict] = None
    i = 0
    n = len(normalized)
    while i < n:
        if normalized[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(normalized, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and "error" in obj:
            found = obj  # keep the last one
        i = end
    return found


# --- Agent / environment templates ----------------------------------------


# Bump when _agent_system_prompt() or any other field of the agent payload
# (model, thinking budget, tools, MCP servers) changes meaningfully. The
# bootstrap path recreates the cached agent whenever the persisted version
# differs, picking up the new payload server-side.
SYSTEM_PROMPT_VERSION = "v9"


def _agent_system_prompt(github_pat: str, circleci_pat: str) -> str:
    return f"""\
You are the PR-babysitter agent. Each user message tells you which subsystem you are acting as (one of `fork_sync`, `merge`, `bugbot`, `greptile`, `veria`, `cicd`) and which PR to operate on. The rules below apply to every session.

REPO. The PR's repository is cloned at `{REPO_MOUNT_PATH}`. Before reading code, `cd {REPO_MOUNT_PATH}`, fetch, and check out the PR's HEAD branch. Use the GitHub PAT below when pushing back.

CREDENTIALS. You may use these PATs for any external calls:
- GitHub PAT: {github_pat}
- CircleCI PAT: {circleci_pat}

SCOPE. Only address the concerns raised by your assigned subsystem. Never fix concerns raised by any of the other subsystems (fork_sync, merge, bugbot, greptile, veria, cicd) in the same session, even if you notice them. `fork_sync` and `merge` are plumbing-only — do not write new code in those sessions beyond what's needed to resolve a merge.

COMMITS. For any git commits, set the author/committer to `mateo-berri <277851410+mateo-berri@users.noreply.github.com>`. Do NOT identify yourself as Claude in the author or committer fields, and do NOT add a `Co-authored-by: Claude ...` (or any other Claude attribution) trailer. The commit metadata and message must contain no reference to Claude or to this being AI-generated. Commit directly to the PR's HEAD branch - do not create a new branch (in particular, no `claude/*` branch).

CODE STYLE. Do not write comments unless absolutely necessary to explain some extremely complex business logic (which ideally should not be written anyway). Code comments are a DRY violation: you must update logic in two locations to change the code, and "hard to change" is literally the definition of tech debt. Aim instead for code that is intuitive to the reader, easy to maintain, and high performance.

REBUTTING INVALID CONCERNS. Whenever your envelope is `{{"error": false, "ignore": true, "reason": "..."}}`, post a direct reply to every unresolved comment or review thread the bot opened on HEAD that you are rebutting — one reply per thread, each explaining concretely why that specific finding is invalid. Do not consolidate into a single standalone PR comment, and do not skip threads with the assumption that a generic rebuttal elsewhere covers them. Do this before emitting the envelope. EXCEPTION: if a thread already contains a reply authored by `mateo-berri` (the commit/comment identity you use), that thread is already rebutted — do NOT post another reply on it. Re-rebutting the same thread spams the bot and risks looping.

OUTPUT. Always end your response with the JSON envelope specified in the user message. That envelope is what the orchestrator parses to decide what to do next.
"""


def _agent_payload(github_pat: str, circleci_pat: str) -> dict:
    return {
        "name": f"PR Babysitter ({SYSTEM_PROMPT_VERSION})",
        "model": AGENT_MODEL,
        "system": _agent_system_prompt(github_pat, circleci_pat),
        "mcp_servers": [
            {
                "type": "url",
                "name": "github",
                "url": "https://api.githubcopilot.com/mcp/",
            },
        ],
        "tools": [
            {"type": "agent_toolset_20260401"},
            {"type": "mcp_toolset", "mcp_server_name": "github"},
        ],
        "metadata": {"system_prompt_version": SYSTEM_PROMPT_VERSION},
    }


def _environment_payload() -> dict:
    return {
        "name": "pr-babysitter-env",
        "config": {"type": "cloud", "networking": {"type": "unrestricted"}},
    }


# --- Client ----------------------------------------------------------------


class ClaudeCloudClient:
    """Thin client over Anthropic's Managed Agents API.

    One agent + one environment are reused across PRs and subsystems. Each
    babysitter spawn creates a fresh session mounted on the PR's repository
    and sends the prompt as a `user.message`. Polling maps managed-agents
    statuses (`idle`/`running`/`rescheduling`/`terminated`) onto the legacy
    `queued`/`running`/`completed`/`failed` contract the babysitter expects.
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str = "",
        agent_version: str = "",
        environment_id: str = "",
        github_pat: str = "",
        circleci_pat: str = "",
        on_ids_updated: Optional[Callable[[str, str, str], None]] = None,
    ):
        self._api_key = api_key
        self._agent_id = agent_id
        self._agent_version = agent_version
        self._environment_id = environment_id
        self._gh_pat = github_pat
        self._circleci_pat = circleci_pat
        self._on_ids_updated = on_ids_updated
        self._bootstrap_lock = asyncio.Lock()
        self._spawn_blocked_until: float = 0.0
        self._client = httpx.AsyncClient(
            base_url=ANTHROPIC_API,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "anthropic-beta": MANAGED_AGENTS_BETA,
                "content-type": "application/json",
                "User-Agent": "pr-babysitter",
            },
            timeout=60.0,
        ) if api_key else None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    # ----- bootstrap --------------------------------------------------

    async def ensure_bootstrapped(self) -> None:
        """Create the babysitter agent + environment if not already cached.
        Recreates the agent if the cached system-prompt version is stale.
        Safe to call repeatedly and from multiple concurrent tasks; only the
        first call hits the API."""
        if not self._client:
            raise RuntimeError("ClaudeCloudClient has no API key")
        agent_ok = (
            self._agent_id and self._agent_version == SYSTEM_PROMPT_VERSION
        )
        if agent_ok and self._environment_id:
            return
        async with self._bootstrap_lock:
            changed = False
            if not self._agent_id or self._agent_version != SYSTEM_PROMPT_VERSION:
                stale_agent_id = self._agent_id
                r = await self._client.post(
                    "/v1/agents",
                    json=_agent_payload(self._gh_pat, self._circleci_pat),
                )
                r.raise_for_status()
                self._agent_id = r.json().get("id", "")
                self._agent_version = SYSTEM_PROMPT_VERSION
                if not self._agent_id:
                    raise RuntimeError("agent create returned empty id")
                changed = True
                # Archive the prior agent. Each agent's system prompt has
                # the GitHub + CircleCI PATs baked in plaintext, so leaving
                # a dangling agent behind on every rotation/version-bump
                # accumulates copies of (potentially rotated/revoked)
                # credentials on Anthropic's servers. Best-effort: archive
                # failure must not block bootstrap.
                if stale_agent_id:
                    try:
                        ar = await self._client.post(
                            f"/v1/agents/{stale_agent_id}/archive"
                        )
                        ar.raise_for_status()
                        log.info(
                            "archived prior agent %s after creating %s",
                            stale_agent_id, self._agent_id,
                        )
                    except Exception as e:
                        log.warning(
                            "failed to archive prior agent %s: %s",
                            stale_agent_id, e,
                        )
            if not self._environment_id:
                r = await self._client.post(
                    "/v1/environments", json=_environment_payload()
                )
                r.raise_for_status()
                self._environment_id = r.json().get("id", "")
                if not self._environment_id:
                    raise RuntimeError("environment create returned empty id")
                changed = True
            if changed and self._on_ids_updated:
                self._on_ids_updated(
                    self._agent_id, self._agent_version, self._environment_id
                )

    # ----- session lifecycle ------------------------------------------

    async def spawn(self, prompt: str, repo: str) -> str:
        """Create a session mounting `repo` (e.g. "BerriAI/litellm"), send
        `prompt` as a `user.message`, and return the session id."""
        if not self._client:
            raise RuntimeError("ClaudeCloudClient has no API key")
        now = time.time()
        if now < self._spawn_blocked_until:
            remaining = self._spawn_blocked_until - now
            raise SessionSpawnQuotaError(
                f"spawn throttled for {remaining:.0f}s after prior 400"
            )
        await self.ensure_bootstrapped()
        repo_url = f"https://github.com/{repo}"
        create = await self._client.post(
            "/v1/sessions",
            json={
                "agent": self._agent_id,
                "environment_id": self._environment_id,
                "title": f"pr-babysitter: {repo}",
                "resources": [
                    {
                        "type": "github_repository",
                        "url": repo_url,
                        "mount_path": REPO_MOUNT_PATH,
                        "authorization_token": self._gh_pat,
                    },
                ],
            },
        )
        if create.status_code == 400:
            # Surface what the server actually said — `raise_for_status()`
            # would otherwise discard the body. The only 400 we've seen here
            # is the concurrent-session quota; treat any 400 as transient
            # and back off rather than routing the caller to a terminal
            # ERROR state.
            body = (create.text or "")[:500]
            self._spawn_blocked_until = time.time() + SPAWN_BACKOFF_SECONDS
            log.warning(
                "POST /v1/sessions returned 400; backing off %.0fs. body=%s",
                SPAWN_BACKOFF_SECONDS, body,
            )
            raise SessionSpawnQuotaError(body)
        create.raise_for_status()
        self._spawn_blocked_until = 0.0
        session_id = create.json().get("id", "")
        if not session_id:
            return ""
        await self.send_user_message(session_id, prompt)
        return session_id

    async def send_user_message(self, session_id: str, prompt: str) -> None:
        """Send a `user.message` event to an existing session. Used both for
        the initial prompt on spawn and for resuming a session that went idle
        without emitting our JSON envelope (the backend's `idle` status is
        explicitly resumable; sending another user.message transitions the
        session back to `running` with the full history intact)."""
        if not self._client:
            raise RuntimeError("ClaudeCloudClient has no API key")
        r = await self._client.post(
            f"/v1/sessions/{session_id}/events",
            json={
                "events": [
                    {
                        "type": "user.message",
                        "content": [{"type": "text", "text": prompt}],
                    },
                ],
            },
        )
        r.raise_for_status()

    async def get(self, session_id: str) -> SessionResult:
        if not self._client:
            raise RuntimeError("ClaudeCloudClient has no API key")
        sess_r = await self._client.get(f"/v1/sessions/{session_id}")
        sess_r.raise_for_status()
        body = sess_r.json()
        status = body.get("status", "")
        stats = body.get("stats") or {}
        active_seconds = float(stats.get("active_seconds") or 0.0)
        log.debug(
            "session %s api_status=%s active_seconds=%.1f",
            session_id[:12], status, active_seconds,
        )

        if status in ("running", "rescheduling"):
            return SessionResult(
                status="running",
                api_status=status,
                active_seconds=active_seconds,
            )
        if status == "terminated":
            err = await self._fetch_error_text(session_id)
            text = await self._fetch_agent_text(session_id)
            return SessionResult(
                status="failed",
                response_text=text,
                error=err or "session terminated",
                api_status=status,
                active_seconds=active_seconds,
            )
        if status == "idle":
            # `idle` is ambiguous: the session has not yet picked up our
            # user.message, the agent finished cleanly, or the backend
            # exhausted its retry budget mid-task after a `session.error`.
            # Critically, `idle` is NOT terminal — per the Managed Agents docs
            # the session is resumable by sending another user.message.
            #
            # Discriminate via the presence of a `session.status_idle` event
            # (carries a `stop_reason`), NOT via agent text. Backend retries
            # can exhaust before the agent writes a single `agent.message`,
            # so "no text" is not the same as "still spinning up" — without
            # this check we'd poll a corpse forever (saw this on PR #28324
            # which sat in status=running for 2h+ while the underlying
            # session was idle with stop_reason=retries_exhausted).
            stop_reason = await self._fetch_latest_stop_reason(session_id)
            if not stop_reason:
                # No idle event yet — session genuinely still spinning up.
                return SessionResult(
                    status="running",
                    api_status=status,
                    active_seconds=active_seconds,
                )
            text = await self._fetch_agent_text(session_id)
            log.debug(
                "session %s idle: stop_reason=%s text_len=%d active=%.1fs",
                session_id[:12], stop_reason, len(text), active_seconds,
            )
            if extract_envelope(text) is not None:
                # Clean finish with envelope — trust it even if there were
                # transient session.errors earlier that got rescheduled.
                return SessionResult(
                    status="completed",
                    response_text=text,
                    api_status=status,
                    active_seconds=active_seconds,
                )
            # Session has ended a turn but didn't produce an envelope. Could be:
            #   (a) backend `retries_exhausted` mid-task — session.error
            #       with retry_status=exhausted, stop_reason=retries_exhausted
            #   (b) agent finished its turn and just forgot the envelope —
            #       stop_reason=end_turn, no terminal session.error
            #   (c) backend killed the session before the agent wrote anything
            #       — same stop_reason as (a) but text is empty
            # All three are resumable via a follow-up user.message; return
            # needs_nudge and let the caller decide.
            err = await self._fetch_error_text(session_id)
            return SessionResult(
                status="needs_nudge",
                response_text=text,
                error=err,
                stop_reason=stop_reason,
                api_status=status,
                active_seconds=active_seconds,
            )
        # Unknown status - treat as still running rather than spuriously
        # erroring out.
        return SessionResult(
            status="running",
            api_status=status,
            active_seconds=active_seconds,
        )

    async def count_events(self, session_id: str) -> dict[str, int]:
        """Bucket-count events by type for observability — used by the
        heartbeat to surface 'agent did N tool calls / M errors so far' on
        long-running sessions. One HTTP call; pages up to ~3 pages then
        gives up to keep the heartbeat cheap (counts are signals, not
        precise totals). Returned dict has keys for the few event types we
        care about and a `total` field; missing keys default to 0.
        """
        assert self._client is not None
        # `total` is a synthetic accumulator, not a real event type. Keep it
        # separate from `type_counts` so an event whose `type` field equals
        # "total" can't double-increment the accumulator.
        type_counts = {"agent.message": 0, "agent.tool_use": 0, "session.error": 0}
        total = 0
        path = f"/v1/sessions/{session_id}/events"
        params: list[tuple[str, str]] = [("limit", "200")]
        pages = 0
        try:
            while pages < 3:
                r = await self._client.get(path, params=params)
                r.raise_for_status()
                data = r.json()
                for ev in data.get("data", []) or []:
                    total += 1
                    t = ev.get("type", "")
                    if t in type_counts:
                        type_counts[t] += 1
                after = data.get("last_id") or data.get("next_cursor")
                if not data.get("has_more") or not after:
                    break
                params = [("limit", "200"), ("after_id", after)]
                pages += 1
        except Exception:
            # Heartbeat observability — never let a failed count call
            # bubble up and break the poll loop. Return whatever we got.
            pass
        return {**type_counts, "total": total}

    # ----- internal: event scraping -----------------------------------

    async def _fetch_agent_text(self, session_id: str) -> str:
        """Concatenate text from all agent.message events in chronological
        order so extract_envelope sees the agent's full transcript."""
        chunks: list[str] = []
        async for ev in self._iter_events(session_id, "agent.message"):
            for block in ev.get("content", []) or []:
                if block.get("type") == "text":
                    chunks.append(block.get("text") or "")
        return "\n".join(chunks)

    async def _fetch_error_text(self, session_id: str) -> str:
        # Returns the *last* session.error message — the terminal one. Earlier
        # session.errors are typically followed by a `session.status_rescheduled`
        # event and recovery; only the trailing one (with no recovery after it)
        # tells us why the session actually died.
        latest = ""
        try:
            async for ev in self._iter_events(session_id, "session.error"):
                err = ev.get("error") or {}
                msg = err.get("message")
                if msg:
                    latest = msg
        except Exception:
            pass
        return latest

    async def _fetch_latest_stop_reason(self, session_id: str) -> str:
        # session.status_idle events carry a stop_reason. The latest one tells
        # us *why* the agent is currently idle: `end_turn` (clean finish),
        # `retries_exhausted` (backend ate the retry budget), or
        # `requires_action` (waiting on tool confirmation we never approve).
        latest = ""
        try:
            async for ev in self._iter_events(session_id, "session.status_idle"):
                reason = (ev.get("stop_reason") or {}).get("type")
                if reason:
                    latest = reason
        except Exception:
            pass
        return latest

    async def _iter_events(self, session_id: str, event_type: str):
        """Iterate session events of one type across all pages."""
        assert self._client is not None
        path = f"/v1/sessions/{session_id}/events"
        params: list[tuple[str, str]] = [("types[]", event_type)]
        while True:
            r = await self._client.get(path, params=params)
            r.raise_for_status()
            data = r.json()
            for ev in data.get("data", []) or []:
                yield ev
            after = data.get("last_id") or data.get("next_cursor")
            has_more = bool(data.get("has_more"))
            if not has_more or not after:
                break
            params = [("types[]", event_type), ("after_id", after)]

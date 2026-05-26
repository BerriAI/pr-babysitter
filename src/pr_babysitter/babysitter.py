from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime
from typing import Callable, Optional

from .claude_cloud import ClaudeCloudClient, SessionSpawnQuotaError, extract_envelope
from .config import CONFIG_DIR, _write_text_secure
from .github_api import Comment, GitHubClient
from .state import (
    GREPTILE_LOGINS,
    PRState,
    SubState,
    Subsystem,
)


# When a claude session ends unexpectedly (failed, or completed without the
# expected JSON envelope), dump the full agent transcript here so it's
# debuggable after the fact without hitting the Anthropic API by hand. Tail
# is also logged inline next to the WARNING. Files are named by session id.
_TRANSCRIPT_DIR = CONFIG_DIR / "transcripts"
_TRANSCRIPT_TAIL_CHARS = 800


def _dump_transcript(session_id: str, text: str) -> Optional[str]:
    """Write `text` to ~/.pr-babysitter/transcripts/<session_id>.txt and return
    the path (or None on failure or empty text).

    Transcripts can echo the PATs baked into the agent system prompt, so we
    tighten the dir/file modes to user-only (matching config.json) rather
    than relying on the default umask on shared systems.
    """
    if not text:
        return None
    try:
        _TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(_TRANSCRIPT_DIR, 0o700)
        except OSError:
            pass
        path = _TRANSCRIPT_DIR / f"{session_id}.txt"
        # Transcripts can echo the PATs from the agent system prompt, so
        # open with 0o600 from the start rather than relying on a later
        # chmod (which leaves a TOCTOU window where the default umask
        # may have created the file world-readable on shared systems).
        _write_text_secure(path, text)
        return str(path)
    except Exception:
        return None


log = logging.getLogger(__name__)


# Heartbeat: while a claude session is still in `running`/`queued`, emit one
# log line per session every ~2 minutes so a stuck session is visible in logs
# without having to grep across spawn+envelope pairs. Independent of poll
# cadence so changing poll_interval doesn't change heartbeat rate.
_HEARTBEAT_INTERVAL_SECONDS = 120.0

# A normal claude session for any of our subsystems wraps up in a few
# minutes. Past this threshold something is almost certainly wrong —
# escalate the heartbeat from INFO to WARNING so it surfaces in alerts
# and isn't lost in the routine-poll firehose. Tuned empirically: cicd
# sessions topping out near 10m, others near 5m, so 30m is well beyond
# any healthy run.
_HEARTBEAT_STUCK_SECONDS = 1800.0


def _format_elapsed(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m, rem = divmod(s, 60)
    if m < 60:
        return f"{m}m{rem:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# Per-subsystem user messages. Each one declares its `subsystem` and `pr_url`,
# then carries only the subsystem-specific decision tree. Everything else
# (commit identity, no-comments, PATs, "stay in your lane", rebut-on-invalid)
# lives in the agent's system prompt; see claude_cloud._AGENT_SYSTEM_PROMPT.

BUGBOT_PROMPT = """\
subsystem = bugbot
pr_url = {pr_url}

Are there any legit bugbot concerns on this PR that aren't fixed yet? Read the code on the PR's HEAD comprehensively, then decide.

- If there are no legit concerns, respond with: {{"error": false, "ignore": true, "reason": "..."}}. After posting your per-thread rebuttal replies (per the system prompt), resolve each unresolved Cursor Bugbot review thread on HEAD via the GraphQL `resolveReviewThread` mutation, then post one top-level issue comment with exactly the body `bugbot run` so Cursor re-reviews HEAD.
- If there are legit concerns, fix them on the PR's branch and respond with: {{"error": false, "ignore": false}}
- If you could not complete the investigation (backend retries exhausted, session was killed mid-task, prior turns in this session don't add up to a complete read), respond with: {{"error": true, "reason": "interrupted: <what got done, what's missing>"}}. Do NOT pick `ignore: true` to escape — that tells the orchestrator everything is fine when it isn't.
"""

GREPTILE_PROMPT = """\
subsystem = greptile
pr_url = {pr_url}

Your goal is to keep fixing greptile's concerns on this PR until greptile's main summary comment (the one carrying the Confidence Score: X/5) contains the exact phrase `No files require special attention.`. That phrase is the ideal terminal state. Read the code on the PR's HEAD comprehensively, then decide.

- If greptile's latest summary contains `No files require special attention.`, respond with: {{"error": false, "ignore": true, "reason": "greptile summary clean"}}
- If the ideal phrase is absent but the summary also does not list any concerns (for whatever reason — greptile didn't re-summarize after the last fix, the summary is partially rendered, etc.), it's still OK to respond with: {{"error": false, "ignore": true, "reason": "..."}}. State explicitly in `reason` that the phrase is missing but no concerns are listed.
- If there are legit concerns left, fix them on the PR's branch and respond with: {{"error": false, "ignore": false}}
- If the summary lists concerns but every remaining greptile finding is invalid — already fixed on HEAD, false positive, or implementing the fix would make the code neutral-or-worse rather than better — respond with: {{"error": false, "ignore": true, "reason": "..."}}. In this case the `reason` MUST enumerate each remaining greptile finding by comment id (or by an unambiguous snippet of its body) and give a concrete per-finding justification. Lumped justifications like "all minor" or "fail-safe" are not acceptable — every finding must be addressed by name.
- If you could not complete the investigation (backend retries exhausted, session was killed mid-task, prior turns in this session don't add up to a real read of greptile's summary and the code), respond with: {{"error": true, "reason": "interrupted: <what got done, what's missing>"}}. Do NOT pick `ignore: true` to escape — that tells the orchestrator the review is clean when it hasn't actually been reviewed.
"""

VERIA_PROMPT = """\
subsystem = veria
pr_url = {pr_url}

Are there any legit veria concerns on this PR that aren't fixed yet? Read the code on the PR's HEAD comprehensively, then decide.

- If there are no legit concerns, respond with: {{"error": false, "ignore": true, "reason": "..."}}
- If there are legit concerns but you cannot fix them, respond with: {{"error": true, "reason": "..."}}
- If there are legit concerns and you fix them, respond with: {{"error": false, "ignore": false}}
- If you could not complete the investigation (backend retries exhausted, session was killed mid-task, prior turns in this session don't add up to a complete read of veria's findings and the code), respond with: {{"error": true, "reason": "interrupted: <what got done, what's missing>"}}. Do NOT pick `ignore: true` to escape — that tells the orchestrator there are no findings when you never actually checked.
"""

FORK_SYNC_PROMPT = """\
subsystem = fork_sync
pr_url = {pr_url}

This PR is an internal copy of an upstream fork PR. New commits have landed on the upstream branch since the copy branch was last synced — bring the copy branch up to date by merging the upstream branch in. Resolve any conflicts.

UPSTREAM (source of new commits):
- repo: `{origin_head_repo}`
- branch: `{origin_head_ref}`
- head sha at spawn time: `{origin_head_sha}`

COPY (where the merge result must be pushed):
- repo: `BerriAI/litellm`
- branch: `{copy_branch}`
- current sha: `{copy_sha}`

Steps:
- `cd` into the cloned BerriAI/litellm checkout, fetch, and check out `{copy_branch}`.
- Add the upstream fork as a remote (e.g. `git remote add upstream https://github.com/{origin_head_repo}.git`) and fetch `{origin_head_ref}`. Use the GitHub PAT in the system prompt if auth is needed.
- Merge `upstream/{origin_head_ref}` into the current `{copy_branch}` checkout. Prefer a merge commit (`git merge --no-ff`) so history is preserved; a fast-forward is fine when there are no fixes on the copy branch.
- If conflicts arise, resolve them. The copy branch may already contain fixes pushed by other pr-babysitter sessions (cicd, merge, bugbot, greptile, veria) — preserve those fixes; do NOT clobber them with the upstream version. When upstream and copy both edit the same lines with semantically different intent, prefer the copy-branch side if it looks like a bugfix or test fix; prefer the upstream side if it looks like the original author's intent the copy is meant to mirror. If you genuinely cannot tell, emit the error envelope below.
- Push the merge commit back to `{copy_branch}` on BerriAI/litellm.

Do NOT write any new code beyond what's needed to resolve merge conflicts. This subsystem is plumbing only.

Respond with `{{"error": false, "ignore": false}}` if and only if you pushed a merge commit (or fast-forward) that brings `{copy_branch}` up to `{origin_head_sha}`. Respond with `{{"error": true, "reason": "..."}}` if the conflicts cannot be safely auto-resolved, or `{{"error": true, "reason": "interrupted: <what got done, what's missing>"}}` if your investigation was killed mid-task and you don't actually know whether the conflicts are safely auto-resolvable. Do NOT push a guess just to emit a clean envelope.
"""

MERGE_CONFLICT_PROMPT = """\
subsystem = merge
pr_url = {pr_url}

This PR has merge conflicts with its base branch. Resolve them before any other automation runs.

IMPORTANT: this PR's base branch is `{base_branch}` (per the PR's GitHub metadata). When you fetch/merge the base, it's `origin/{base_branch}`. Do not assume `main` - use the base branch above exactly.

- Fetch the base branch, merge `origin/{base_branch}` into the PR's HEAD branch, resolve every conflict, and push the merge commit back to the PR branch.
- If the conflicts are genuinely ambiguous and require human judgment (semantic conflicts you cannot safely resolve), do NOT guess - respond with the error envelope below.

Respond with `{{"error": false, "ignore": false}}` if and only if you pushed a resolved merge commit. Respond with `{{"error": true, "reason": "..."}}` if the conflicts cannot be safely auto-resolved, or `{{"error": true, "reason": "interrupted: <what got done, what's missing>"}}` if your investigation was killed mid-task and you don't actually know whether the conflicts are safely auto-resolvable. Do NOT push a guess just to emit a clean envelope.
"""

CICD_PROMPT = """\
subsystem = cicd
pr_url = {pr_url}
attempt = {attempt}

Why did CI/CD fail on this PR's HEAD? Read both the code and the CI/CD logs comprehensively. The `attempt` field above is how many times a CI/CD agent (including you) has been spawned on this exact commit — `1` means this is the first attempt; anything higher means previous attempts already ran and CI/CD is still red. Each attempt runs in a fresh session, so the prior attempts' turns are NOT visible to you here — but the fact that CI/CD is still failing after `attempt - 1` prior attempts means the obvious thing (a fix, a flaky re-run) was likely already tried on this exact commit and didn't stick. Treat that as a signal that the failure is not flaky and not what you'd first assume — investigate further (read the failing logs and the diff carefully before acting) or give up with `{{"error": true, "reason": "..."}}` rather than re-running mechanically.

IMPORTANT: this PR's base branch is `{base_branch}` (per the PR's GitHub metadata). Any reference below to "the base branch" means `origin/{base_branch}`. Do not assume `main` - use the base branch above exactly.

- If it failed for a legit reason: before writing a fresh fix, consider whether the failure was already fixed by someone else on the base branch and this PR is simply out of date. Do `git fetch origin {base_branch}`, then scan commits on `origin/{base_branch}` that landed after this branch diverged. If you can point to a specific commit on the base branch that plausibly addresses this exact failing test/path/symbol, fix the PR by merging `origin/{base_branch}` in and pushing. If you can't tie the failure to a base-branch commit, write a direct fix on the branch as usual. Do NOT merge the base branch speculatively - the goal is "is this PR stale?", not "keep this PR in sync with the base."
- If it failed because of flakiness, surgically re-run only the flaky tests. Do NOT "fix" a flaky test by adding a skip/xfail marker, by deleting the test, or by loosening its assertions until it passes — those are not fixes, they are concealment. Instead, address the root cause head-on: identify the actual race / shared-state leak / time-or-order dependency / network-dependence and fix that. The ONLY acceptable cases for skipping or deleting are when the test exercises behavior that genuinely no longer exists, or when the root cause is provably outside this repo's control (e.g. a third-party service that is intermittently down) AND fixing it is infeasible from inside this PR — and even then, explain that reasoning in the commit message rather than silently muting the test.

IMPORTANT: as soon as you have either pushed a fix OR issued the surgical re-run API call, emit the envelope below and stop. Do NOT wait for the re-run to complete or the new CI/CD status to settle — the babysitter re-evaluates CI/CD on its own schedule and will spawn you again if the failure persists. Long blocking waits inside the session (e.g. `sleep` + polling CircleCI) risk the backend killing the session before you get a chance to emit the envelope.

Respond with `{{"error": false}}` if and only if either you fixed the CI/CD problems, or they were flaky and you re-ran only the flaky tests. Respond with `{{"error": true, "reason": "..."}}` if and only if the CI/CD failure could not be fixed (e.g. OpenAI billing out of money). Respond with `{{"error": true, "reason": "interrupted: <what got done, what's missing>"}}` if your investigation was killed mid-task (backend retries exhausted, internal service error before you could read the failing logs) — do NOT issue a speculative re-run or guess at a fix just to emit a clean envelope.
"""


# Sent to a session that went idle without emitting the JSON envelope our
# original user message asked for. Per the Managed Agents docs, `idle` is
# "agent waiting for input", not terminated — so the full event history
# (system prompt, original task, every tool result) is still in the session's
# context. We just need to remind it to wrap up.
RESUME_PROMPT = """\
Your previous turn ended without emitting the required JSON envelope. There are two reasons that happens, and they require different envelopes:

1. You finished the investigation and just forgot the envelope on end_turn. Your prior tool calls in this session DO add up to a complete read of the subsystem's findings and the code. → Emit the envelope your original task asked for, exactly as specified.

2. Your previous turn was interrupted mid-investigation — backend retries exhausted (stop_reason=retries_exhausted), an internal service error killed your turn before tools finished, or the session was rescheduled before you could read the findings. Your prior tool calls do NOT constitute a complete investigation. → Emit `{"error": true, "reason": "interrupted: <what got done, what's missing>"}` so the orchestrator knows to retry.

Look honestly at your tool-call history in THIS session and pick the right one. Do NOT pick `ignore: true` or `error: false` just to escape — that lies to the orchestrator and routes incomplete work to DONE instead of retry. Do NOT start any new investigation, re-run tools, push code, or post comments. Your reply this turn must be only the envelope, nothing else.
"""


# Grace period after HEAD before we treat "no bugbot check-run" as "bugbot
# won't review this PR" (paused for billing, uninstalled, repo-filtered). Long
# enough to absorb normal Cursor webhook + queue latency; short enough that a
# permanently-skipped bugbot doesn't pin the TUI in UNKNOWN forever.
BUGBOT_NO_CHECK_RUN_GRACE_SECONDS = 300


# Grace period after HEAD during which we hold cicd at "pending" if GitHub's
# combined-status endpoint reports state=pending with zero statuses. That
# combination is genuinely ambiguous: it means either "no service has posted
# a commit status yet" (true pending — e.g. CircleCI webhook race right after
# push) or "this repo only posts check-runs, never legacy statuses" (in which
# case we'd wait forever). After grace, we fall through to check-runs only.
CICD_STATUS_GRACE_SECONDS = 600


# Cap on greptile claude spawns per commit. The initial mention costs one
# spawn; subsequent greptile follow-up comments after each rebuttal cost one
# more. Cap stops a pathological back-and-forth (claude keeps rebutting,
# greptile keeps replying) from looping forever on the same SHA.
GREPTILE_MAX_SPAWNS_PER_COMMIT = 4


# Per-SHA cap on automatic respawns triggered by an `interrupted:` envelope
# (the agent reporting "my prior turn was killed mid-flight by a backend
# retries_exhausted / internal service error"). Tracked in its own
# `interrupted_retry_count` field on `Subsystem` so backend-induced retries
# do NOT eat into greptile's per-commit follow-up budget
# (`GREPTILE_MAX_SPAWNS_PER_COMMIT`). Three lets us absorb a brief Anthropic
# backend incident (saw 2 sessions die within a 10-minute window on
# 2026-05-20) without looping forever if the backend stays sick.
INTERRUPTED_RETRY_CAP = 3


# Number of consecutive `failure` cicd verdicts required before re-spawning
# the cicd agent on the same SHA. Each tick is ~30s, so 3 ticks ≈ 90s — long
# enough for CircleCI to flip the GitHub commit-status from `failure` (stale,
# from the previous run) to `pending` (re-run in progress) after the agent
# issued its surgical re-run API call. Without this gate, the very next tick
# after a successful agent run would still see the stale `failure` and spawn
# another agent immediately. We trust the agent itself to decide when to
# stop trying (it's told the attempt count in the prompt); there is no hard
# cap on the number of spawns per SHA.
CICD_FAILURE_TICKS_BEFORE_RESPAWN = 3


def _login_matches(login: str, candidates: tuple[str, ...]) -> bool:
    lo = (login or "").lower()
    return any(c in lo for c in candidates)


class PRBabysitter:
    """Owns the polling loop and state machine for a single PR."""

    def __init__(
        self,
        pr: PRState,
        github: GitHubClient,
        claude: ClaudeCloudClient,
        poll_interval: float,
        on_change: Callable[[], None],
    ):
        self.pr = pr
        self._github = github
        self._claude = claude
        self._poll_interval = poll_interval
        self._on_change = on_change
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        # caches per tick
        self._check_runs: list[dict] = []  # for HEAD SHA; bugbot/veria/cicd verdicts
        self._issue_comments: list[Comment] = []  # for greptile mention adoption
        self._latest_commit_at: float = 0.0  # HEAD commit timestamp; filters stale mentions
        # PR's `created_at` timestamp. Used by greptile to detect "no commits
        # since the PR was opened" — in that case greptile auto-runs its
        # initial review on a non-draft PR, so an explicit @mention is just
        # noise. Refreshed every tick from `get_pr()`.
        self._pr_opened_at: float = 0.0

    # ----- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name=f"babysit-{self.pr.key}")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run(self) -> None:
        # immediate first tick so the UI updates fast
        while not self._stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("tick failed for %s", self.pr.key)
                self.pr.last_error = f"{type(e).__name__}: {e}"
                self._on_change()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._poll_interval)
            except asyncio.TimeoutError:
                pass

    # ----- core tick ------------------------------------------------------

    async def tick(self) -> None:
        self.pr.last_polled_at = time.time()
        await self._refresh_pr_meta()
        # check-runs feed bugbot/veria/cicd verdicts; issue comments feed
        # greptile's "adopt an existing @greptile mention" path.
        try:
            if self.pr.last_commit_sha:
                self._check_runs = await self._github.list_check_runs(
                    self.pr.repo, self.pr.last_commit_sha
                )
            else:
                self._check_runs = []
            self._issue_comments = await self._github.list_issue_comments(
                self.pr.repo, self.pr.number
            )
        except Exception as e:
            self.pr.last_error = f"github fetch failed: {e}"
            self._on_change()
            return

        # Bring the copy branch up to date with its upstream fork branch
        # before anything else: bots/cicd/merge all reason about the copy
        # branch's HEAD, so running them on a stale copy wastes work (or
        # acts on outdated code). When a sync session is in flight, return
        # early — the copy's HEAD is about to advance and reset everything.
        await self._process_fork_sync()
        if self.pr.fork_sync.state == SubState.CLAUDE_RUNNING:
            self._on_change()
            return

        # Resolve merge conflicts before anything else. If a merge-resolution
        # session is in flight, don't engage bots/CICD on a known-broken tree.
        await self._process_merge_conflicts()
        if self.pr.merge.state == SubState.CLAUDE_RUNNING:
            self._on_change()
            return

        await self._process_bugbot()
        await self._process_greptile()
        await self._process_veria()
        await self._process_cicd()

        self._on_change()

    async def _refresh_pr_meta(self) -> None:
        pr = await self._github.get_pr(self.pr.repo, self.pr.number)
        self.pr.title = pr.get("title", self.pr.title)
        self.pr.html_url = pr.get("html_url", self.pr.html_url)
        self.pr.head_branch = (pr.get("head") or {}).get("ref", self.pr.head_branch)
        self.pr.base_branch = (pr.get("base") or {}).get("ref", self.pr.base_branch)
        new_sha = (pr.get("head") or {}).get("sha", "")
        if new_sha and new_sha != self.pr.last_commit_sha:
            log.info(
                "%s: HEAD changed %s -> %s; resetting all subsystems",
                self.pr.key,
                (self.pr.last_commit_sha or "none")[:12],
                new_sha[:12],
            )
            self.pr.last_commit_sha = new_sha
            self.pr.reset_subsystems()
        # mergeable_state may be "unknown" briefly while GitHub recomputes after
        # a push; the next tick will have a definitive value.
        self.pr.mergeable_state = pr.get("mergeable_state", "") or ""
        self._pr_opened_at = _parse_iso(pr.get("created_at", ""))
        # HEAD commit timestamp — needed by greptile to ignore stale @-mentions
        # from before the current commit was pushed.
        if self.pr.last_commit_sha:
            try:
                commit = await self._github.get_commit(
                    self.pr.repo, self.pr.last_commit_sha
                )
                self._latest_commit_at = _parse_iso(
                    ((commit.get("commit") or {}).get("committer") or {}).get("date", "")
                )
            except Exception:
                self._latest_commit_at = 0.0
        else:
            self._latest_commit_at = 0.0

    # ----- fork sync -----------------------------------------------------

    async def _process_fork_sync(self) -> None:
        """If this PR is an internal copy of an upstream fork PR, detect new
        commits on the upstream branch and spawn claude to merge them in.

        Respawn guard uses `last_spawn_commit` to hold the *origin* sha we
        last attempted to sync. When the fork pushes again the origin sha
        changes and we spawn again. When the copy branch advances (claude
        pushed a merge), `reset_subsystems()` clears the guard naturally.
        """
        sub = self.pr.fork_sync
        if not self.pr.is_copy:
            return
        if sub.state == SubState.CLAUDE_RUNNING:
            await self._poll_claude(sub, self._on_fork_sync_done, "fork_sync")
            return
        if sub.state == SubState.ERROR:
            return

        try:
            origin_pr = await self._github.get_pr(
                self.pr.origin_repo, self.pr.origin_number
            )
        except Exception as e:
            sub.detail = f"failed to fetch origin PR: {e}"
            log.warning(
                "fork_sync %s: failed to fetch origin %s#%d: %s",
                self.pr.key, self.pr.origin_repo, self.pr.origin_number, e,
            )
            return
        origin_head = (origin_pr.get("head") or {})
        origin_sha = origin_head.get("sha", "") or ""
        # Keep our local upstream-branch metadata in sync with what GitHub
        # currently reports — the user may have force-pushed a different
        # branch name onto the fork PR, which would have rotated head.ref.
        new_head_repo = ((origin_head.get("repo") or {}).get("full_name") or "")
        new_head_ref = origin_head.get("ref", "") or ""
        if new_head_repo:
            self.pr.origin_head_repo = new_head_repo
        if new_head_ref:
            self.pr.origin_head_ref = new_head_ref

        if not origin_sha:
            sub.state = SubState.UNKNOWN
            sub.detail = "origin PR has no head sha (fork deleted?)"
            return

        copy_ref = await self._github.get_branch_ref(
            "BerriAI/litellm", self.pr.copy_branch
        )
        if copy_ref is None:
            sub.state = SubState.ERROR
            sub.error_reason = (
                f"copy branch {self.pr.copy_branch} not found on BerriAI/litellm"
            )
            sub.detail = sub.error_reason
            log.warning("fork_sync %s: %s", self.pr.key, sub.error_reason)
            return
        copy_sha = ((copy_ref.get("object") or {}).get("sha") or "")

        # If the copy branch already contains the origin sha, we're in sync.
        # Compare via the commits-compare endpoint, since the copy branch may
        # be *ahead* of origin (babysitter pushed fixes after the last sync)
        # — that's still "in sync" from the fork_sync standpoint.
        if copy_sha == origin_sha:
            sub.state = SubState.DONE
            sub.detail = "in sync with upstream"
            return
        try:
            cmp_data = await self._github.compare_commits(
                "BerriAI/litellm", base=copy_sha, head=origin_sha
            )
        except Exception as e:
            sub.detail = f"failed to compare {copy_sha[:12]}...{origin_sha[:12]}: {e}"
            log.warning("fork_sync %s: %s", self.pr.key, sub.detail)
            return
        ahead_by = int(cmp_data.get("ahead_by") or 0)
        if ahead_by == 0:
            # origin has no commits the copy branch lacks; we're caught up
            # (copy may be ahead, which is fine).
            sub.state = SubState.DONE
            sub.detail = "in sync with upstream"
            return

        # Respawn guard: same origin sha as last attempt → wait.
        if sub.last_spawn_commit and sub.last_spawn_commit == origin_sha:
            sub.state = SubState.UNKNOWN
            sub.detail = (
                f"waiting for new commit after fork_sync attempt on {origin_sha[:12]}"
            )
            return

        log.info(
            "fork_sync %s: upstream %s/%s ahead by %d; spawning claude "
            "(origin_sha=%s, copy_sha=%s)",
            self.pr.key,
            self.pr.origin_head_repo, self.pr.origin_head_ref,
            ahead_by, origin_sha[:12], copy_sha[:12],
        )
        await self._spawn_claude(
            sub,
            FORK_SYNC_PROMPT,
            "fork_sync",
            origin_head_repo=self.pr.origin_head_repo,
            origin_head_ref=self.pr.origin_head_ref,
            origin_head_sha=origin_sha,
            copy_branch=self.pr.copy_branch,
            copy_sha=copy_sha,
        )
        # `_spawn_claude` already recorded `last_spawn_commit = self.pr.last_commit_sha`,
        # but for fork_sync the meaningful identity is the origin sha — overwrite.
        # If the spawn itself failed (state != CLAUDE_RUNNING), `_spawn_claude`
        # may have rolled the guard back to None; only stamp the origin sha
        # when the session actually launched.
        if sub.state == SubState.CLAUDE_RUNNING:
            sub.last_spawn_commit = origin_sha

    async def _on_fork_sync_done(self, sub: Subsystem, envelope: dict) -> None:
        if envelope.get("error") is True:
            # For fork_sync `last_spawn_commit` holds the *origin* sha (the
            # meaningful per-attempt identity). `_spawn_claude`, called from
            # `_maybe_retry_interrupted` below, would overwrite that with the
            # copy's `last_commit_sha`. Capture-and-restore so the respawn
            # guard keeps working across interrupted retries.
            origin_sha = sub.last_spawn_commit or ""
            if await self._maybe_retry_interrupted(
                sub, envelope, FORK_SYNC_PROMPT, "fork_sync",
                origin_head_repo=self.pr.origin_head_repo,
                origin_head_ref=self.pr.origin_head_ref,
                origin_head_sha=origin_sha,
                copy_branch=self.pr.copy_branch,
                copy_sha=(self.pr.last_commit_sha or ""),
            ):
                if sub.state == SubState.CLAUDE_RUNNING and origin_sha:
                    sub.last_spawn_commit = origin_sha
                return
            sub.state = SubState.ERROR
            sub.error_reason = envelope.get("reason", "fork_sync failed")
            sub.detail = sub.error_reason
        else:
            # claude pushed a merge into the copy branch; new commit will
            # trip reset_subsystems() on the next refresh.
            sub.state = SubState.UNKNOWN
            sub.detail = "claude pushed merge; re-evaluating"

    # ----- merge conflicts -----------------------------------------------

    async def _process_merge_conflicts(self) -> None:
        sub = self.pr.merge
        if sub.state == SubState.CLAUDE_RUNNING:
            await self._poll_claude(sub, self._on_merge_done, "merge")
            return
        if sub.state == SubState.ERROR:
            return

        # mergeable_state values we care about:
        #   "dirty"   -> real merge conflicts; spawn claude
        #   "clean" / "unstable" / "blocked" / "behind" / "has_hooks" -> no
        #              conflicts (other gates are someone else's problem)
        #   "" / "unknown" -> GitHub still computing; wait
        # We deliberately re-evaluate even when state is DONE because
        # mergeable_state can flip back to "dirty" if the base branch
        # advances and introduces new conflicts without any push to this PR.
        state = self.pr.mergeable_state
        if state == "dirty":
            if sub.last_spawn_commit and sub.last_spawn_commit == self.pr.last_commit_sha:
                sub.detail = "waiting for new commit after claude merge"
                return
            await self._spawn_claude(sub, MERGE_CONFLICT_PROMPT, "merge")
            return
        if state in ("", "unknown"):
            sub.state = SubState.UNKNOWN
            sub.detail = "waiting for github to compute mergeability"
            return
        sub.state = SubState.DONE
        sub.detail = "no conflicts"

    async def _on_merge_done(self, sub: Subsystem, envelope: dict) -> None:
        if envelope.get("error") is True:
            if await self._maybe_retry_interrupted(sub, envelope, MERGE_CONFLICT_PROMPT, "merge"):
                return
            sub.state = SubState.ERROR
            sub.error_reason = envelope.get("reason", "unrecoverable merge conflict")
            sub.detail = sub.error_reason
        else:
            # claude pushed a merge commit; new commit will reset all subs.
            sub.state = SubState.UNKNOWN
            sub.detail = "claude pushed merge; re-evaluating"

    # ----- bugbot ---------------------------------------------------------

    async def _process_bugbot(self) -> None:
        sub = self.pr.bugbot
        if sub.state == SubState.CLAUDE_RUNNING:
            await self._poll_claude(sub, self._on_bugbot_done, "bugbot")
            return
        if sub.state in (SubState.DONE, SubState.ERROR):
            return
        # NOTE: WAITING_AUTOFIX is intentionally NOT short-circuited here. The
        # autofix verdict may flip from "running" to "completed without a fix"
        # (the false-positive path) while we're parked in WAITING_AUTOFIX, and
        # we need to re-classify to detect that. The per-HEAD guards on
        # `bugbot_run_posted_commit` (and the per-thread guard inside the FP
        # handler) prevent re-acting on the same verdict more than once.

        verdict = self._classify_bugbot_for_head()
        if verdict == "clear":
            sub.state = SubState.DONE
            sub.detail = "no concerns"
            return
        if verdict == "skipped":
            sub.state = SubState.DONE
            sub.detail = "no bugbot check-run on HEAD"
            return
        if verdict == "autofix":
            sub.state = SubState.WAITING_AUTOFIX
            sub.detail = "waiting on bugbot autofix"
            return
        if verdict == "autofix_done_no_fix":
            # If we already kicked off the re-review for this HEAD, stay parked
            # until the new bugbot check-run lands (or commit changes).
            if sub.bugbot_run_posted_commit == self.pr.last_commit_sha:
                sub.state = SubState.WAITING_AUTOFIX
                sub.detail = "re-requested bugbot after false-positive autofix"
                return
            handled = await self._handle_bugbot_autofix_false_positive(sub)
            if handled:
                sub.bugbot_run_posted_commit = self.pr.last_commit_sha
                sub.state = SubState.WAITING_AUTOFIX
                sub.detail = "re-requested bugbot after false-positive autofix"
                return
            # No false-positive replies to act on (autofix gave up silently, or
            # the finding lives on an issue comment rather than a review thread).
            # Treat the original bugbot findings as real concerns to address.
            verdict = "concerns"
        if verdict == "concerns":
            # Respawn guard: if we already spawned claude for this exact commit,
            # don't fire another session while waiting for the new commit (from
            # claude's push) to propagate through the GitHub API.
            if sub.last_spawn_commit and sub.last_spawn_commit == self.pr.last_commit_sha:
                sub.state = SubState.UNKNOWN
                sub.detail = "waiting for new commit after claude fix"
                return
            await self._spawn_claude(sub, BUGBOT_PROMPT, "bugbot")
            return
        # verdict == "none"
        sub.state = SubState.UNKNOWN
        sub.detail = "waiting for bugbot to review HEAD"

    # Substring (lowercased) we look for in cursor[bot]'s reply on a review
    # thread to recognise the "autofix decided this is a false positive"
    # outcome. Cursor's exact wording today is "[Bugbot Autofix](...) determined
    # this is a false positive."; we match on the stable tail so a brand
    # refresh doesn't break detection.
    _BUGBOT_AUTOFIX_FP_MARKER = "determined this is a false positive"

    async def _handle_bugbot_autofix_false_positive(
        self, sub: Subsystem
    ) -> bool:
        """Find unresolved review threads where Cursor Bugbot Autofix posted
        a "false positive" verdict, reply + resolve each, then post a single
        `bugbot run` issue comment to ask Cursor for a fresh review.

        Returns True if we handled at least one thread (i.e. caller should
        mark `bugbot_run_posted_commit` and park in WAITING_AUTOFIX).
        """
        try:
            threads = await self._github.list_review_threads(
                self.pr.repo, self.pr.number
            )
        except Exception as e:
            log.warning(
                "bugbot %s: failed to list review threads while handling "
                "autofix false-positive verdict: %s",
                self.pr.key, e,
            )
            return False

        handled_any = False
        already = set(sub.bugbot_fp_handled_thread_ids or [])
        for t in threads:
            if t.is_resolved:
                continue
            if t.latest_cursor_comment_id is None:
                continue
            if self._BUGBOT_AUTOFIX_FP_MARKER not in (
                t.latest_cursor_comment_body or ""
            ).lower():
                continue
            if t.thread_id in already:
                continue
            try:
                await self._github.post_review_comment_reply(
                    self.pr.repo,
                    self.pr.number,
                    t.latest_cursor_comment_id,
                    "Resolving as false positive",
                )
            except Exception as e:
                log.warning(
                    "bugbot %s: failed to reply on thread %s (parent #%d): %s",
                    self.pr.key, t.thread_id, t.latest_cursor_comment_id, e,
                )
                continue
            try:
                await self._github.resolve_review_thread(t.thread_id)
            except Exception as e:
                # Reply already posted; mark handled anyway so we don't re-
                # reply on the next tick. The thread will just stay
                # unresolved until a human resolves it manually.
                log.warning(
                    "bugbot %s: replied on thread %s but resolve mutation "
                    "failed: %s",
                    self.pr.key, t.thread_id, e,
                )
            sub.bugbot_fp_handled_thread_ids.append(t.thread_id)
            already.add(t.thread_id)
            handled_any = True
            log.info(
                "bugbot %s: resolved autofix false-positive on thread %s "
                "(parent cursor comment #%d, path=%s)",
                self.pr.key, t.thread_id, t.latest_cursor_comment_id, t.path,
            )

        if not handled_any:
            return False

        try:
            posted = await self._github.post_issue_comment(
                self.pr.repo, self.pr.number, "bugbot run"
            )
        except Exception as e:
            # Threads are resolved already; without `bugbot run` Cursor won't
            # re-review on this HEAD, but a new commit will. Returning False
            # would re-enter the FP loop next tick — the per-thread guard
            # blocks the replies, but the per-HEAD guard wouldn't be armed.
            # Return True so we set bugbot_run_posted_commit and park.
            log.warning(
                "bugbot %s: posted FP replies but failed to post `bugbot run`: %s",
                self.pr.key, e,
            )
            return True
        log.info(
            "bugbot %s: posted `bugbot run` issue comment #%d to retrigger "
            "Cursor Bugbot review after %d false-positive thread(s)",
            self.pr.key, posted.id, len(sub.bugbot_fp_handled_thread_ids),
        )
        return True

    async def _on_bugbot_done(self, sub: Subsystem, envelope: dict) -> None:
        if envelope.get("error") is True:
            if await self._maybe_retry_interrupted(sub, envelope, BUGBOT_PROMPT, "bugbot"):
                return
            sub.state = SubState.ERROR
            sub.error_reason = envelope.get("reason", "unknown")
            sub.detail = sub.error_reason
        elif envelope.get("ignore") is True:
            # Claude rebutted on each thread, resolved them, and posted `bugbot
            # run` (per BUGBOT_PROMPT). Don't declare DONE yet — Cursor's check-
            # run for HEAD is still `neutral`, which keeps the PR `blocked`.
            # Park at WAITING_AUTOFIX with the per-HEAD guard armed; the next
            # tick re-classifies and DONE only fires when the fresh check-run
            # lands as `success` (verdict=clear). If Cursor disagrees a second
            # time (verdict=concerns again), the `last_spawn_commit == HEAD`
            # guard at the `concerns` branch parks us at UNKNOWN rather than
            # respawning, so we don't loop on the same commit.
            sub.bugbot_run_posted_commit = self.pr.last_commit_sha
            sub.state = SubState.WAITING_AUTOFIX
            sub.detail = envelope.get("reason", "rebutted; waiting for cursor re-review")
        else:
            # ignore=false: claude pushed a fix. Reset; next tick will pick up new commit.
            sub.state = SubState.UNKNOWN
            sub.detail = "claude pushed fix; re-evaluating"

    # ----- greptile -------------------------------------------------------

    async def _process_greptile(self) -> None:
        sub = self.pr.greptile
        if sub.state == SubState.CLAUDE_RUNNING:
            await self._poll_claude(sub, self._on_greptile_done, "greptile")
            return
        if sub.state == SubState.ERROR:
            return
        if sub.state == SubState.DONE:
            await self._maybe_respawn_greptile_on_followup(sub)
            return

        commit = self.pr.last_commit_sha
        if not commit:
            return

        # WAITING_BOT means we already mentioned greptile for this commit.
        if sub.state == SubState.WAITING_BOT and sub.last_mention_commit == commit:
            mention_comment_id = _extract_mention_comment_id(sub.detail)
            if mention_comment_id:
                try:
                    reactions = await self._github.get_issue_comment_reactions(
                        self.pr.repo, mention_comment_id
                    )
                except Exception:
                    return
                has_thumbs = any(
                    r.content in ("+1", "rocket", "heart")
                    and _login_matches(r.user_login, GREPTILE_LOGINS)
                    for r in reactions
                )
                if has_thumbs:
                    log.info(
                        "greptile %s: mention #%d got greptile +1/rocket/heart; spawning claude",
                        self.pr.key, mention_comment_id,
                    )
                    await self._spawn_claude(sub, GREPTILE_PROMPT, "greptile")
                    return
                # still waiting on greptile to review
                has_eyes = any(
                    r.content == "eyes" and _login_matches(r.user_login, GREPTILE_LOGINS)
                    for r in reactions
                )
                sub.detail = (
                    f"#mention:{mention_comment_id} | greptile reviewing"
                    if has_eyes
                    else f"#mention:{mention_comment_id} | waiting for greptile"
                )
            return

        # Don't re-post if we've already mentioned for this exact commit.
        # State may be UNKNOWN here because _on_greptile_done just fired with
        # ignore=false, but the new commit hasn't propagated through the
        # GitHub API yet — wait for reset_subsystems() on the next refresh
        # rather than spamming a second @-mention on the old commit.
        if sub.last_mention_commit == commit:
            return

        # If greptile's existing summary already covers HEAD (created or last
        # edited after the HEAD commit), greptile has already reviewed this
        # commit. Spawn claude on the existing summary directly — a fresh
        # @greptileai mention would either be ignored or trigger a redundant
        # re-review.
        summary_at = self._greptile_summary_at()
        if self._latest_commit_at and summary_at > self._latest_commit_at:
            sub.last_mention_commit = commit
            log.info(
                "greptile %s: summary already covers HEAD "
                "(summary_at=%.0f > commit_at=%.0f); spawning claude without @mention",
                self.pr.key, summary_at, self._latest_commit_at,
            )
            await self._spawn_claude(sub, GREPTILE_PROMPT, "greptile")
            return

        # If no commit has landed since the PR was opened, greptile auto-runs
        # its initial review on a non-draft PR. Wait for that review (which
        # will then trip the branch above on a later tick) rather than
        # @-mentioning ahead of greptile's own first pass.
        if (
            self._latest_commit_at
            and self._pr_opened_at
            and self._latest_commit_at <= self._pr_opened_at
        ):
            sub.state = SubState.UNKNOWN
            sub.detail = "waiting for greptile's initial auto-review"
            return

        # If a human already @-mentioned greptile after HEAD, adopt that
        # comment instead of double-posting. Greptile reacts (+1) on the
        # human's mention just as it would on ours, so the existing
        # reaction-polling branch above can drive the rest of the flow.
        adopted = self._find_existing_greptile_mention()
        if adopted is not None:
            sub.state = SubState.WAITING_BOT
            sub.last_mention_commit = commit
            sub.detail = f"#mention:{adopted.id} | waiting for greptile"
            log.info(
                "greptile %s: adopted existing mention #%d by %s for sha=%s",
                self.pr.key, adopted.id, adopted.user_login, commit[:12],
            )
            return

        # need to post a fresh @greptileai mention for this commit
        try:
            posted = await self._github.post_issue_comment(
                self.pr.repo, self.pr.number, "@greptileai"
            )
        except Exception as e:
            sub.state = SubState.UNKNOWN
            sub.detail = f"failed to @mention greptile: {e}"
            log.warning("greptile %s: failed to post @greptileai mention: %s", self.pr.key, e)
            return
        sub.state = SubState.WAITING_BOT
        sub.last_mention_commit = commit
        sub.detail = f"#mention:{posted.id} | waiting for greptile"
        log.info(
            "greptile %s: posted @greptileai mention #%d for sha=%s",
            self.pr.key, posted.id, commit[:12],
        )

    async def _on_greptile_done(self, sub: Subsystem, envelope: dict) -> None:
        if envelope.get("error") is True:
            if await self._maybe_retry_interrupted(sub, envelope, GREPTILE_PROMPT, "greptile"):
                return
            sub.state = SubState.ERROR
            sub.error_reason = envelope.get("reason", "unknown")
            sub.detail = sub.error_reason
        elif envelope.get("ignore") is True:
            sub.state = SubState.DONE
            sub.detail = envelope.get("reason", "no legit concerns")
        else:
            # ignore=false: claude pushed a fix. New commit will reset; we'll re-mention.
            sub.state = SubState.UNKNOWN
            sub.detail = "claude pushed fix; re-mentioning greptile next tick"

    async def _maybe_respawn_greptile_on_followup(self, sub: Subsystem) -> None:
        """If greptile posted or edited a comment after our last claude spawn,
        re-spawn claude so it can read the new content. Capped per commit.

        Why this exists: claude's previous greptile session may have ended with
        `ignore=true` after posting a rebuttal comment. Greptile then often
        replies ("Concern resolved" or sticks to its guns) — but the babysitter
        used to latch DONE and never look again, so the PR's TUI would show a
        stale rebuttal forever even after greptile agreed with it.
        """
        if (sub.claude_spawn_count or 0) >= GREPTILE_MAX_SPAWNS_PER_COMMIT:
            return
        spawned_at = sub.claude_last_spawn_at
        if spawned_at is None:
            # We're in DONE without ever having spawned claude — shouldn't
            # happen via normal flow (DONE always comes from _on_greptile_done
            # which runs after a spawn), but be defensive rather than nudge.
            return
        followup = self._latest_greptile_followup_after(spawned_at)
        if followup is None:
            return
        log.info(
            "greptile %s: detected greptile follow-up #%d (updated=%s) since "
            "last claude spawn at %.0f; re-spawning claude (spawn_count=%d/%d)",
            self.pr.key, followup.id, followup.updated_at, spawned_at,
            sub.claude_spawn_count or 0, GREPTILE_MAX_SPAWNS_PER_COMMIT,
        )
        await self._spawn_claude(sub, GREPTILE_PROMPT, "greptile")

    def _greptile_summary_at(self) -> float:
        """Latest timestamp at which greptile's summary comment was created or
        edited. Returns 0.0 if greptile hasn't posted a summary on this PR.

        Greptile marks its main summary comment with `Confidence Score: X/5`;
        plain reactions and inline review threads don't carry that marker, so
        the substring match cleanly excludes follow-up chatter. We want the
        latest such comment in case greptile ever posts more than one (it
        normally edits the existing one in place, but be defensive)."""
        latest_ts = 0.0
        for c in self._issue_comments:
            if not _login_matches(c.user_login, GREPTILE_LOGINS):
                continue
            if "confidence score" not in (c.body or "").lower():
                continue
            ts = max(_parse_iso(c.created_at), _parse_iso(c.updated_at))
            if ts > latest_ts:
                latest_ts = ts
        return latest_ts

    def _latest_greptile_followup_after(self, after_ts: float) -> Optional[Comment]:
        """Latest greptile-authored issue comment whose created/updated time is
        after `after_ts`. Returns None if greptile hasn't said anything new."""
        latest: Optional[Comment] = None
        latest_ts = 0.0
        for c in self._issue_comments:
            if not _login_matches(c.user_login, GREPTILE_LOGINS):
                continue
            ts = max(_parse_iso(c.created_at), _parse_iso(c.updated_at))
            if ts <= after_ts:
                continue
            if ts < latest_ts:
                continue
            latest_ts = ts
            latest = c
        return latest

    # ----- veria ----------------------------------------------------------

    async def _process_veria(self) -> None:
        sub = self.pr.veria
        if sub.state == SubState.CLAUDE_RUNNING:
            await self._poll_claude(sub, self._on_veria_done, "veria")
            return
        if sub.state == SubState.ERROR:
            return
        # Re-evaluate even when DONE: veria can replace a completed `skipped`
        # check-run with a fresh `in_progress` review (and concerns may show up
        # later), so latching DONE on the first tick would hide the new verdict.

        prev_state = sub.state.value
        verdict = self._classify_veria_for_head()
        if verdict == "clear":
            sub.state = SubState.DONE
            sub.detail = "no open security findings"
        elif verdict == "concerns":
            if sub.last_spawn_commit and sub.last_spawn_commit == self.pr.last_commit_sha:
                # Already spawned claude on this SHA; don't loop on the same
                # unchanging check-run. If claude already reviewed and dismissed
                # the concerns (state=DONE via ignore=true in _on_veria_done),
                # latch DONE — re-spawning would just yield the same verdict.
                # Otherwise claude pushed a fix and we're waiting for the new
                # commit, so demote to UNKNOWN so overall_status() doesn't show
                # a stale DONE. Fall through to the diagnostic log below so
                # this guard path is visible in logs like every other branch.
                if sub.state != SubState.DONE:
                    sub.state = SubState.UNKNOWN
                    sub.detail = "waiting for new commit after claude fix"
            else:
                await self._spawn_claude(sub, VERIA_PROMPT, "veria")
        else:
            sub.state = SubState.UNKNOWN
            sub.detail = "waiting for veria to review HEAD"
        log.info(
            "veria %s: verdict=%s sha=%s prev_state=%s -> new_state=%s "
            "last_spawn_commit=%s",
            self.pr.key, verdict, (self.pr.last_commit_sha or "none")[:12],
            prev_state, sub.state.value,
            (sub.last_spawn_commit or "none")[:12],
        )

    async def _on_veria_done(self, sub: Subsystem, envelope: dict) -> None:
        if envelope.get("error") is True:
            if await self._maybe_retry_interrupted(sub, envelope, VERIA_PROMPT, "veria"):
                return
            sub.state = SubState.ERROR
            sub.error_reason = envelope.get("reason", "unknown")
            sub.detail = sub.error_reason
        elif envelope.get("ignore") is True:
            sub.state = SubState.DONE
            sub.detail = envelope.get("reason", "no legit concerns")
        else:
            sub.state = SubState.UNKNOWN
            sub.detail = "claude pushed fix; re-evaluating"

    # ----- cicd -----------------------------------------------------------

    async def _process_cicd(self) -> None:
        sub = self.pr.cicd
        if sub.state == SubState.CLAUDE_RUNNING:
            await self._poll_claude(sub, self._on_cicd_done, "cicd")
            return
        # Re-evaluate every tick regardless of stored state (DONE, ERROR,
        # UNKNOWN): CircleCI commit statuses post on their own schedule, so a
        # SHA that looked "all green" from Actions check-runs at HEAD+0s can
        # flip to failure once CircleCI reports. Latching DONE would hide
        # those failures. Re-evaluating from ERROR also lets us retry when a
        # previous claude session was killed mid-task (backend unknown_error,
        # missing envelope, etc.) and CI/CD is still red.

        sha = self.pr.last_commit_sha
        if not sha:
            return
        prev_state = sub.state.value
        verdict = await self._compute_cicd_verdict(sha)
        if verdict == "success":
            sub.state = SubState.DONE
            sub.detail = "all checks green"
            sub.error_reason = None
            sub.cicd_consecutive_failure_ticks = 0
        elif verdict == "pending":
            sub.state = SubState.UNKNOWN
            sub.detail = "checks running"
            sub.error_reason = None
            # `pending` after a previous `failure` is the signal that the
            # agent's re-run actually kicked off — reset the gate so the next
            # failure (if it comes) starts a fresh confirmation window.
            sub.cicd_consecutive_failure_ticks = 0
        elif verdict == "failure":
            # Respawn guard: if we already spawned (or attempted to spawn)
            # claude for this exact commit, don't fire another session. This
            # matches the bugbot/veria guards and, crucially, caps retries on
            # persistent infrastructure failures: a failed _spawn_claude sets
            # `last_spawn_commit = last_commit_sha` but leaves the subsystem
            # in ERROR, and without this guard cicd would just count back up
            # to CICD_FAILURE_TICKS_BEFORE_RESPAWN and respawn every ~90s.
            if sub.last_spawn_commit and sub.last_spawn_commit == self.pr.last_commit_sha:
                if sub.state != SubState.ERROR:
                    sub.state = SubState.UNKNOWN
                    sub.detail = "waiting for new commit after claude fix"
                log.debug(
                    "cicd %s: respawn guarded by last_spawn_commit=%s (sha=%s)",
                    self.pr.key, (sub.last_spawn_commit or "none")[:12], sha[:12],
                )
            else:
                # Gate respawn on consecutive-failure confirmation. The first
                # `failure` tick after an agent issued a re-run is frequently
                # stale (the previous run's status, not the re-run's), so we
                # wait until we've seen `failure` CICD_FAILURE_TICKS_BEFORE_RESPAWN
                # times in a row before spawning. On success/pending the counter
                # resets, naturally re-arming the gate for the next failure.
                sub.cicd_consecutive_failure_ticks = (
                    sub.cicd_consecutive_failure_ticks or 0
                ) + 1
                if sub.cicd_consecutive_failure_ticks >= CICD_FAILURE_TICKS_BEFORE_RESPAWN:
                    attempt = (
                        (sub.claude_spawn_count or 0)
                        + (sub.interrupted_retry_count or 0)
                        + 1
                    )
                    log.info(
                        "cicd %s: failure confirmed across %d ticks; spawning "
                        "agent (attempt %d, prev_state=%s, prev_error=%r)",
                        self.pr.key, sub.cicd_consecutive_failure_ticks,
                        attempt, prev_state, sub.error_reason,
                    )
                    sub.error_reason = None
                    sub.cicd_consecutive_failure_ticks = 0
                    await self._spawn_claude(sub, CICD_PROMPT, "cicd", attempt=attempt)
                else:
                    # Confirming. Surface the live failure verdict so the TUI
                    # doesn't keep showing a stale DONE/"all checks green"
                    # during the ~60s confirmation window. Preserve a prior
                    # ERROR's error_reason so the user can still see why we
                    # last gave up.
                    if sub.state != SubState.ERROR:
                        sub.state = SubState.UNKNOWN
                        sub.detail = (
                            f"cicd failure detected, confirming "
                            f"({sub.cicd_consecutive_failure_ticks}/"
                            f"{CICD_FAILURE_TICKS_BEFORE_RESPAWN})"
                        )
                    log.debug(
                        "cicd %s: failure tick %d/%d (sha=%s) — waiting for "
                        "confirmation before respawn",
                        self.pr.key, sub.cicd_consecutive_failure_ticks,
                        CICD_FAILURE_TICKS_BEFORE_RESPAWN, sha[:12],
                    )
        log.info(
            "cicd %s: verdict=%s sha=%s prev_state=%s -> new_state=%s "
            "fail_ticks=%d last_spawn_commit=%s",
            self.pr.key, verdict, sha[:12], prev_state, sub.state.value,
            sub.cicd_consecutive_failure_ticks,
            (sub.last_spawn_commit or "none")[:12],
        )

    async def _compute_cicd_verdict(self, sha: str) -> str:
        # check-runs for this SHA were already fetched into self._check_runs
        # at the top of tick(); reuse them instead of paginating again.
        try:
            combined = await self._github.get_combined_status(self.pr.repo, sha)
        except Exception as e:
            # Don't compute a verdict from check-runs alone on a transient
            # fetch failure. CircleCI (and other providers) report via
            # legacy commit statuses, not check-runs — so if every
            # check-run is green but a CircleCI status was failing, we'd
            # falsely conclude success. Stay pending; next tick retries.
            self.pr.last_error = f"cicd fetch failed: {e}"
            log.warning(
                "cicd %s: combined-status fetch failed (%s); returning pending",
                self.pr.key, e,
            )
            return "pending"

        cr_breakdown: dict[str, int] = {"success": 0, "failure": 0, "pending": 0}
        cr_failures: list[str] = []
        st_breakdown: dict[str, int] = {"success": 0, "failure": 0, "pending": 0}
        st_failures: list[str] = []
        states: list[str] = []
        for run in self._check_runs:
            name = (run.get("name") or "").lower()
            if "bugbot" in name or "veria" in name:
                continue
            if run.get("status") != "completed":
                states.append("pending")
                cr_breakdown["pending"] += 1
                continue
            conclusion = run.get("conclusion") or ""
            if conclusion in ("success", "neutral", "skipped", "stale"):
                states.append("success")
                cr_breakdown["success"] += 1
            elif conclusion in ("failure", "timed_out", "cancelled", "action_required", "startup_failure"):
                states.append("failure")
                cr_breakdown["failure"] += 1
                cr_failures.append(f"{run.get('name')}={conclusion}")
            else:
                states.append("pending")
                cr_breakdown["pending"] += 1

        combined_state = combined.get("state", "")
        statuses_list = combined.get("statuses", []) or []
        # combined_state=="pending" with zero statuses is the GitHub-side race
        # we get burned by: right after HEAD lands, no provider has posted a
        # commit status yet, so check-runs alone (which arrive from Actions
        # first) make the verdict look green. Hold pending within the grace
        # window so CircleCI/etc. have a chance to report; after grace, fall
        # through (some repos use only check-runs and would otherwise wait
        # forever).
        if combined_state == "pending" and not statuses_list:
            within_grace = (
                self._latest_commit_at
                and time.time() - self._latest_commit_at < CICD_STATUS_GRACE_SECONDS
            )
            if within_grace:
                states.append("pending")
                log.info(
                    "cicd %s: combined_state=pending with 0 statuses; holding "
                    "pending within %ds grace window (head age=%.0fs)",
                    self.pr.key, CICD_STATUS_GRACE_SECONDS,
                    time.time() - self._latest_commit_at,
                )

        for status in statuses_list:
            ctx = status.get("context", "?")
            ctx_l = ctx.lower() if isinstance(ctx, str) else ""
            if "bugbot" in ctx_l or "veria" in ctx_l:
                # bugbot/veria are handled by their own subsystems; don't let
                # their failure contexts bleed into the cicd verdict.
                continue
            s = status.get("state", "")
            if s == "success":
                states.append("success")
                st_breakdown["success"] += 1
            elif s in ("failure", "error"):
                states.append("failure")
                st_breakdown["failure"] += 1
                st_failures.append(f"{ctx}={s}")
            else:
                states.append("pending")
                st_breakdown["pending"] += 1

        log.info(
            "cicd %s: sha=%s combined_top_state=%s combined_total=%s "
            "check_runs=%s statuses_seen=%s "
            "check_runs_breakdown=%s statuses_breakdown=%s",
            self.pr.key, sha[:12],
            combined.get("state"), combined.get("total_count"),
            len(self._check_runs), len(combined.get("statuses", []) or []),
            cr_breakdown, st_breakdown,
        )
        if cr_failures:
            log.info("cicd %s: check-run failures: %s", self.pr.key, cr_failures)
        if st_failures:
            log.info("cicd %s: status failures: %s", self.pr.key, st_failures)

        if not states:
            return "pending"
        if "failure" in states:
            return "failure"
        if "pending" in states:
            return "pending"
        return "success"

    async def _on_cicd_done(self, sub: Subsystem, envelope: dict) -> None:
        # Release the per-SHA respawn guard now that a real claude session
        # has completed. The guard set by `_spawn_claude` is only there to
        # cap infinite loops on infrastructure failures (where the spawn
        # itself raised and `_on_cicd_done` is never called); for normal
        # session completions we want the tick-based confirmation gate
        # (`CICD_FAILURE_TICKS_BEFORE_RESPAWN`) to drive any further retries
        # on the same SHA, matching the documented multi-attempt design.
        sub.last_spawn_commit = None
        if envelope.get("error") is True:
            attempt = (
                (sub.claude_spawn_count or 0)
                + (sub.interrupted_retry_count or 0)
                + 1
            )
            if await self._maybe_retry_interrupted(
                sub, envelope, CICD_PROMPT, "cicd", attempt=attempt
            ):
                return
            sub.state = SubState.ERROR
            sub.error_reason = envelope.get("reason", "unrecoverable cicd failure")
            sub.detail = sub.error_reason
        else:
            sub.state = SubState.UNKNOWN
            sub.detail = "claude reran/fixed cicd; re-evaluating"

    # ----- claude helpers -------------------------------------------------

    async def _maybe_retry_interrupted(
        self,
        sub: Subsystem,
        envelope: dict,
        prompt_tpl: str,
        label: str,
        **extra_prompt_kwargs: object,
    ) -> bool:
        """If the envelope is `error: true, reason: "interrupted: ..."` AND we
        haven't blown the per-SHA retry budget, respawn claude immediately and
        return True. Caller should then return without touching state — the
        spawn already transitioned the subsystem to CLAUDE_RUNNING. Returns
        False for any non-`interrupted:` envelope (caller falls through to its
        usual error/ignore routing) or when the budget is exhausted (caller
        proceeds to mark the subsystem ERROR)."""
        reason = envelope.get("reason", "")
        is_interrupted = (
            envelope.get("error") is True
            and isinstance(reason, str)
            and reason.startswith("interrupted:")
        )
        if not is_interrupted:
            return False
        if (sub.interrupted_retry_count or 0) >= INTERRUPTED_RETRY_CAP:
            log.warning(
                "%s/%s: interrupted retry budget exhausted (%d/%d on this SHA); "
                "routing to ERROR. reason=%r",
                self.pr.key, label,
                sub.interrupted_retry_count or 0, INTERRUPTED_RETRY_CAP, reason,
            )
            return False
        log.info(
            "%s/%s: claude reported interrupted; respawning (attempt %d/%d). reason=%r",
            self.pr.key, label,
            (sub.interrupted_retry_count or 0) + 1, INTERRUPTED_RETRY_CAP, reason,
        )
        await self._spawn_claude(
            sub, prompt_tpl, label,
            _interrupted_retry=True,
            **extra_prompt_kwargs,
        )
        return True

    async def _spawn_claude(
        self,
        sub: Subsystem,
        prompt_tpl: str,
        label: str,
        *,
        _interrupted_retry: bool = False,
        **extra_prompt_kwargs: object,
    ) -> None:
        pr_url = f"https://github.com/{self.pr.repo}/pull/{self.pr.number}"
        prompt = prompt_tpl.format(
            pr_url=pr_url,
            base_branch=self.pr.base_branch,
            **extra_prompt_kwargs,
        )
        # Record the attempt up front so a failed spawn still parks the
        # respawn guard. Without setting `last_spawn_commit` here, cicd
        # (which re-evaluates from ERROR) would spin every tick on a
        # persistent infrastructure failure (bad API key, network down) -
        # state goes ERROR but `last_spawn_commit` stays None. The actual
        # `claude_spawn_count` / `interrupted_retry_count` bumps happen
        # below, AFTER we confirm a real session id was returned, so a
        # failed spawn doesn't inflate the `attempt` value the agent sees
        # in CICD_PROMPT or eat greptile's per-commit follow-up budget.
        prev_last_spawn_commit = sub.last_spawn_commit
        prev_claude_last_spawn_at = sub.claude_last_spawn_at
        prev_detail = sub.detail
        sub.last_spawn_commit = self.pr.last_commit_sha
        sub.claude_last_spawn_at = time.time()
        try:
            session_id = await self._claude.spawn(prompt, self.pr.repo)
        except SessionSpawnQuotaError as e:
            # Anthropic's concurrent-session quota is full (or we're inside
            # the client-side backoff window after a recent 400). Roll back
            # the respawn guard AND the spawn timestamp so a future tick
            # re-attempts once the quota frees, instead of consuming this
            # SHA's spawn budget on a call that never actually opened a
            # session. Restoring `claude_last_spawn_at` also keeps the
            # greptile follow-up detector from filtering out comments that
            # arrived just before this failed attempt.
            sub.last_spawn_commit = prev_last_spawn_commit
            sub.claude_last_spawn_at = prev_claude_last_spawn_at
            # Preserve the `#mention:NN` prefix (if any) on `sub.detail` so
            # the greptile WAITING_BOT reaction-polling branch can still
            # extract the mention id on the next tick — otherwise the
            # subsystem would stall until a new commit reset state.
            mention_id = _extract_mention_comment_id(prev_detail)
            deferred_msg = f"claude spawn ({label}) deferred — session quota busy"
            sub.detail = (
                f"#mention:{mention_id} | {deferred_msg}"
                if mention_id is not None
                else deferred_msg
            )
            log.info(
                "%s/%s: claude spawn deferred (quota busy): %s",
                self.pr.key, label, e.detail or e,
            )
            return
        except Exception as e:
            # Terminal: don't loop. For greptile in particular, a non-ERROR
            # state would re-post `@greptileai` on the very next tick,
            # spamming the PR. ERROR pauses the subsystem until the user
            # intervenes or a new commit resets state.
            sub.state = SubState.ERROR
            sub.error_reason = f"failed to spawn claude ({label}): {e}"
            sub.detail = sub.error_reason
            log.warning("%s/%s: claude spawn failed: %s", self.pr.key, label, e)
            return
        if not session_id:
            # spawn() returned but produced no id — there's nothing to poll, and
            # _poll_claude would silently reset to UNKNOWN, causing the same
            # mention-spam loop as a raised exception. Treat as ERROR.
            sub.state = SubState.ERROR
            sub.error_reason = f"claude spawn ({label}) returned empty session id"
            sub.detail = sub.error_reason
            log.warning("%s/%s: claude spawn returned empty session id", self.pr.key, label)
            return
        if _interrupted_retry:
            sub.interrupted_retry_count = (sub.interrupted_retry_count or 0) + 1
        else:
            sub.claude_spawn_count = (sub.claude_spawn_count or 0) + 1
        now = sub.claude_last_spawn_at
        sub.state = SubState.CLAUDE_RUNNING
        sub.claude_session_id = session_id
        sub.poll_failures = 0
        sub.claude_started_at = now
        sub.last_heartbeat_log_at = now
        # Fresh session — fresh per-session nudge budget. We allow one resume
        # nudge per session id (see _poll_claude's needs_nudge handling).
        sub.claude_nudged_session_id = None
        sub.detail = f"claude session {session_id[:8]} running ({label}) — 0s"
        log.info(
            "%s/%s: spawned claude session %s for sha=%s",
            self.pr.key, label, session_id[:12],
            (self.pr.last_commit_sha or "none")[:12],
        )

    # After this many consecutive HTTP failures polling a claude session, give
    # up and transition the subsystem to ERROR instead of looping forever. This
    # matters after an app restart where the persisted session id has expired
    # server-side (e.g. 404) — without the threshold we'd poll-fail every tick.
    _POLL_FAILURE_THRESHOLD = 5

    async def _poll_claude(self, sub: Subsystem, on_done, label: str = "?") -> None:
        if not sub.claude_session_id:
            sub.state = SubState.UNKNOWN
            sub.poll_failures = 0
            return
        session_id = sub.claude_session_id
        try:
            result = await self._claude.get(session_id)
        except Exception as e:
            sub.poll_failures += 1
            sub.detail = (
                f"poll error ({sub.poll_failures}/{self._POLL_FAILURE_THRESHOLD}): {e}"
            )
            if sub.poll_failures >= self._POLL_FAILURE_THRESHOLD:
                sub.state = SubState.ERROR
                sub.error_reason = f"claude poll failed repeatedly: {e}"
                sub.claude_session_id = None
                sub.poll_failures = 0
                sub.claude_started_at = None
                sub.last_heartbeat_log_at = 0.0
                log.warning(
                    "%s/%s: claude session %s gave up after %d poll failures: %s",
                    self.pr.key, label, session_id[:12],
                    self._POLL_FAILURE_THRESHOLD, e,
                )
            return
        sub.poll_failures = 0
        if result.status in ("queued", "running"):
            now = time.time()
            started = sub.claude_started_at or now
            elapsed = now - started
            elapsed_str = _format_elapsed(elapsed)
            sub.detail = (
                f"claude session {session_id[:8]} running ({label}) — {elapsed_str}"
            )
            if now - sub.last_heartbeat_log_at >= _HEARTBEAT_INTERVAL_SECONDS:
                sub.last_heartbeat_log_at = now
                # One extra HTTP per heartbeat (every ~2 min) to count
                # events by type — gives us "agent has done 5 tool calls,
                # 0 messages, 4 errors" so a corpse is obvious without
                # having to curl the Anthropic API by hand.
                counts = await self._claude.count_events(session_id)
                stuck = elapsed >= _HEARTBEAT_STUCK_SECONDS
                emit = log.warning if stuck else log.info
                emit(
                    "%s/%s: claude session %s still running (%s elapsed, "
                    "api=%s active=%.1fs msgs=%d tools=%d errors=%d total_evts=%d)%s",
                    self.pr.key, label, session_id[:12], elapsed_str,
                    result.api_status or "?", result.active_seconds,
                    counts.get("agent.message", 0), counts.get("agent.tool_use", 0),
                    counts.get("session.error", 0), counts.get("total", 0),
                    " [STUCK?]" if stuck else "",
                )
            return
        elapsed_str = _format_elapsed(
            time.time() - sub.claude_started_at
        ) if sub.claude_started_at else "?"
        if result.status == "failed":
            sub.state = SubState.ERROR
            sub.error_reason = result.error or "claude session failed"
            sub.detail = sub.error_reason
            sub.claude_session_id = None
            sub.claude_started_at = None
            sub.last_heartbeat_log_at = 0.0
            # The per-SHA respawn guard only exists to cap spawn-time
            # infrastructure failures (see `_spawn_claude`). A backend-killed
            # session is not a spawn failure, so clear the guard to let
            # subsystems with multi-attempt retry semantics (cicd) try again
            # on the same SHA. Latching subsystems (bugbot/veria) stay
            # paused via their `_process_*` ERROR early-return.
            sub.last_spawn_commit = None
            transcript_path = _dump_transcript(session_id, result.response_text)
            tail = result.response_text[-_TRANSCRIPT_TAIL_CHARS:] if result.response_text else ""
            log.warning(
                "%s/%s: claude session %s failed after %s: %s%s%s",
                self.pr.key, label, session_id[:12], elapsed_str, sub.error_reason,
                f" (transcript: {transcript_path})" if transcript_path else "",
                f"\n--- last {len(tail)} chars of transcript ---\n{tail}" if tail else "",
            )
            return
        if result.status == "needs_nudge":
            # Session went idle without an envelope. `idle` is resumable (see
            # docs: "agent waiting for input") — its full event history is
            # intact, so a short nudge usually gets the envelope without
            # paying the cost of a fresh investigation. Allow exactly one
            # nudge per session id; if the nudge also yields no envelope, we
            # fall through to ERROR on the next idle tick.
            already_nudged = sub.claude_nudged_session_id == session_id
            if not already_nudged:
                try:
                    await self._claude.send_user_message(session_id, RESUME_PROMPT)
                except Exception as e:
                    # Backend refused the nudge — give up on this session.
                    sub.state = SubState.ERROR
                    sub.error_reason = f"resume nudge failed: {e}"
                    sub.claude_session_id = None
                    sub.claude_started_at = None
                    sub.last_heartbeat_log_at = 0.0
                    transcript_path = _dump_transcript(session_id, result.response_text)
                    log.warning(
                        "%s/%s: claude session %s nudge POST failed after %s: %s%s",
                        self.pr.key, label, session_id[:12], elapsed_str, e,
                        f" (transcript: {transcript_path})" if transcript_path else "",
                    )
                    return
                sub.claude_nudged_session_id = session_id
                sub.detail = (
                    f"claude session {session_id[:8]} nudged for envelope "
                    f"({label}, stop_reason={result.stop_reason or 'unknown'})"
                )
                log.info(
                    "%s/%s: claude session %s idle without envelope after %s "
                    "(stop_reason=%s, last_error=%r) — sent resume nudge",
                    self.pr.key, label, session_id[:12], elapsed_str,
                    result.stop_reason or "unknown", result.error or "",
                )
                # Stay in CLAUDE_RUNNING; next poll will pick up the new turn.
                return
            # Already nudged this session and it's *still* coming back without
            # an envelope. Treat as a genuine missing-envelope error and let
            # the subsystem-level retry path (cicd has one) take over.
            sub.state = SubState.ERROR
            sub.error_reason = (
                "claude session ended idle without envelope even after a "
                f"resume nudge (stop_reason={result.stop_reason or 'unknown'}, "
                f"last_session_error={result.error or 'none'!r})"
            )
            sub.detail = sub.error_reason
            sub.claude_session_id = None
            sub.claude_started_at = None
            sub.last_heartbeat_log_at = 0.0
            # Spawn succeeded; release the per-SHA guard so cicd can retry.
            sub.last_spawn_commit = None
            transcript_path = _dump_transcript(session_id, result.response_text)
            tail = result.response_text[-_TRANSCRIPT_TAIL_CHARS:] if result.response_text else ""
            log.warning(
                "%s/%s: claude session %s still no envelope after nudge "
                "(elapsed %s, stop_reason=%s)%s\n--- last %d chars of transcript ---\n%s",
                self.pr.key, label, session_id[:12], elapsed_str,
                result.stop_reason or "unknown",
                f" (transcript: {transcript_path})" if transcript_path else "",
                len(tail), tail,
            )
            return
        # completed
        envelope = extract_envelope(result.response_text)
        sub.claude_session_id = None
        sub.claude_started_at = None
        sub.last_heartbeat_log_at = 0.0
        if envelope is None:
            # Treat a missing envelope as an error so we don't fall into the
            # "claude pushed a fix" branch and re-spawn another session next tick.
            sub.state = SubState.ERROR
            sub.error_reason = "claude finished but envelope not found in response"
            sub.detail = "claude finished but envelope not found in response"
            # Spawn succeeded; release the per-SHA guard so cicd can retry.
            sub.last_spawn_commit = None
            transcript_path = _dump_transcript(session_id, result.response_text)
            tail = result.response_text[-_TRANSCRIPT_TAIL_CHARS:] if result.response_text else ""
            log.warning(
                "%s/%s: claude session %s completed after %s but no envelope in response%s\n"
                "--- last %d chars of transcript ---\n%s",
                self.pr.key, label, session_id[:12], elapsed_str,
                f" (transcript: {transcript_path})" if transcript_path else "",
                len(tail), tail,
            )
            return
        log.info(
            "%s/%s: claude session %s completed after %s envelope=%s",
            self.pr.key, label, session_id[:12], elapsed_str, envelope,
        )
        await on_done(sub, envelope)
        log.info(
            "%s/%s: post-envelope state=%s detail=%r",
            self.pr.key, label, sub.state.value, sub.detail,
        )

    # ----- check-run classification --------------------------------------
    #
    # Bugbot and veria publish their "I've reviewed HEAD" signal as a GitHub
    # check-run on the head SHA. The check-run is the source of truth — body
    # parsing of issue/review comments was unreliable because both bots skip
    # posting a top-level summary when there are no new issues, and veria's
    # per-issue feedback is a line-level review comment that doesn't appear
    # in /issues/{n}/comments at all.

    def _classify_bugbot_for_head(self) -> str:
        """Verdict from Cursor Bugbot's check-runs on HEAD.

        Returns one of:
          'clear'              - main bugbot run completed with no findings
          'autofix'            - autofix still in-flight or succeeded (waiting for
                                 the resulting commit to land and reset us)
          'autofix_done_no_fix'- autofix completed without pushing a fix (e.g.
                                 it decided the finding was a false positive
                                 and posted a reply on the review thread)
          'concerns'           - bugbot found issues and there's no autofix to
                                 wait on; claude needs to address them
          'skipped'            - no bugbot check-run created within the grace
                                 window (paused for billing, uninstalled, etc.)
          'none'               - still running, no decision yet
        """
        main_candidates = _matching_check_runs(self._check_runs, "bugbot", exclude="autofix")
        autofix_candidates = _matching_check_runs(self._check_runs, "bugbot autofix")
        main = _pick_latest(main_candidates)
        autofix = _pick_latest(autofix_candidates)
        head_age = (time.time() - self._latest_commit_at) if self._latest_commit_at else None
        log.info(
            "bugbot %s: main_candidates=%d autofix_candidates=%d "
            "main_picked=%s autofix_picked=%s head_age=%s | main=[%s] | autofix=[%s]",
            self.pr.key, len(main_candidates), len(autofix_candidates),
            _describe_check_run(main), _describe_check_run(autofix),
            f"{head_age:.0f}s" if head_age is not None else "?",
            " ; ".join(_describe_check_run(r) for r in main_candidates) or "(none)",
            " ; ".join(_describe_check_run(r) for r in autofix_candidates) or "(none)",
        )
        if main is None:
            # Bugbot may just not have created its check-run yet (webhook +
            # queue latency), or it may never create one for this PR (paused
            # for billing, uninstalled, or filtered out repo-side). Wait the
            # grace period after HEAD before assuming the latter.
            if self._latest_commit_at and (
                time.time() - self._latest_commit_at
                > BUGBOT_NO_CHECK_RUN_GRACE_SECONDS
            ):
                return "skipped"
            return "none"
        if main.get("status") != "completed":
            return "none"
        if main.get("conclusion") == "success":
            return "clear"
        # bugbot found issues. If autofix is configured and still working,
        # wait for it to either push a fix (new commit resets us) or give up.
        if autofix is not None:
            if autofix.get("status") != "completed":
                return "autofix"
            if autofix.get("conclusion") == "success":
                # autofix pushed a fix; new commit is on its way
                return "autofix"
            # autofix completed but didn't push a fix (e.g. conclusion=neutral
            # because it decided every finding was a false positive). Caller
            # decides whether to engage the FP-resolution flow or fall through
            # to spawning claude on the original concerns.
            return "autofix_done_no_fix"
        return "concerns"

    def _classify_veria_for_head(self) -> str:
        """Verdict from Veria AI's check-run on HEAD."""
        candidates = _matching_check_runs(self._check_runs, "veria")
        run = _pick_latest(candidates)
        log.info(
            "veria %s: candidates=%d picked=%s | %s",
            self.pr.key, len(candidates), _describe_check_run(run),
            " ; ".join(_describe_check_run(r) for r in candidates) or "(none)",
        )
        if run is None or run.get("status") != "completed":
            return "none"
        # "skipped" means Veria opted not to review this PR — treat as clean.
        # "neutral" is NOT clean: Veria uses it for "found findings, non-blocking
        # severity" (e.g. risk 5/10), so it must still route to concerns.
        # "stale" means the run was superseded by a newer check suite, so it
        # shouldn't block — mirror _compute_cicd_verdict's handling.
        if run.get("conclusion") in ("success", "skipped", "stale"):
            return "clear"
        return "concerns"

    def _find_existing_greptile_mention(self) -> Optional[Comment]:
        """Latest @greptile or @greptileai mention posted by a non-greptile user
        after the HEAD commit timestamp. Used to avoid double-posting when a
        human already pinged greptile for HEAD."""
        # Without a known HEAD commit timestamp we can't tell which mentions are
        # stale, so refuse to adopt any to avoid acting on outdated feedback.
        if not self._latest_commit_at:
            return None
        latest: Optional[Comment] = None
        latest_at = 0.0
        for c in self._issue_comments:
            if _login_matches(c.user_login, GREPTILE_LOGINS):
                continue  # greptile's own summary contains its handle
            if not _GREPTILE_MENTION_RE.search(c.body or ""):
                continue
            ts = _parse_iso(c.created_at)
            if ts < self._latest_commit_at:
                continue
            if ts < latest_at:
                continue
            latest_at = ts
            latest = c
        return latest


# ----- utilities ----------------------------------------------------------


# Matches `@greptile` or `@greptileai` (case-insensitive) at a word boundary,
# so `@greptilevolution` or `@greptileai_other` don't match. The lookbehind
# also prevents matching inside email addresses like `user@greptile.com`.
_GREPTILE_MENTION_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])@greptile(?:ai)?\b")


def _parse_iso(ts: str) -> float:
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _matching_check_runs(
    runs: list[dict], name_substr: str, exclude: Optional[str] = None
) -> list[dict]:
    """All check-runs whose lowercased name contains `name_substr`.

    `exclude`, if given, filters out runs whose name also contains it — used to
    keep the main "Cursor Bugbot" run separate from "Cursor Bugbot Autofix".
    """
    needle = name_substr.lower()
    excl = exclude.lower() if exclude else None
    matched: list[dict] = []
    for r in runs:
        name = (r.get("name") or "").lower()
        if needle not in name:
            continue
        if excl and excl in name:
            continue
        matched.append(r)
    return matched


def _pick_latest(runs: list[dict]) -> Optional[dict]:
    """Pick the most recent check-run by started_at (falls back to completed_at)."""
    best: Optional[dict] = None
    best_ts = -1.0
    for r in runs:
        ts = _parse_iso(r.get("started_at", "") or r.get("completed_at", ""))
        if ts < best_ts:
            continue
        best_ts = ts
        best = r
    return best


def _describe_check_run(run: Optional[dict]) -> str:
    """One-line summary of a check-run for log lines — `id status/conclusion started`."""
    if run is None:
        return "none"
    return (
        f"id={run.get('id')} "
        f"name={run.get('name')!r} "
        f"status={run.get('status')} "
        f"conclusion={run.get('conclusion')} "
        f"started={run.get('started_at')}"
    )


def _extract_mention_comment_id(detail: str) -> Optional[int]:
    """detail looks like '#mention:12345 | ...'"""
    if not detail:
        return None
    try:
        head = detail.split("|", 1)[0].strip()
        if head.startswith("#mention:"):
            return int(head.split(":", 1)[1].strip())
    except (ValueError, IndexError):
        pass
    return None

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


ALLOWED_REPOS = {"BerriAI/litellm", "BerriAI/litellm-docs"}

# GitHub logins we treat as each bot. Lowercased substring match against `login`.
# Bugbot and veria verdicts now come from check-runs (see babysitter.py);
# only the greptile mention/reaction loop still needs a login match.
GREPTILE_LOGINS = ("greptile",)


PR_URL_RE = re.compile(r"^https?://github\.com/([^/]+/[^/]+)/pull/(\d+)/?$")


def parse_pr_url(url: str) -> Optional[tuple[str, int]]:
    """Return (repo, pr_number) if the URL is a supported PR URL, else None."""
    m = PR_URL_RE.match(url.strip())
    if not m:
        return None
    repo, num = m.group(1), int(m.group(2))
    if repo not in ALLOWED_REPOS:
        return None
    return repo, num


def subsystem_label(name: str) -> str:
    """Short label for UI status strings (internal keys stay unchanged)."""
    return "ci" if name == "cicd" else name


class SubState(str, Enum):
    """State of one of the four subsystems we track per PR."""
    UNKNOWN = "unknown"          # not yet evaluated for current commit
    WAITING_AUTOFIX = "waiting_autofix"  # bugbot autofix in progress
    WAITING_BOT = "waiting_bot"  # waiting for greptile reaction
    CLAUDE_RUNNING = "claude_running"    # a claude session is working on it
    DONE = "done"                # subsystem resolved for this commit
    ERROR = "error"              # unrecoverable


@dataclass
class Subsystem:
    state: SubState = SubState.UNKNOWN
    claude_session_id: Optional[str] = None
    error_reason: Optional[str] = None
    detail: str = ""             # human-readable detail
    # subsystem-specific bookkeeping
    last_mention_commit: Optional[str] = None   # greptile only
    # commit sha for which we last spawned a claude session (bugbot/veria/cicd).
    # Used as a respawn guard so we don't keep firing claude every tick while
    # waiting for a new commit to land in the GitHub API.
    last_spawn_commit: Optional[str] = None
    # transient: count of consecutive _poll_claude HTTP failures. Not persisted;
    # used to break out of CLAUDE_RUNNING when the session is unreachable.
    poll_failures: int = 0
    # epoch seconds when the current claude session was spawned; None when no
    # session is running. Persisted so elapsed time survives a restart.
    claude_started_at: Optional[float] = None
    # transient: epoch seconds of the last heartbeat log line emitted for the
    # current claude session. Reset when a session starts or ends.
    last_heartbeat_log_at: float = 0.0
    # epoch seconds of the most recent claude spawn for this subsystem,
    # regardless of session outcome. Survives session end (unlike
    # `claude_started_at`, which is cleared when the session completes).
    # Used by greptile to detect "did the bot say something new since we last
    # asked claude to weigh in?" and re-spawn if so. Persisted.
    claude_last_spawn_at: Optional[float] = None
    # number of claude sessions successfully spawned for the current commit
    # via the normal (non-interrupted-retry) path. Reset on commit change so
    # the per-commit cap stays meaningful. Persisted.
    claude_spawn_count: int = 0
    # number of claude sessions successfully spawned by the interrupted-retry
    # path (`_maybe_retry_interrupted`) for the current commit. Tracked
    # separately from `claude_spawn_count` so that a brief Anthropic backend
    # incident burning interrupted retries doesn't silently eat greptile's
    # per-commit follow-up budget. Reset on commit change. Persisted.
    interrupted_retry_count: int = 0
    # session id we last sent a "resume / emit your envelope" nudge to. Used
    # to cap nudges at one per session so a broken agent can't soak our poll
    # loop forever. Cleared on each new spawn. Persisted so a restart in the
    # middle of a nudged session doesn't lose track of the budget.
    claude_nudged_session_id: Optional[str] = None
    # Number of consecutive ticks the cicd verdict has been `failure` since
    # the last spawn (or since the counter was last reset by a non-failure
    # verdict). Used as the gate before respawning the cicd agent — we wait
    # until the failure is confirmed across multiple ticks so a single stale
    # poll right after the agent kicked off a re-run doesn't trigger an
    # immediate respawn. Only meaningful for the cicd subsystem; other
    # subsystems leave it at 0. Persisted.
    cicd_consecutive_failure_ticks: int = 0
    # Bugbot only. Commit sha for which we already posted a `bugbot run`
    # issue comment after Cursor Bugbot Autofix gave up with a "false
    # positive" verdict. Used to prevent re-posting `bugbot run` (and re-
    # replying on the same threads) every tick while waiting for the new
    # Cursor Bugbot check-run to land on this HEAD. Cleared on commit change
    # (reset()). Persisted.
    bugbot_run_posted_commit: Optional[str] = None
    # Bugbot only. GraphQL review-thread node ids ("PRRT_kwDO...") we have
    # already replied to + resolved as false-positive on this HEAD. Acts as a
    # per-thread idempotency guard so concurrent ticks (or a restart between
    # `reply` and `resolve`) can't double-post. Cleared on commit change.
    # Persisted.
    bugbot_fp_handled_thread_ids: List[str] = field(default_factory=list)
    # CICD only. Commit sha for which the cicd agent confirmed every failing
    # job's root cause was an unfixable infra error (AWS unauthorized-IAM-
    # action — see CICD_PROMPT + _on_cicd_done). When set, `_process_cicd`
    # short-circuits to DONE without re-evaluating the GitHub verdict, so we
    # don't respawn the agent every confirmation window to re-confirm the
    # same unfixable thing. Cleared on commit change (reset()). Persisted.
    cicd_ignored_commit: Optional[str] = None

    def reset(self) -> None:
        self.state = SubState.UNKNOWN
        self.claude_session_id = None
        self.error_reason = None
        self.detail = ""
        self.last_mention_commit = None
        self.last_spawn_commit = None
        self.poll_failures = 0
        self.claude_started_at = None
        self.last_heartbeat_log_at = 0.0
        self.claude_last_spawn_at = None
        self.claude_spawn_count = 0
        self.interrupted_retry_count = 0
        self.claude_nudged_session_id = None
        self.cicd_consecutive_failure_ticks = 0
        self.bugbot_run_posted_commit = None
        self.bugbot_fp_handled_thread_ids = []
        self.cicd_ignored_commit = None

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "claude_session_id": self.claude_session_id,
            "error_reason": self.error_reason,
            "detail": self.detail,
            "last_mention_commit": self.last_mention_commit,
            "last_spawn_commit": self.last_spawn_commit,
            "claude_started_at": self.claude_started_at,
            "claude_last_spawn_at": self.claude_last_spawn_at,
            "claude_spawn_count": self.claude_spawn_count,
            "interrupted_retry_count": self.interrupted_retry_count,
            "claude_nudged_session_id": self.claude_nudged_session_id,
            "cicd_consecutive_failure_ticks": self.cicd_consecutive_failure_ticks,
            "bugbot_run_posted_commit": self.bugbot_run_posted_commit,
            "bugbot_fp_handled_thread_ids": list(self.bugbot_fp_handled_thread_ids),
            "cicd_ignored_commit": self.cicd_ignored_commit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Subsystem":
        try:
            state = SubState(d.get("state", SubState.UNKNOWN.value))
        except ValueError:
            state = SubState.UNKNOWN
        fp_ids = d.get("bugbot_fp_handled_thread_ids") or []
        if not isinstance(fp_ids, list):
            fp_ids = []
        return cls(
            state=state,
            claude_session_id=d.get("claude_session_id"),
            error_reason=d.get("error_reason"),
            detail=d.get("detail", ""),
            last_mention_commit=d.get("last_mention_commit"),
            last_spawn_commit=d.get("last_spawn_commit"),
            claude_started_at=d.get("claude_started_at"),
            claude_last_spawn_at=d.get("claude_last_spawn_at"),
            claude_spawn_count=int(d.get("claude_spawn_count") or 0),
            interrupted_retry_count=int(d.get("interrupted_retry_count") or 0),
            claude_nudged_session_id=d.get("claude_nudged_session_id"),
            cicd_consecutive_failure_ticks=int(d.get("cicd_consecutive_failure_ticks") or 0),
            bugbot_run_posted_commit=d.get("bugbot_run_posted_commit"),
            bugbot_fp_handled_thread_ids=[str(x) for x in fp_ids],
            cicd_ignored_commit=d.get("cicd_ignored_commit"),
        )


@dataclass
class PRState:
    repo: str
    number: int
    title: str = ""
    html_url: str = ""
    head_branch: str = ""
    base_branch: str = ""
    last_commit_sha: str = ""
    # GitHub's mergeable_state for HEAD: "dirty" means conflicts with base.
    mergeable_state: str = ""
    # When this PR is an internal copy of a fork/non-litellm_* PR (see pr_copy),
    # these fields record the original PR + its fork branch so the fork_sync
    # subsystem can detect new commits on the upstream branch and merge them
    # into the copy. Empty on PRs that aren't copies.
    origin_repo: str = ""
    origin_number: int = 0
    origin_head_repo: str = ""   # e.g., "someuser/litellm" (the fork)
    origin_head_ref: str = ""    # e.g., "my-feature-branch"
    copy_branch: str = ""        # e.g., "litellm_my-feature-branch"
    fork_sync: Subsystem = field(default_factory=Subsystem)
    merge: Subsystem = field(default_factory=Subsystem)
    bugbot: Subsystem = field(default_factory=Subsystem)
    greptile: Subsystem = field(default_factory=Subsystem)
    veria: Subsystem = field(default_factory=Subsystem)
    cicd: Subsystem = field(default_factory=Subsystem)

    # transient/runtime
    last_polled_at: float = 0.0
    last_error: str = ""

    @property
    def key(self) -> str:
        return f"{self.repo}#{self.number}"

    @property
    def html_link(self) -> str:
        return self.html_url or f"https://github.com/{self.repo}/pull/{self.number}"

    def reset_subsystems(self) -> None:
        self.fork_sync.reset()
        self.merge.reset()
        self.bugbot.reset()
        self.greptile.reset()
        self.veria.reset()
        self.cicd.reset()

    @property
    def is_copy(self) -> bool:
        """True if this PR is an internal copy of an upstream fork PR (i.e.,
        fork_sync should run on it)."""
        return bool(self.origin_head_ref) and bool(self.copy_branch)

    def overall_status(self) -> str:
        # error takes precedence; include `fork_sync` and `merge` here so
        # unresolved fork-sync / merge issues surface in the main status even
        # though those subsystems are not shown in the "waiting on" list.
        errored = [
            name
            for name, sub in {
                "fork_sync": self.fork_sync,
                "merge": self.merge,
                **self._subs(),
            }.items()
            if sub.state == SubState.ERROR
        ]
        if errored:
            return "error"

        waiting: list[str] = []
        for name, sub in self._subs().items():
            if sub.state == SubState.DONE:
                continue
            if name == "bugbot" and sub.state == SubState.WAITING_AUTOFIX:
                waiting.append("bugbot autofix")
            elif name == "bugbot" and sub.state == SubState.CLAUDE_RUNNING:
                waiting.append("bugbot manual fix")
            else:
                waiting.append(subsystem_label(name))
        if not waiting:
            return "done"
        return "waiting on " + ", ".join(waiting)

    def table_status(self) -> str:
        """Compact status for the main table (detail screen uses overall_status)."""
        s = self.overall_status()
        if not s.startswith("waiting on "):
            return s
        parts: list[str] = []
        for name, sub in self._subs().items():
            if sub.state == SubState.DONE:
                continue
            if name == "bugbot" and sub.state == SubState.WAITING_AUTOFIX:
                parts.append("bugbot~")
            elif name == "bugbot" and sub.state == SubState.CLAUDE_RUNNING:
                parts.append("bugbot*")
            else:
                parts.append(subsystem_label(name))
        return "wait: " + ", ".join(parts)

    def _subs(self) -> dict[str, Subsystem]:
        # `fork_sync` and `merge` are intentionally excluded: they're pre-steps
        # that only fire on actual upstream drift / conflicts, not statuses the
        # user should see "waiting on" for healthy PRs.
        return {
            "bugbot": self.bugbot,
            "greptile": self.greptile,
            "veria": self.veria,
            "cicd": self.cicd,
        }

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "number": self.number,
            "title": self.title,
            "html_url": self.html_url,
            "head_branch": self.head_branch,
            "base_branch": self.base_branch,
            "last_commit_sha": self.last_commit_sha,
            "mergeable_state": self.mergeable_state,
            "origin_repo": self.origin_repo,
            "origin_number": self.origin_number,
            "origin_head_repo": self.origin_head_repo,
            "origin_head_ref": self.origin_head_ref,
            "copy_branch": self.copy_branch,
            "fork_sync": self.fork_sync.to_dict(),
            "merge": self.merge.to_dict(),
            "bugbot": self.bugbot.to_dict(),
            "greptile": self.greptile.to_dict(),
            "veria": self.veria.to_dict(),
            "cicd": self.cicd.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PRState":
        pr = cls(
            repo=d["repo"],
            number=int(d["number"]),
            title=d.get("title", ""),
            html_url=d.get("html_url", ""),
            head_branch=d.get("head_branch", ""),
            base_branch=d.get("base_branch", ""),
            last_commit_sha=d.get("last_commit_sha", ""),
            mergeable_state=d.get("mergeable_state", ""),
            origin_repo=d.get("origin_repo", "") or "",
            origin_number=int(d.get("origin_number") or 0),
            origin_head_repo=d.get("origin_head_repo", "") or "",
            origin_head_ref=d.get("origin_head_ref", "") or "",
            copy_branch=d.get("copy_branch", "") or "",
        )
        if isinstance(d.get("fork_sync"), dict):
            pr.fork_sync = Subsystem.from_dict(d["fork_sync"])
        if isinstance(d.get("merge"), dict):
            pr.merge = Subsystem.from_dict(d["merge"])
        if isinstance(d.get("bugbot"), dict):
            pr.bugbot = Subsystem.from_dict(d["bugbot"])
        if isinstance(d.get("greptile"), dict):
            pr.greptile = Subsystem.from_dict(d["greptile"])
        if isinstance(d.get("veria"), dict):
            pr.veria = Subsystem.from_dict(d["veria"])
        if isinstance(d.get("cicd"), dict):
            pr.cicd = Subsystem.from_dict(d["cicd"])
        return pr

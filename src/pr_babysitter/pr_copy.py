from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .github_api import GitHubClient


log = logging.getLogger(__name__)


LITELLM_REPO = "BerriAI/litellm"
LITELLM_OWNER = "BerriAI"
INTERNAL_STAGING_BASE = "litellm_internal_staging"
INTERNAL_BRANCH_PREFIX = "litellm_"


@dataclass
class ResolvedPR:
    repo: str
    number: int
    # True if `(repo, number)` is a freshly-created or pre-existing copy that
    # replaced the original URL the user submitted; lets the TUI surface a
    # "your PR was redirected to ..." notification.
    redirected_from: Optional[tuple[str, int]] = None
    # Origin-PR metadata, populated only when `redirected_from` is set. The
    # babysitter's fork_sync subsystem uses these to detect new commits on the
    # upstream branch and merge them into the copy.
    origin_head_repo: str = ""   # e.g. "someuser/litellm"
    origin_head_ref: str = ""    # the upstream branch name
    copy_branch: str = ""        # the litellm_* branch in BerriAI/litellm


def _needs_internal_copy(repo: str, head_repo: str, head_ref: str) -> bool:
    """A litellm PR needs an internal copy if it comes from a fork or from a
    branch that doesn't follow the `litellm_*` convention."""
    if repo != LITELLM_REPO:
        return False
    if not head_repo or not head_ref:
        # Defensive: missing head metadata (deleted fork, weird API state).
        # Can't safely create a copy with no sha, so don't try.
        return False
    is_fork = head_repo != LITELLM_REPO
    bad_prefix = not head_ref.startswith(INTERNAL_BRANCH_PREFIX)
    return is_fork or bad_prefix


async def resolve_to_babysit_target(
    github: GitHubClient, repo: str, number: int
) -> ResolvedPR:
    """If the submitted PR is a litellm PR from a fork or a non-`litellm_*`
    branch, create (or reuse) a copy PR into `litellm_internal_staging` and
    return its identity. Otherwise return `(repo, number)` unchanged.

    The copy branch lives in `BerriAI/litellm` and is named
    `litellm_<original-head-ref>`. GitHub stores forks in the same network as
    the parent, so creating a branch ref at the fork's HEAD sha works without
    any local git operations.
    """
    pr = await github.get_pr(repo, number)
    head = pr.get("head") or {}
    head_repo_obj = head.get("repo") or {}
    head_repo = head_repo_obj.get("full_name", "") or ""
    head_ref = head.get("ref", "") or ""
    head_sha = head.get("sha", "") or ""

    if not _needs_internal_copy(repo, head_repo, head_ref):
        return ResolvedPR(repo=repo, number=number)

    if not head_sha:
        raise RuntimeError(
            f"{repo}#{number}: cannot copy - PR head sha is missing "
            "(the fork or branch may have been deleted)"
        )

    # Avoid `litellm_litellm_*` when a fork branch already follows the
    # internal naming convention.
    if head_ref.startswith(INTERNAL_BRANCH_PREFIX):
        copy_branch = head_ref
    else:
        copy_branch = f"{INTERNAL_BRANCH_PREFIX}{head_ref}"

    existing = await github.list_open_prs(
        LITELLM_REPO,
        head=f"{LITELLM_OWNER}:{copy_branch}",
        base=INTERNAL_STAGING_BASE,
    )
    if existing:
        copy_pr = existing[0]
        copy_number = int(copy_pr["number"])
        log.info(
            "pr_copy %s#%d: reusing existing copy PR %s#%d (branch=%s)",
            repo, number, LITELLM_REPO, copy_number, copy_branch,
        )
        return ResolvedPR(
            repo=LITELLM_REPO,
            number=copy_number,
            redirected_from=(repo, number),
            origin_head_repo=head_repo,
            origin_head_ref=head_ref,
            copy_branch=copy_branch,
        )

    existing_ref = await github.get_branch_ref(LITELLM_REPO, copy_branch)
    if existing_ref is None:
        await github.create_branch_ref(LITELLM_REPO, copy_branch, head_sha)
        log.info(
            "pr_copy %s#%d: created branch %s @ %s",
            repo, number, copy_branch, head_sha[:12],
        )
    else:
        existing_sha = ((existing_ref.get("object") or {}).get("sha") or "")
        if existing_sha != head_sha:
            await github.update_branch_ref(
                LITELLM_REPO, copy_branch, head_sha, force=True
            )
            log.info(
                "pr_copy %s#%d: moved branch %s from %s to %s",
                repo, number, copy_branch,
                existing_sha[:12] or "?", head_sha[:12],
            )

    orig_url = pr.get("html_url") or f"https://github.com/{repo}/pull/{number}"
    orig_title = pr.get("title") or f"PR #{number}"
    title = f"[internal copy of #{number}] {orig_title}"
    body = (
        f"Automated copy of {orig_url} into `{INTERNAL_STAGING_BASE}` "
        f"for pr-babysitter.\n\n"
        f"Original head: `{head_repo}:{head_ref}` @ `{head_sha[:12]}`"
    )
    created = await github.create_pr(
        LITELLM_REPO,
        title=title,
        body=body,
        head=copy_branch,
        base=INTERNAL_STAGING_BASE,
    )
    copy_number = int(created["number"])
    log.info(
        "pr_copy %s#%d: created copy PR %s#%d (head=%s base=%s)",
        repo, number, LITELLM_REPO, copy_number,
        copy_branch, INTERNAL_STAGING_BASE,
    )
    return ResolvedPR(
        repo=LITELLM_REPO,
        number=copy_number,
        redirected_from=(repo, number),
        origin_head_repo=head_repo,
        origin_head_ref=head_ref,
        copy_branch=copy_branch,
    )

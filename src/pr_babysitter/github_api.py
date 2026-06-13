from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx


log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


@dataclass
class Comment:
    id: int
    user_login: str
    body: str
    created_at: str  # ISO8601
    # GitHub returns updated_at == created_at for un-edited comments; greptile
    # often edits its summary in place rather than posting a new comment, so we
    # need the updated_at to tell "the bot said something new" from staleness.
    updated_at: str = ""


@dataclass
class Reaction:
    content: str
    user_login: str


@dataclass
class ReviewThread:
    """A single review-comment thread on a PR.

    `thread_id` is the GraphQL node id (e.g. "PRRT_kwDO...") - required for the
    resolveReviewThread mutation. `latest_cursor_comment_*` carry the most
    recent cursor[bot] comment on the thread (None if cursor[bot] hasn't posted
    here), since that's the comment we'd reply to and inspect for the
    autofix's "determined this is a false positive" verdict.
    """
    thread_id: str
    is_resolved: bool
    path: str
    latest_cursor_comment_id: Optional[int]
    latest_cursor_comment_body: str


class GitHubClient:
    def __init__(self, pat: str):
        self._pat = pat
        self._client = httpx.AsyncClient(
            base_url=GITHUB_API,
            headers={
                "Authorization": f"Bearer {pat}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "pr-babysitter",
            },
            timeout=20.0,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: Optional[dict] = None) -> Any:
        r = await self._client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    async def _post(self, path: str, json: Optional[dict] = None) -> Any:
        r = await self._client.post(path, json=json)
        r.raise_for_status()
        return r.json() if r.text else None

    async def _patch(self, path: str, json: Optional[dict] = None) -> Any:
        r = await self._client.patch(path, json=json)
        r.raise_for_status()
        return r.json() if r.text else None

    async def _put(self, path: str, json: Optional[dict] = None) -> Any:
        r = await self._client.put(path, json=json)
        r.raise_for_status()
        return r.json() if r.text else None

    async def get_pr(self, repo: str, number: int) -> dict:
        return await self._get(f"/repos/{repo}/pulls/{number}")

    async def list_open_prs(self, repo: str, head: str, base: str) -> list[dict]:
        """`head` must be in the form `owner:branch`; matches GitHub's filter format."""
        return await self._get(
            f"/repos/{repo}/pulls",
            params={"state": "open", "head": head, "base": base, "per_page": 100},
        )

    async def get_branch_ref(self, repo: str, branch: str) -> Optional[dict]:
        try:
            return await self._get(f"/repos/{repo}/git/ref/heads/{branch}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def create_branch_ref(self, repo: str, branch: str, sha: str) -> dict:
        return await self._post(
            f"/repos/{repo}/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": sha},
        )

    async def update_branch_ref(
        self, repo: str, branch: str, sha: str, force: bool = True
    ) -> dict:
        return await self._patch(
            f"/repos/{repo}/git/refs/heads/{branch}",
            json={"sha": sha, "force": force},
        )

    async def create_pr(
        self, repo: str, title: str, body: str, head: str, base: str
    ) -> dict:
        return await self._post(
            f"/repos/{repo}/pulls",
            json={"title": title, "body": body, "head": head, "base": base},
        )

    async def merge_pr(
        self,
        repo: str,
        number: int,
        *,
        merge_method: str = "merge",
        sha: Optional[str] = None,
        commit_title: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> dict:
        """Merge a PR via PUT /repos/{repo}/pulls/{number}/merge.

        `sha`, when given, is GitHub's optimistic-concurrency guard: the merge
        is rejected (409 Conflict) if the PR's HEAD has advanced past it, so we
        never merge a commit we didn't actually vet. `merge_method` is one of
        "merge" / "squash" / "rebase" and must be enabled on the repo or GitHub
        returns 405. A PR that can't be merged yet (failing required checks,
        unmet branch protection, draft) also returns 405 — callers should treat
        that as retryable rather than fatal.
        """
        body: dict[str, Any] = {"merge_method": merge_method}
        if sha:
            body["sha"] = sha
        if commit_title is not None:
            body["commit_title"] = commit_title
        if commit_message is not None:
            body["commit_message"] = commit_message
        return await self._put(f"/repos/{repo}/pulls/{number}/merge", json=body)

    async def get_commit(self, repo: str, sha: str) -> dict:
        return await self._get(f"/repos/{repo}/commits/{sha}")

    async def compare_commits(self, repo: str, base: str, head: str) -> dict:
        """GET /repos/{repo}/compare/{base}...{head}. The response includes
        `ahead_by` / `behind_by` counts and the diverged commit lists.

        Both `base` and `head` must be reachable from `{repo}` (i.e., commits
        in the same network — forks of the same upstream qualify, since
        GitHub stores forks in the same git repository server-side).
        """
        return await self._get(f"/repos/{repo}/compare/{base}...{head}")

    async def list_issue_comments(self, repo: str, number: int) -> list[Comment]:
        """Top-level conversation comments on the PR (not review comments)."""
        out: list[Comment] = []
        page = 1
        while True:
            data = await self._get(
                f"/repos/{repo}/issues/{number}/comments",
                params={"per_page": 100, "page": page},
            )
            for c in data:
                out.append(
                    Comment(
                        id=c["id"],
                        user_login=(c.get("user") or {}).get("login", ""),
                        body=c.get("body") or "",
                        created_at=c["created_at"],
                        updated_at=c.get("updated_at") or c["created_at"],
                    )
                )
            if len(data) < 100:
                break
            page += 1
        return out

    async def get_issue_comment_reactions(self, repo: str, comment_id: int) -> list[Reaction]:
        data = await self._get(
            f"/repos/{repo}/issues/comments/{comment_id}/reactions",
            params={"per_page": 100},
        )
        return [
            Reaction(
                content=r.get("content", ""),
                user_login=(r.get("user") or {}).get("login", ""),
            )
            for r in data
        ]

    async def post_issue_comment(self, repo: str, number: int, body: str) -> Comment:
        data = await self._post(
            f"/repos/{repo}/issues/{number}/comments",
            json={"body": body},
        )
        return Comment(
            id=data["id"],
            user_login=(data.get("user") or {}).get("login", ""),
            body=data.get("body") or "",
            created_at=data["created_at"],
            updated_at=data.get("updated_at") or data["created_at"],
        )

    async def post_review_comment_reply(
        self, repo: str, pr_number: int, parent_comment_id: int, body: str
    ) -> Comment:
        """Reply to an existing PR review comment, threading the reply under it.

        Uses POST /repos/{repo}/pulls/{n}/comments/{id}/replies. The reply
        appears in the same review thread as the parent, which is what
        resolveReviewThread later acts on.
        """
        data = await self._post(
            f"/repos/{repo}/pulls/{pr_number}/comments/{parent_comment_id}/replies",
            json={"body": body},
        )
        return Comment(
            id=data["id"],
            user_login=(data.get("user") or {}).get("login", ""),
            body=data.get("body") or "",
            created_at=data["created_at"],
            updated_at=data.get("updated_at") or data["created_at"],
        )

    async def _graphql(self, query: str, variables: Optional[dict] = None) -> dict:
        # GitHub's GraphQL endpoint lives at /graphql on the same host as REST.
        # Errors come back as HTTP 200 with an "errors" array, so raise on
        # those explicitly rather than relying on raise_for_status alone.
        r = await self._client.post(
            "/graphql", json={"query": query, "variables": variables or {}}
        )
        r.raise_for_status()
        body = r.json()
        if body.get("errors"):
            raise RuntimeError(f"graphql error: {body['errors']}")
        return body.get("data") or {}

    async def list_review_threads(
        self, repo: str, pr_number: int
    ) -> list[ReviewThread]:
        """Every review thread on the PR, with the latest cursor[bot] comment
        on each (if any). Used to find unresolved threads where Cursor Bugbot
        Autofix posted a "determined this is a false positive" verdict so we
        can reply + resolve them programmatically.
        """
        owner, name = repo.split("/", 1)
        query = """
        query($owner: String!, $name: String!, $number: Int!, $after: String) {
          repository(owner: $owner, name: $name) {
            pullRequest(number: $number) {
              reviewThreads(first: 100, after: $after) {
                pageInfo { hasNextPage endCursor }
                nodes {
                  id
                  isResolved
                  path
                  comments(first: 50) {
                    nodes {
                      databaseId
                      body
                      author { login }
                    }
                  }
                }
              }
            }
          }
        }
        """
        out: list[ReviewThread] = []
        after: Optional[str] = None
        while True:
            data = await self._graphql(
                query, {"owner": owner, "name": name, "number": pr_number, "after": after}
            )
            pr = ((data.get("repository") or {}).get("pullRequest")) or {}
            rt = pr.get("reviewThreads") or {}
            for node in rt.get("nodes") or []:
                latest_cursor_id: Optional[int] = None
                latest_cursor_body: str = ""
                for c in (node.get("comments") or {}).get("nodes") or []:
                    login = ((c.get("author") or {}).get("login") or "").lower()
                    # GitHub returns the bot's login without the "[bot]" suffix
                    # in GraphQL (REST adds it), so match the prefix instead.
                    if "cursor" not in login:
                        continue
                    db_id = c.get("databaseId")
                    if db_id is None:
                        continue
                    # comments() is ordered chronologically; keep the latest.
                    latest_cursor_id = int(db_id)
                    latest_cursor_body = c.get("body") or ""
                out.append(
                    ReviewThread(
                        thread_id=node.get("id") or "",
                        is_resolved=bool(node.get("isResolved")),
                        path=node.get("path") or "",
                        latest_cursor_comment_id=latest_cursor_id,
                        latest_cursor_comment_body=latest_cursor_body,
                    )
                )
            page = rt.get("pageInfo") or {}
            if not page.get("hasNextPage"):
                break
            after = page.get("endCursor")
        return out

    async def resolve_review_thread(self, thread_id: str) -> None:
        """Mark a review thread as resolved via the resolveReviewThread mutation."""
        await self._graphql(
            """
            mutation($id: ID!) {
              resolveReviewThread(input: {threadId: $id}) {
                thread { id isResolved }
              }
            }
            """,
            {"id": thread_id},
        )

    async def list_check_runs(self, repo: str, commit_sha: str) -> list[dict]:
        out: list[dict] = []
        page = 1
        while True:
            data = await self._get(
                f"/repos/{repo}/commits/{commit_sha}/check-runs",
                params={"per_page": 100, "page": page},
            )
            runs = data.get("check_runs", [])
            out.extend(runs)
            if len(runs) < 100:
                break
            page += 1
        return out

    async def get_combined_status(self, repo: str, commit_sha: str) -> dict:
        # /commits/{sha}/status paginates `statuses` (default 30, max 100).
        # total_count can exceed a single page (e.g. CircleCI repos with
        # many parallel jobs), so failures may live past the first page
        # while the page itself looks all-green. Walk every page.
        first = await self._get(
            f"/repos/{repo}/commits/{commit_sha}/status",
            params={"per_page": 100, "page": 1},
        )
        statuses = list(first.get("statuses", []) or [])
        total = first.get("total_count", len(statuses))
        page = 2
        while len(statuses) < total:
            data = await self._get(
                f"/repos/{repo}/commits/{commit_sha}/status",
                params={"per_page": 100, "page": page},
            )
            chunk = data.get("statuses", []) or []
            if not chunk:
                break
            statuses.extend(chunk)
            page += 1
        first["statuses"] = statuses
        log.debug(
            "get_combined_status %s@%s: top_state=%s total=%s fetched=%s",
            repo, commit_sha[:12], first.get("state"), total, len(statuses),
        )
        return first

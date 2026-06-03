# pr-babysitter

A small TUI that babysits BerriAI/litellm (and litellm-docs) PRs until they're
green. It watches CI/CD, BugBot, Greptile, and Veria; @-mentions Greptile when
needed; and spawns [Claude Managed
Agents](https://platform.claude.com/docs/en/managed-agents/quickstart)
sessions to resolve real concerns.

## Install

```sh
pip install -e .
```

## Run

```sh
pr-babysitter
```

First launch prompts you for:

1. **GitHub PAT** — `repo` + `pull-requests` scopes. Used by the local TUI to
   read PRs and post the `@greptileai` mention, and passed through to each
   Managed-Agents session as the `authorization_token` for the mounted
   GitHub repository.
2. **Anthropic API key** — used by `pr-babysitter` to drive the Managed
   Agents API. Find or create one in the [Anthropic
   Console](https://console.anthropic.com/settings/keys).
3. **CircleCI PAT** (optional) — passed into each Managed-Agents session so
   the cicd subsystem can re-run flaky jobs and read CircleCI job logs. Create
   one in [CircleCI personal tokens](https://app.circleci.com/settings/user/tokens).

Config is saved at `~/.pr-babysitter/config.json`. The babysit list is
persisted at `~/.pr-babysitter/state.json` so PRs you were watching survive
restarts. A PR is dropped from the list automatically once GitHub reports it
merged (you can also remove one by hand with `d`).

## Keys

| Mode  | Key        | Action                              |
|-------|------------|-------------------------------------|
| input | type + ⏎   | start babysitting that PR URL        |
| input | esc        | switch to edit mode                  |
| edit  | ↑/↓        | navigate the PR list                 |
| edit  | ⏎          | open details (status per subsystem)  |
| edit  | d          | stop babysitting (no confirm)        |
| edit  | esc        | back to input mode                   |
| any   | ctrl-q     | quit                                 |

## How it talks to Claude

`pr-babysitter` is a thin client over the Anthropic [Managed
Agents](https://platform.claude.com/docs/en/managed-agents/quickstart) API:

1. On first spawn, it creates one shared **agent** (`POST /v1/agents`)
   declaring the model (`claude-opus-4-8`), a short system prompt, the full
   pre-built `agent_toolset_20260401`, and the [GitHub MCP
   server](https://platform.claude.com/docs/en/managed-agents/github). It
   also creates one shared **environment** (`POST /v1/environments`) with
   unrestricted networking. The two IDs are cached in `config.json` so we
   reuse them across runs.
2. For each subsystem that needs Claude (bugbot / greptile / veria / cicd),
   it creates a fresh **session** (`POST /v1/sessions`) mounting the PR's
   repository as a `github_repository` resource at `/workspace/repo` with
   the GitHub PAT as `authorization_token`, then sends the subsystem's
   prompt as a `user.message` event.
3. The babysitter polls `GET /v1/sessions/{id}` and reads back
   `agent.message` events via `GET /v1/sessions/{id}/events?types[]=agent.message`
   when the session goes `idle`. The agent's response carries a
   `{"error": ..., "ignore": ...}` envelope the babysitter parses to decide
   what to do next.

The full integration lives in `src/pr_babysitter/claude_cloud.py` — only
`spawn()` and `get()` are called from the rest of the codebase.

## Repo allowlist

By design only `BerriAI/litellm` and `BerriAI/litellm-docs` PR URLs are
accepted; everything else is rejected at the input prompt.

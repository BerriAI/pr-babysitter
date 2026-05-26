from __future__ import annotations

import asyncio
from typing import Optional

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.events import Resize
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
)

from .babysitter import PRBabysitter
from .claude_cloud import ClaudeCloudClient
from .config import Config, load_state, save_state
from .github_api import GitHubClient
from .pr_copy import resolve_to_babysit_target
from .state import PRState, SubState, parse_pr_url, subsystem_label


# ----- setup screen --------------------------------------------------------


class SetupScreen(Screen):
    """First-run setup. Captures GitHub PAT and Claude Cloud endpoint."""

    CSS = """
    SetupScreen {
        align: center middle;
    }
    #setup-card {
        width: 80;
        height: auto;
        border: round $accent;
        padding: 1 2;
    }
    #setup-card Label {
        margin-top: 1;
    }
    .setup-help {
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    BINDINGS = [Binding("ctrl+s", "submit", "Save", show=True)]

    def __init__(self) -> None:
        super().__init__()
        self._existing = Config.load()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="setup-card"):
            yield Label("[b]One-time setup[/b]")
            yield Label(
                "Paste your credentials below. Stored locally at ~/.pr-babysitter/config.json.",
                classes="setup-help",
            )
            yield Label("GitHub personal access token (repo, pull-requests):")
            yield Input(
                placeholder="ghp_...",
                password=True,
                value=self._existing.github_pat,
                id="github-pat",
            )
            yield Label("CircleCI personal access token (optional, for CI/CD re-runs):")
            yield Input(
                placeholder="CCIPAT_...",
                password=True,
                value=self._existing.circleci_pat,
                id="circleci-pat",
            )
            yield Label("Anthropic API key (for Claude Managed Agents):")
            yield Input(
                placeholder="sk-ant-...",
                password=True,
                value=self._existing.anthropic_api_key,
                id="anthropic-key",
            )
            yield Label(
                "On first run we'll create a Managed-Agents agent + environment "
                "and cache their IDs in config.json. Press [b]Ctrl+S[/b] to save and continue.",
                classes="setup-help",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#github-pat", Input).focus()

    def action_submit(self) -> None:
        cfg = Config()
        cfg.github_pat = self.query_one("#github-pat", Input).value.strip()
        cfg.circleci_pat = self.query_one("#circleci-pat", Input).value.strip()
        cfg.anthropic_api_key = self.query_one("#anthropic-key", Input).value.strip()
        # Carry over previously cached agent/environment IDs only when every
        # credential that gets baked into the agent's system prompt is
        # unchanged. The Anthropic API key obviously needs to match (the
        # cached IDs live in that workspace), and the GitHub + CircleCI PATs
        # are embedded as plaintext in the agent's system prompt - so
        # rotating either one must invalidate the cached agent so a fresh
        # POST /v1/agents picks up the new values.
        creds_unchanged = (
            cfg.anthropic_api_key == self._existing.anthropic_api_key
            and cfg.github_pat == self._existing.github_pat
            and cfg.circleci_pat == self._existing.circleci_pat
        )
        if creds_unchanged:
            cfg.agent_id = self._existing.agent_id
            cfg.agent_system_prompt_version = self._existing.agent_system_prompt_version
            cfg.agent_credentials_fingerprint = (
                self._existing.agent_credentials_fingerprint
            )
            cfg.environment_id = self._existing.environment_id
        elif cfg.anthropic_api_key == self._existing.anthropic_api_key:
            # Same Anthropic key, but a PAT rotated. Keep the environment
            # (it has no credentials embedded) but drop the agent so the
            # next bootstrap recreates it with the fresh PATs.
            cfg.environment_id = self._existing.environment_id
        cfg.sync_agent_credentials_fingerprint()
        if not cfg.github_pat:
            self.notify("GitHub PAT is required.", severity="error")
            return
        if not cfg.anthropic_api_key:
            self.notify("Anthropic API key is required.", severity="error")
            return
        cfg.save()
        app: "PRBabysitterApp" = self.app  # type: ignore[assignment]
        app.config = cfg
        app._init_clients()
        app.pop_screen()
        app.push_screen(MainScreen())


# ----- detail screen -------------------------------------------------------


class DetailScreen(ModalScreen):
    CSS = """
    DetailScreen {
        align: center middle;
    }
    #detail-card {
        width: 100;
        height: auto;
        border: round $accent;
        padding: 1 2;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Back", show=True),
    ]

    def __init__(self, pr: PRState) -> None:
        super().__init__()
        self.pr = pr

    def compose(self) -> ComposeResult:
        pr = self.pr
        with VerticalScroll(id="detail-card"):
            yield Label(f"[b]{pr.title or pr.key}[/b]")
            yield Static(f"[link='{pr.html_link}']{pr.html_link}[/link]")
            yield Static(f"Branch: [b]{pr.head_branch or '?'}[/b]")
            yield Static(f"HEAD: [b]{(pr.last_commit_sha or '?')[:12]}[/b]")
            yield Static(f"Overall: [b]{pr.overall_status()}[/b]")
            yield Label("")
            yield Label("[b]Subsystems[/b]")
            rows: list[tuple[str, "Subsystem"]] = []
            # Only surface the merge row when conflict resolution is actually
            # in flight or errored - otherwise it's just noise.
            if pr.merge.state in (SubState.CLAUDE_RUNNING, SubState.ERROR):
                rows.append(("merge", pr.merge))
            rows.extend([
                ("bugbot", pr.bugbot),
                ("greptile", pr.greptile),
                ("veria", pr.veria),
                (subsystem_label("cicd"), pr.cicd),
            ])
            for name, sub in rows:
                line = f"[b]{name:8}[/b] {sub.state.value:18} {sub.detail}"
                if sub.error_reason:
                    line += f"  [red]reason:[/red] {sub.error_reason}"
                yield Static(line)
            if pr.last_error:
                yield Label("")
                yield Static(f"[red]Last error:[/red] {pr.last_error}")
            yield Label("")
            yield Label("[dim]esc to close[/dim]")

    def action_close(self) -> None:
        self.app.pop_screen()


# ----- main screen ---------------------------------------------------------


class MainScreen(Screen):
    # Fixed widths for status (compact `table_status()`) and link (full GitHub URL).
    # Title fills whatever horizontal space remains after padding/chrome.
    _STATUS_COL_WIDTH = 34
    _LINK_COL_WIDTH = 52

    CSS = """
    MainScreen {
        layout: vertical;
    }
    #pr-table {
        height: 1fr;
    }
    #input-row {
        height: auto;
        dock: bottom;
        padding: 0 1;
        border-top: solid $accent;
    }
    #mode-line {
        height: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "toggle_mode", "Toggle mode", show=True),
        Binding("d", "delete_selected", "Delete", show=False),
        Binding("ctrl+q", "quit_app", "Quit", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        # mode: "input" (cursor at bottom Input) | "edit" (table focused)
        self._mode = "input"
        # row_key (str = pr.key) -> babysitter
        self._babysitters: dict[str, PRBabysitter] = {}
        # "repo#number" of URLs currently being resolved (copy-PR creation in
        # flight). Prevents double-submission from racing two copy PRs into
        # existence for the same original.
        self._resolving: set[str] = set()
        # Maps an origin "repo#number" (the URL the user submitted) to the
        # resolved babysitter key when `pr_copy` redirected the PR to an
        # internal copy. Lets the early duplicate-submission guard catch a
        # resubmission of the same fork URL without paying for a full
        # `resolve_to_babysit_target` round-trip.
        self._origin_to_resolved: dict[str, str] = {}
        # Last persisted state payload; used to suppress redundant disk
        # writes when nothing observable changed since the previous tick.
        self._last_persisted_state: Optional[dict] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        table = DataTable(id="pr-table", cursor_type="row", zebra_stripes=True)
        table.add_column("Status", key="status", width=self._STATUS_COL_WIDTH)
        table.add_column("Title", key="title", width=20)
        table.add_column("Link", key="link", width=self._LINK_COL_WIDTH)
        yield table
        with Container(id="input-row"):
            yield Static("[dim]mode: input  •  paste a PR URL and press Enter  •  esc → edit[/dim]", id="mode-line")
            yield Input(placeholder="https://github.com/BerriAI/litellm/pull/...", id="pr-url-input")
        yield Footer()

    def on_mount(self) -> None:
        self.set_focus(self.query_one("#pr-url-input", Input))
        self._restore_state()
        self.call_after_refresh(self._fit_table_columns)

    @on(Resize)
    def _on_resize(self, _: Resize) -> None:
        self._fit_table_columns()

    def _fit_table_columns(self) -> None:
        table = self.query_one("#pr-table", DataTable)
        total = table.size.width
        if total <= 0:
            return
        pad = table.cell_padding * 2
        chrome = 2
        title_w = max(
            16,
            total
            - chrome
            - (self._STATUS_COL_WIDTH + pad)
            - (self._LINK_COL_WIDTH + pad)
            - pad,
        )
        for col_key, column in table.columns.items():
            name = col_key.value
            if name == "status":
                column.width = self._STATUS_COL_WIDTH
            elif name == "title":
                column.width = title_w
            elif name == "link":
                column.width = self._LINK_COL_WIDTH
            column.auto_width = False
        table.refresh()

    # ----- mode handling ----------------------------------------------

    def action_toggle_mode(self) -> None:
        if self._mode == "input":
            self._enter_edit_mode()
        else:
            self._enter_input_mode()

    def _enter_input_mode(self) -> None:
        self._mode = "input"
        self.query_one("#mode-line", Static).update(
            "[dim]mode: input  •  paste a PR URL and press Enter  •  esc → edit[/dim]"
        )
        self.set_focus(self.query_one("#pr-url-input", Input))

    def _enter_edit_mode(self) -> None:
        table = self.query_one("#pr-table", DataTable)
        if table.row_count == 0:
            self.notify("No PRs to edit. Paste one below first.", timeout=2)
            return
        self._mode = "edit"
        self.query_one("#mode-line", Static).update(
            "[dim]mode: edit  •  ↑/↓ navigate  •  enter for details  •  d to delete  •  esc → input[/dim]"
        )
        table.cursor_coordinate = Coordinate(0, 0)
        self.set_focus(table)

    # ----- input submission -------------------------------------------

    @on(DataTable.RowSelected, "#pr-table")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        if self._mode != "edit":
            return
        key = event.row_key.value if event.row_key else None
        if not key:
            return
        bs = self._babysitters.get(key)
        if bs:
            self.app.push_screen(DetailScreen(bs.pr))

    @on(Input.Submitted, "#pr-url-input")
    def _on_url_submitted(self, event: Input.Submitted) -> None:
        url = event.value.strip()
        if not url:
            return
        parsed = parse_pr_url(url)
        if not parsed:
            self.notify(
                "Only BerriAI/litellm and BerriAI/litellm-docs PR URLs are supported.",
                severity="error",
                timeout=3,
            )
            return
        repo, number = parsed
        origin_key = f"{repo}#{number}"
        if origin_key in self._resolving:
            self.notify(f"{origin_key} is already being resolved.", timeout=2)
            event.input.value = ""
            return
        if origin_key in self._babysitters:
            self.notify(f"{origin_key} is already being babysat.", timeout=2)
            event.input.value = ""
            return
        resolved_key = self._origin_to_resolved.get(origin_key)
        if resolved_key and resolved_key in self._babysitters:
            self.notify(
                f"{origin_key} is already being babysat as {resolved_key}.",
                timeout=2,
            )
            event.input.value = ""
            return
        event.input.value = ""
        self._resolving.add(origin_key)
        asyncio.create_task(self._resolve_and_add(repo, number, url))

    async def _resolve_and_add(self, repo: str, number: int, url: str) -> None:
        origin_key = f"{repo}#{number}"
        app: "PRBabysitterApp" = self.app  # type: ignore[assignment]
        try:
            resolved = await resolve_to_babysit_target(app.github, repo, number)
        except Exception as e:
            self.notify(
                f"Failed to set up internal copy for {origin_key}: {e}",
                severity="error",
                timeout=5,
            )
            return
        finally:
            self._resolving.discard(origin_key)
        if resolved.redirected_from:
            self.notify(
                f"{origin_key} redirected to copy "
                f"{resolved.repo}#{resolved.number} on litellm_internal_staging.",
                timeout=4,
            )
            html_url = f"https://github.com/{resolved.repo}/pull/{resolved.number}"
        else:
            html_url = url
        pr = PRState(repo=resolved.repo, number=resolved.number, html_url=html_url)
        if resolved.redirected_from:
            origin_repo_full, origin_number = resolved.redirected_from
            pr.origin_repo = origin_repo_full
            pr.origin_number = origin_number
            pr.origin_head_repo = resolved.origin_head_repo
            pr.origin_head_ref = resolved.origin_head_ref
            pr.copy_branch = resolved.copy_branch
            self._origin_to_resolved[origin_key] = pr.key
        if pr.key in self._babysitters:
            self.notify(f"{pr.key} is already being babysat.", timeout=2)
            return
        self._add_pr(pr)

    # ----- table actions ----------------------------------------------

    def action_delete_selected(self) -> None:
        if self._mode != "edit":
            return
        key = self._selected_key()
        if key:
            self._remove_pr(key)

    def action_quit_app(self) -> None:
        self.app.exit()

    def _selected_key(self) -> Optional[str]:
        table = self.query_one("#pr-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:
            return None
        return row_key

    # ----- pr lifecycle -----------------------------------------------

    def _add_pr(self, pr: PRState) -> None:
        app: "PRBabysitterApp" = self.app  # type: ignore[assignment]
        bs = PRBabysitter(
            pr=pr,
            github=app.github,
            claude=app.claude,
            poll_interval=app.config.poll_interval_seconds,
            on_change=lambda: self.call_from_thread_safe(self._refresh_row, pr.key),
        )
        self._babysitters[pr.key] = bs
        table = self.query_one("#pr-table", DataTable)
        table.add_row(
            pr.table_status(),
            pr.title or pr.key,
            f"[link='{pr.html_link}']{pr.html_link}[/link]",
            key=pr.key,
        )
        bs.start()
        self._persist_state()

    def _remove_pr(self, key: str) -> None:
        bs = self._babysitters.pop(key, None)
        if bs:
            asyncio.create_task(bs.stop())
        # Drop any origin->resolved entries that pointed to this babysitter so
        # a future submission of the same origin URL doesn't get rejected by
        # the early duplicate guard against a now-removed key.
        self._origin_to_resolved = {
            o: r for o, r in self._origin_to_resolved.items() if r != key
        }
        table = self.query_one("#pr-table", DataTable)
        try:
            table.remove_row(key)
        except Exception:
            pass
        self._persist_state()
        if table.row_count == 0 and self._mode == "edit":
            self._enter_input_mode()

    def _refresh_row(self, key: str) -> None:
        bs = self._babysitters.get(key)
        if not bs:
            return
        pr = bs.pr
        table = self.query_one("#pr-table", DataTable)
        try:
            table.update_cell(key, "status", pr.table_status())
            table.update_cell(key, "title", pr.title or pr.key)
            table.update_cell(key, "link", f"[link='{pr.html_link}']{pr.html_link}[/link]")
        except Exception:
            pass
        self._persist_state()

    # `call_from_thread_safe` shim: babysitter callbacks run in the asyncio loop
    # already, so we can just invoke directly. The name is intentional so we
    # could swap to `call_from_thread` if we ever move callbacks off-loop.
    def call_from_thread_safe(self, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception:
            pass

    # ----- persistence -------------------------------------------------

    def _persist_state(self) -> None:
        state = {"prs": [bs.pr.to_dict() for bs in self._babysitters.values()]}
        # Skip the disk write when nothing has changed since the last persist.
        # `_refresh_row` calls this on every tick per PR, so without a guard
        # we'd rewrite the entire state file N times per poll interval even
        # when no observable field changed.
        if state == self._last_persisted_state:
            return
        try:
            save_state(state)
        except Exception:
            return
        self._last_persisted_state = state

    def _restore_state(self) -> None:
        state = load_state()
        for pr_dict in state.get("prs", []):
            try:
                pr = PRState.from_dict(pr_dict)
            except Exception:
                continue
            if pr.key in self._babysitters:
                continue
            self._add_pr(pr)


# ----- app -----------------------------------------------------------------


class PRBabysitterApp(App):
    TITLE = "PR Babysitter"
    CSS = ""

    def __init__(self) -> None:
        super().__init__()
        self.config = Config.load()
        self.github: GitHubClient = None  # type: ignore[assignment]
        self.claude: ClaudeCloudClient = None  # type: ignore[assignment]

    def on_mount(self) -> None:
        if not self.config.is_complete():
            self.push_screen(SetupScreen())
        else:
            self._init_clients()
            self.push_screen(MainScreen())

    def _init_clients(self) -> None:
        self.github = GitHubClient(self.config.github_pat)
        self.claude = ClaudeCloudClient(
            api_key=self.config.anthropic_api_key,
            agent_id=self.config.agent_id,
            agent_version=self.config.agent_system_prompt_version,
            environment_id=self.config.environment_id,
            github_pat=self.config.github_pat,
            circleci_pat=self.config.circleci_pat,
            on_ids_updated=self._persist_managed_agents_ids,
        )

    def _persist_managed_agents_ids(
        self, agent_id: str, agent_version: str, environment_id: str
    ) -> None:
        self.config.agent_id = agent_id
        self.config.agent_system_prompt_version = agent_version
        self.config.environment_id = environment_id
        self.config.sync_agent_credentials_fingerprint()
        try:
            self.config.save()
        except Exception:
            pass

    async def on_unmount(self) -> None:
        # stop any active babysitter tasks before tearing down the HTTP
        # clients they use, otherwise their next tick would hit closed
        # clients and spam errors through on_change callbacks.
        for screen in list(self.screen_stack):
            if isinstance(screen, MainScreen):
                stops = [bs.stop() for bs in list(screen._babysitters.values())]
                screen._babysitters.clear()
                if stops:
                    await asyncio.gather(*stops, return_exceptions=True)
        if self.github is not None:
            try:
                await self.github.close()
            except Exception:
                pass
        if self.claude is not None:
            try:
                await self.claude.close()
            except Exception:
                pass

    async def action_quit(self) -> None:
        self.exit()

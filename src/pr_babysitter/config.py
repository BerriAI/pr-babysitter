from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def agent_credentials_fingerprint(github_pat: str, circleci_pat: str) -> str:
    """Hash of PATs embedded in the managed agent system prompt."""
    digest = hashlib.sha256()
    digest.update(github_pat.encode("utf-8"))
    digest.update(b"\0")
    digest.update(circleci_pat.encode("utf-8"))
    return digest.hexdigest()[:32]


def anthropic_workspace_fingerprint(anthropic_api_key: str) -> str:
    """Hash of the Anthropic API key. Used to invalidate cached agent_id and
    environment_id when the key changes (those IDs live in a specific
    workspace and are stale against any other key)."""
    digest = hashlib.sha256()
    digest.update(anthropic_api_key.encode("utf-8"))
    return digest.hexdigest()[:32]


CONFIG_DIR = Path(os.environ.get("PR_BABYSITTER_HOME", str(Path.home() / ".pr-babysitter")))
CONFIG_PATH = CONFIG_DIR / "config.json"
STATE_PATH = CONFIG_DIR / "state.json"


def _write_text_secure(path: Path, text: str) -> None:
    """Write `text` to `path` with mode 0o600 from the moment the file is
    created, avoiding the TOCTOU window between `Path.write_text` (which
    honours the process umask, typically 0o644) and a subsequent `os.chmod`.
    Used for files that contain PATs or other credentials.

    On platforms that support `os.fchmod` we also tighten the descriptor
    before writing, so an existing file (e.g. one that was created by an
    older babysitter build with default umask) is locked down before the
    sensitive content lands on disk."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
    except (AttributeError, OSError):
        pass
    with os.fdopen(fd, "w") as f:
        f.write(text)


@dataclass
class Config:
    github_pat: str = ""
    circleci_pat: str = ""
    anthropic_api_key: str = ""
    # Cached IDs for the Managed-Agents resources we lazily create on first
    # spawn. Persisted so we don't recreate the agent + environment on every
    # launch. `agent_system_prompt_version` invalidates a stale agent when
    # the in-code system prompt evolves.
    agent_id: str = ""
    agent_system_prompt_version: str = ""
    # Fingerprint of github_pat + circleci_pat when `agent_id` was created.
    agent_credentials_fingerprint: str = ""
    # Fingerprint of anthropic_api_key when `agent_id` / `environment_id`
    # were created. Used to drop both IDs when the key is rotated outside
    # the SetupScreen flow (e.g., manual config edit), since they belong
    # to whichever workspace the old key authenticated.
    anthropic_workspace_fingerprint: str = ""
    environment_id: str = ""
    poll_interval_seconds: int = 30

    def sync_agent_credentials_fingerprint(self) -> None:
        self.agent_credentials_fingerprint = agent_credentials_fingerprint(
            self.github_pat, self.circleci_pat
        )
        self.anthropic_workspace_fingerprint = anthropic_workspace_fingerprint(
            self.anthropic_api_key
        )

    def invalidate_stale_agent(self) -> bool:
        """Drop cached IDs that no longer match the credentials in config.

        Two independent triggers:
          - PATs changed -> agent_id is stale (system prompt embeds old PATs).
          - Anthropic API key changed -> agent_id AND environment_id are stale
            (they belong to the old key's workspace).
        Returns True when any field was cleared.
        """
        if not self.agent_id and not self.environment_id:
            return False
        changed = False
        current_workspace = anthropic_workspace_fingerprint(self.anthropic_api_key)
        if self.anthropic_workspace_fingerprint != current_workspace:
            # Both IDs live in the old workspace; clear them.
            self.agent_id = ""
            self.agent_system_prompt_version = ""
            self.environment_id = ""
            changed = True
        else:
            current_creds = agent_credentials_fingerprint(
                self.github_pat, self.circleci_pat
            )
            if self.agent_id and self.agent_credentials_fingerprint != current_creds:
                self.agent_id = ""
                self.agent_system_prompt_version = ""
                changed = True
        return changed

    @classmethod
    def load(cls) -> "Config":
        if not CONFIG_PATH.exists():
            return cls()
        try:
            data = json.loads(CONFIG_PATH.read_text())
        except Exception:
            return cls()
        cfg = cls(
            github_pat=data.get("github_pat", ""),
            circleci_pat=data.get("circleci_pat", ""),
            anthropic_api_key=data.get("anthropic_api_key", ""),
            agent_id=data.get("agent_id", ""),
            agent_system_prompt_version=data.get("agent_system_prompt_version", ""),
            agent_credentials_fingerprint=data.get("agent_credentials_fingerprint", ""),
            anthropic_workspace_fingerprint=data.get("anthropic_workspace_fingerprint", ""),
            environment_id=data.get("environment_id", ""),
            poll_interval_seconds=data.get("poll_interval_seconds", 30),
        )
        if cfg.invalidate_stale_agent():
            cfg.sync_agent_credentials_fingerprint()
            try:
                cfg.save()
            except OSError:
                pass
        return cfg

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        # Open with 0o600 from the start so the PATs we're about to write
        # are never visible to other users on shared systems (the default
        # umask would otherwise create the file world-readable until the
        # subsequent chmod tightens it).
        _write_text_secure(CONFIG_PATH, json.dumps(asdict(self), indent=2))

    def is_complete(self) -> bool:
        return bool(self.github_pat) and bool(self.anthropic_api_key)


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"prs": []}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"prs": []}


def save_state(state: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _write_text_secure(STATE_PATH, json.dumps(state, indent=2))

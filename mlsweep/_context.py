"""Named client contexts: which manager, token and campaign a command talks to.

A context lives in the client config file, ``~/.config/mlsweep/config.toml``
(``$XDG_CONFIG_HOME`` is honored, and ``$MLSWEEP_CONFIG`` names another file)::

    current = "lab"

    [contexts.lab]
    manager = "http://lab-box:7891"
    token_file = "~/.mlsweep/lab.token"   # or token_env = "LAB_MLSWEEP_TOKEN"
    campaign = "lr-study"

A token is never stored in the file itself.  ``token_file`` names a file that
holds it, and ``token_env`` names an environment variable.

Precedence, highest first:

1. ``--manager``, ``--token`` and ``--campaign`` on the command line.
2. A context named by ``--context NAME`` or ``$MLSWEEP_CONTEXT``.
3. ``$MLSWEEP_MANAGER``, ``$MLSWEEP_TOKEN`` and ``$MLSWEEP_CAMPAIGN``.
4. The config's ``current`` context, unless ``$MLSWEEP_MANAGER`` is set, since its
   token and campaign belong to its own manager.
5. ``http://localhost:7891``, ``$MLSWEEP_DIR/manager.token`` and the default campaign.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[import-not-found,unused-ignore]  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef,unused-ignore]  # Python < 3.11

from mlsweep._shared import validate_campaign

_FIELDS = ("manager", "token_file", "token_env", "campaign")


def config_path() -> Path:
    """Where the client config file is (it may not exist)."""
    if os.environ.get("MLSWEEP_CONFIG"):
        return Path(os.environ["MLSWEEP_CONFIG"]).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "mlsweep" / "config.toml"


@dataclass
class Context:
    name: str
    manager: str | None = None
    token_file: str | None = None
    token_env: str | None = None
    campaign: str | None = None

    def token(self) -> str | None:
        """The token this context names, or None if it names none (or it can't be read)."""
        if self.token_env:
            value = os.environ.get(self.token_env, "").strip()
            if value:
                return value
        if self.token_file:
            try:
                return Path(self.token_file).expanduser().read_text().strip() or None
            except OSError:
                return None
        return None

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("name", *_FIELDS) if getattr(self, k) is not None}


class ContextError(ValueError):
    pass


def load_config(path: Path | None = None) -> dict[str, Any]:
    """The parsed config file, or {} if there is none.  Raises ContextError if it is malformed."""
    path = path or config_path()
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ContextError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data.get("contexts", {}), dict):
        raise ContextError(f"{path}: 'contexts' must be a table")
    return dict(data)


def contexts(config: dict[str, Any]) -> dict[str, Context]:
    """Every context in *config*, by name.  Raises ContextError on a bad entry."""
    out = {}
    for name, raw in config.get("contexts", {}).items():
        if not isinstance(raw, dict):
            raise ContextError(f"context {name!r} must be a table")
        unknown = sorted(set(raw) - set(_FIELDS))
        if unknown:
            hint = " (keep the token in a file: token_file)" if "token" in unknown else ""
            raise ContextError(f"context {name!r} has unknown key(s) {unknown}{hint}")
        if not all(isinstance(v, str) for v in raw.values()):
            raise ContextError(f"context {name!r}: every value must be a string")
        out[name] = Context(name=name, **raw)
    return out


def get_context(name: str, config: dict[str, Any] | None = None) -> Context:
    found = contexts(load_config() if config is None else config)
    if name not in found:
        known = ", ".join(sorted(found)) or "none; add one with `mlsweep context add`"
        raise ContextError(f"no context named {name!r} in {config_path()} (known: {known})")
    return found[name]


def active_context() -> tuple[Context | None, bool]:
    """``(context, explicit)``: the context named by $MLSWEEP_CONTEXT (explicit), else the
    config's current one when $MLSWEEP_MANAGER is unset, else None.

    An unknown $MLSWEEP_CONTEXT is an error; a broken config with nothing asking
    for a context only warns, so commands keep working without it.
    """
    name = os.environ.get("MLSWEEP_CONTEXT")
    if name:
        return get_context(name), True
    if os.environ.get("MLSWEEP_MANAGER"):
        return None, False
    try:
        config = load_config()
        current = config.get("current")
        if not isinstance(current, str) or not current:
            return None, False
        return get_context(current, config), False
    except ContextError as exc:
        print(f"mlsweep: warning: ignoring the current context: {exc}", file=sys.stderr)
        return None, False


# ── Writing the config ─────────────────────────────────────────────────────────


def _toml_str(value: str) -> str:
    return json.dumps(value)  # a JSON string is a valid TOML basic string


def _toml_key(name: str) -> str:
    return name if name.replace("-", "").replace("_", "").isalnum() else _toml_str(name)


def save_config(config: dict[str, Any], path: Path | None = None) -> Path:
    """Write *config* (``current`` plus ``contexts``) back, replacing the file atomically."""
    path = path or config_path()
    lines = ["# mlsweep client contexts; see `mlsweep context --help`.", ""]
    if config.get("current"):
        lines += [f"current = {_toml_str(config['current'])}", ""]
    for name, ctx in sorted(contexts(config).items()):
        lines.append(f"[contexts.{_toml_key(name)}]")
        lines += [f"{k} = {_toml_str(getattr(ctx, k))}" for k in _FIELDS if getattr(ctx, k) is not None]
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, path)
    return path


def validate_context_name(name: str) -> str:
    try:
        return validate_campaign(name)  # same rules: short, filesystem- and URL-safe
    except ValueError:
        raise ContextError(f"bad context name {name!r}: use up to 128 letters, digits, '-' and '_'") from None

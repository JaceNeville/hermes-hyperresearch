"""Hermes Agent runtime — install, model tiers, and subagent spawning.

This fork runs the hyperresearch pipeline on Hermes Agent
(https://github.com/NousResearch/hermes-agent) instead of Claude Code or Codex.

Design (chosen so upstream template changes merge cleanly):

* **Zero template edits.** Step files, the entry skill, and agent prompts are
  rendered with upstream's `platform == "codex"` branch (the closest runtime:
  step procedures are plain files the orchestrator reads, subagents cannot
  spawn subagents, file edits are patches) and then passed through
  `translate()`, a small codex -> hermes vocabulary table. A test asserts no
  Codex-specific token survives translation, so upstream wording changes that
  the table misses fail CI instead of shipping a confusing prompt.

* **Subagents are separate `hermes chat` processes**, launched by
  `hpr hermes spawn`. Hermes' built-in `delegate_task` inherits the parent's
  model and caps concurrency, which defeats per-role model routing; a process
  per subagent gives each role its own model, toolsets, and token ledger.

* **Tool locks are real here.** Roles without Bash get no `terminal` toolset;
  only roles that search get `web`. (Codex can only ask nicely.)

* **Model routing lives in one file**, `.hyperresearch/hermes.toml`, as tiers
  (bulk / analysis / synthesis) mapped to roles. It is a separate file because
  `VaultConfig.save()` rewrites config.toml and drops unknown sections.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

HERMES_DIR = Path(".hyperresearch") / "hermes"
STEPS_DIR = HERMES_DIR / "steps"
AGENTS_DIR = HERMES_DIR / "agents"
ENTRY_SKILL = HERMES_DIR / "SKILL.md"
BATCHES_DIR = HERMES_DIR / "batches"
LOGS_DIR = HERMES_DIR / "logs"
CONFIG_FILE = Path(".hyperresearch") / "hermes.toml"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_TOML = """\
# hyperresearch on Hermes: model routing and runtime limits.
#
# Edit this file, then run `hpr hermes install` to re-render prompts.
# Secrets never go here: providers read API keys from the environment.

[hermes]
# Tier cap for runs that don't ask for one. "light" (1 -> 2 -> 10 -> 15 -> 16)
# is the cheap default; "full" runs all 16 steps; "auto" lets step 1 decide.
default_tier = "light"
# Tier the orchestrator (the process that sequences the steps) runs on.
orchestrator_tier = "analysis"
# Concurrent subagent processes, memory-aware: a new one starts only while
# MemAvailable stays above reserve_mb + per_agent_mb. Each `hermes chat` is
# ~220 MB RSS; the reserve protects the rest of the host (gateway, OS).
max_parallel = 6
min_parallel = 1
per_agent_mb = 260
reserve_mb = 700
# ICM runs stop (blocked, resumable) once estimated spend passes this, in USD.
# Needs [hermes.prices]. 0 = no ceiling.
max_cost_usd = 0
# ICM step 14.5: code splits cite-checking into batches of this many pairs.
cite_batch = 30
# Hard wall-clock cap per subagent, seconds.
spawn_timeout_s = 1800
# Tool-call iteration cap per subagent.
max_turns = 120
# Extra `hermes chat` flags for every process, e.g. ["--yolo"].
extra_args = []

# A tier is a provider + model. Swap models here as better ones ship.
[hermes.tiers.bulk]
provider = "anthropic"
model = "claude-haiku-4-5"

[hermes.tiers.analysis]
provider = "anthropic"
model = "claude-sonnet-5-5"

[hermes.tiers.synthesis]
provider = "anthropic"
model = "claude-opus-5-5"

# ICM mode (`hpr hermes icm`): pipeline step -> tier for that stage's own
# session. "<step>@<tier>" overrides one run tier. Unlisted steps use
# orchestrator_tier. Subagents a stage spawns follow [hermes.roles].
[hermes.stages]
"10@light" = "synthesis"   # light: the stage writes the report itself
# full: steps 10/11/12 coordinate Opus subagents (draft-orchestrators,
# synthesizer, critics) per [hermes.roles]; the stage sessions stay on analysis.

# Intake (`hpr hermes intake`, or `hpr hermes icm --intake`): before stage 1,
# one small session turns a rough question into a full research prompt using
# your prompt-structure file, or asks up to 3 clarifying questions first.
[hermes.intake]
structure_file = ""               # path (absolute or vault-relative); "" = built-in
tier = "analysis"

# Publishing a finished run into an Obsidian vault (`hpr hermes publish`,
# or `hpr hermes icm --publish`). Off until `vault` is set. Keep real paths in
# your local copy of this file, not in anything you commit.
[hermes.publish]
vault = ""                        # absolute path to the Obsidian vault
library = "research/sources"      # shared source library (one note per source)
reports = "research/reports"      # reports for runs without --project
runs = "research/runs"            # one sealed folder per run
reports_subdir = "Research"       # with --project P: reports go to P/Research/
write_prefix = []                 # e.g. ["runuser", "-u", "vault", "--"]
# If the Obsidian app mounts the vault over NFS on another machine, write
# there so the app sees new files immediately (NFS has no cross-client
# change notifications). write_prefix then runs on that host.
write_ssh = ""                    # ssh destination, e.g. "root@obsidian-host"
write_root = ""                   # vault path on that host (default: vault)

# Optional list prices, $/Mtok: [input, output, cache_read, cache_write].
# Used only for the estimated cost in published run records.
[hermes.prices]

# Role -> tier. Roles match upstream's ModelMap keys.
[hermes.roles]
fetcher = "bulk"
source_analyst = "analysis"
loci_analyst = "analysis"
depth_investigator = "analysis"
corpus_critic = "analysis"
cite_checker = "analysis"
browser_fetcher = "bulk"
draft_orchestrator = "synthesis"
synthesizer = "synthesis"
critics = "synthesis"
patcher = "analysis"
polish_auditor = "analysis"
readability_recommender = "analysis"
"""

TIERS_REQUIRED = ("bulk", "analysis", "synthesis")


class HermesError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tier:
    provider: str
    model: str


@dataclass
class HermesConfig:
    default_tier: str = "light"
    orchestrator_tier: str = "analysis"
    max_parallel: int = 6
    min_parallel: int = 1
    per_agent_mb: int = 260
    reserve_mb: int = 700
    max_cost_usd: float = 0.0
    spawn_timeout_s: int = 1800
    max_turns: int = 120
    extra_args: list[str] = field(default_factory=list)
    tiers: dict[str, Tier] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    stages: dict[str, str] = field(default_factory=dict)
    intake_structure_file: str = ""
    intake_tier: str = "analysis"
    cite_batch: int = 30

    def tier_for_role(self, role: str | None) -> Tier:
        name = self.roles.get(role or "", "analysis")
        return self.tier(name)

    def tier(self, name: str) -> Tier:
        if name not in self.tiers:
            raise HermesError(f"unknown tier '{name}' (defined: {', '.join(self.tiers)})")
        return self.tiers[name]


def load_config(vault_root: Path) -> HermesConfig:
    """Load `.hyperresearch/hermes.toml`, falling back to the shipped defaults."""
    path = vault_root / CONFIG_FILE
    text = path.read_text(encoding="utf-8") if path.exists() else DEFAULT_CONFIG_TOML
    data = tomllib.loads(text).get("hermes", {})
    defaults = tomllib.loads(DEFAULT_CONFIG_TOML)["hermes"]
    tiers_raw = {**defaults["tiers"], **data.get("tiers", {})}
    tiers = {}
    for name, t in tiers_raw.items():
        if not t.get("model"):
            raise HermesError(f"tier '{name}' has no model")
        tiers[name] = Tier(provider=t.get("provider", ""), model=t["model"])
    cfg = HermesConfig(
        default_tier=data.get("default_tier", defaults["default_tier"]),
        orchestrator_tier=data.get("orchestrator_tier", defaults["orchestrator_tier"]),
        max_parallel=int(data.get("max_parallel", defaults["max_parallel"])),
        min_parallel=int(data.get("min_parallel", defaults["min_parallel"])),
        per_agent_mb=int(data.get("per_agent_mb", defaults["per_agent_mb"])),
        reserve_mb=int(data.get("reserve_mb", defaults["reserve_mb"])),
        spawn_timeout_s=int(data.get("spawn_timeout_s", defaults["spawn_timeout_s"])),
        max_cost_usd=float(data.get("max_cost_usd", defaults["max_cost_usd"])),
        max_turns=int(data.get("max_turns", defaults["max_turns"])),
        extra_args=list(data.get("extra_args", defaults["extra_args"])),
        tiers=tiers,
        roles={**defaults["roles"], **data.get("roles", {})},
        stages={**defaults["stages"], **data.get("stages", {})},
        intake_structure_file=str((data.get("intake") or {}).get("structure_file", "")),
        intake_tier=str((data.get("intake") or {}).get("tier", "analysis")),
        cite_batch=int(data.get("cite_batch", defaults.get("cite_batch", 30))),
    )
    if cfg.default_tier not in ("light", "full", "auto"):
        raise HermesError("default_tier must be light, full, or auto")
    for tier in [*cfg.roles.values(), *cfg.stages.values()]:
        cfg.tier(tier)  # raises on a dangling tier name
    cfg.tier(cfg.orchestrator_tier)
    cfg.tier(cfg.intake_tier)
    if cfg.max_parallel < 1:
        raise HermesError("max_parallel must be >= 1")
    return cfg


def ensure_config(vault_root: Path) -> bool:
    """Write the default hermes.toml if absent. Returns True if written."""
    path = vault_root / CONFIG_FILE
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Codex -> Hermes vocabulary translation
# ---------------------------------------------------------------------------

# Ordered: specific phrases first, the bare word "Codex" last.
_TRANSLATIONS: list[tuple[re.Pattern, Any]] = [
    (re.compile(r"\.hyperresearch/codex/steps/"), ".hyperresearch/hermes/steps/"),
    (re.compile(r"\.agents/skills/hyperresearch/SKILL\.md"), ".hyperresearch/hermes/SKILL.md"),
    (re.compile(r"\.codex/agents/(hyperresearch-[\w<>-]+)\.toml"), r".hyperresearch/hermes/agents/\1.md"),
    (re.compile(r"\.codex/agents/"), ".hyperresearch/hermes/agents/"),
    (re.compile(r"\.codex/hooks\.json"), "the `hpr hermes run` supervisor"),
    (re.compile(r"# spawn the custom agent defined in"), "# spawn with `hpr hermes spawn`; prompt in"),
    (re.compile(r"custom_agent:"), "agent:"),
    (re.compile(r"custom-agent subagents"), "subagents"),
    (re.compile(r"Codex custom agents?"), "Hermes subagent"),
    (re.compile(r"[Cc]ustom agents?"), lambda m: "Subagent" if m.group(0)[0] == "C" else "subagent"),
    (re.compile(r"`apply_patch`"), "`patch`"),
    (re.compile(r"apply_patch"), "patch"),
    (re.compile(r"`update_plan`"), "`todo`"),
    (re.compile(r"update_plan"), "todo"),
    (re.compile(r"`codex exec`"), "a non-interactive Hermes session"),
    (re.compile(r"codex exec"), "a non-interactive Hermes session"),
    (re.compile(r"\$hyperresearch"), "`hpr hermes run`"),
    (re.compile(r"hyperresearch install --steps-only \. --target codex --json"), "hyperresearch hermes install . --json"),
    (re.compile(r"--target codex"), "--target hermes"),
    (re.compile(r"OpenAI Codex"), "Hermes Agent"),
    (re.compile(r"\bCodex\b"), "Hermes"),
    (re.compile(r"\bcodex_step_file\b"), "hermes_step_file"),
]

# Anything matching this after translation is a Codex-ism the table missed.
LEFTOVER_RE = re.compile(
    r"\.codex/|apply_patch|update_plan|custom[_ -]agent|\bcodex\b|--steps-only|\.agents/skills",
    re.IGNORECASE,
)


def translate(text: str) -> str:
    for pattern, repl in _TRANSLATIONS:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# Runtime notes injected into prompts
# ---------------------------------------------------------------------------

def orchestrator_notes(hpr: str) -> str:
    return f"""\
## Hermes runtime notes (read first; these override conflicting tool wording below)

You are the hyperresearch orchestrator running inside a Hermes Agent session.
The procedure below was written for another runtime. Map it as follows:

- **Read a file / step file** -> `read_file` (page through long files; read step files IN FULL).
- **Write / patch a file** -> `write_file` for new files, `patch` for surgical edits.
- **Shell** -> `terminal`.
- **Plan** -> the `todo` tool.
- **Web search** -> `web_search`, for planning only.

### You do not fetch or read sources (enforced)

`{hpr} fetch` and `{hpr} fetch-batch` refuse to run in your session (error code
`DELEGATE_FETCH`). Every source page goes through `hyperresearch-fetcher`
subagents, which fetch, read, and extract claims on a cheaper model in their own
short sessions. You work from their outputs: `{hpr} note list`, claims files, and
the notes they flag. Don't `read_file` full source notes to "check" a fetcher's
work; that re-imports the very context delegation exists to keep out. On light
tier, still spawn 2-3 fetchers with non-overlapping batches.

### Spawning subagents

Subagents are separate processes, each on the model tier configured for its role.
To spawn, write each subagent's full message (research_query block, pipeline position,
inputs, shim) to its own file under `research/runs/<vault_tag>/temp/spawn/`, then launch
ALL the subagents a step calls for in ONE command:

```bash
{hpr} hermes spawn --tag <vault_tag> \\
  --job hyperresearch-fetcher=research/runs/<vault_tag>/temp/spawn/fetcher-1.md \\
  --job hyperresearch-fetcher=research/runs/<vault_tag>/temp/spawn/fetcher-2.md -j
```

The command runs as many jobs in parallel as memory allows and returns within
~150 seconds. If it returns `"status": "running"`, call `{hpr} hermes wait <batch_id> -j`
until `"status": "done"`. The batch keeps running between calls. Each job's result carries its exit code, the
tail of its final message, and a log path. A failed job is re-spawned once; after that,
note the gap and continue.

**Never do a subagent's work inline.** If the step says spawn, spawn.

### Finishing

Your session is supervised: if you stop before `{hpr} run finish <vault_tag> --json`
reports `"passed": true`, you will be resumed and told to continue. End only after the
gate passes, or when the run is honestly blocked (`{hpr} run block`).

---

"""


def _agent_notes(tools: list[str]) -> str:
    have = set(tools)
    lines = [
        "## Hermes runtime notes (read first; these override conflicting tool wording below)",
        "",
        "You are a hyperresearch subagent running as a Hermes Agent process. The",
        "instructions below were written for another runtime. Map them as follows:",
        "",
        "- **Read** -> `read_file`. Read long files in pages; never skip parts you were told to read in full.",
        "- **Write** -> `write_file`. **Edit** -> `patch` (surgical hunks only).",
    ]
    if "Bash" in have:
        lines.append("- **Bash** -> `terminal`.")
    else:
        lines.append("- You have no shell in this role, by design. Do not attempt one.")
    if "WebSearch" in have:
        lines.append("- **WebSearch** -> `web_search`. Fetch pages ONLY with the hyperresearch CLI `fetch`.")
    lines.append("- **Task** / **Skill** tools do not exist. You cannot spawn subagents.")
    lines.append(
        "- Your working directory is already the research vault. Don't `cd` elsewhere and "
        "never run `init`; the hyperresearch CLI is pinned to this vault."
    )
    if "Task" in have:
        lines.append(
            "- Wherever the instructions tell you to delegate fetching to fetcher "
            "subagents, fetch yourself with the hyperresearch CLI (`fetch` or "
            "`fetch-batch`) using the same tags, then do the fetcher's job on each note."
        )
    if "Edit" in have and "Write" not in have:
        lines.append(
            "- You may ONLY make surgical `patch` edits to the files your task names. "
            "Never create, delete, or wholesale rewrite a file."
        )
    if "Write" in have and "Edit" not in have:
        lines.append(
            "- Create or overwrite only the output files your task names. Never edit a "
            "file another stage owns."
        )
    lines.append(
        "- When done, reply with a short summary of what you wrote and where. "
        "Your final message is returned to the orchestrator."
    )
    return "\n".join(lines) + "\n\n---\n\n"


def toolsets_for(tools: list[str]) -> list[str]:
    """Upstream `tools:` allowlist -> Hermes toolsets (a real lock, not a request)."""
    have = set(tools)
    sets = ["file"]
    if "Bash" in have:
        sets.append("terminal")
    if "WebSearch" in have:
        sets.append("web")
    return sets


ORCHESTRATOR_TOOLSETS = ["file", "terminal", "web", "todo"]


# ---------------------------------------------------------------------------
# Install (called from core/hooks.py under a hermes render target)
# ---------------------------------------------------------------------------

_MODEL_ROLE_RE = re.compile(r"^model:\s*<<\s*p\.models\.(\w+)\s*>>", re.MULTILINE)


def _write_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return True


def write_agent(vault_root: Path, filename: str, template: str, rendered: str, label: str) -> str | None:
    """Write one agent prompt to .hyperresearch/hermes/agents/<name>.md.

    Frontmatter keeps `name`, `description`, `tools`, `role` (the ModelMap key
    that picks the tier) and `toolsets`; the body gets the Hermes preamble.
    """
    import yaml

    from hyperresearch.core.codex import parse_tools, split_frontmatter

    meta, body = split_frontmatter(rendered)
    name = str(meta.get("name", "")).strip() or Path(filename).stem
    tools = parse_tools(meta.get("tools"))
    role_match = _MODEL_ROLE_RE.search(template)
    front = {
        "name": name,
        "description": " ".join(str(meta.get("description", "")).split()),
        "role": role_match.group(1) if role_match else None,
        "tools": tools,
        "toolsets": toolsets_for(tools),
    }
    text = (
        "---\n"
        + yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=10_000)
        + "---\n"
        + translate(_agent_notes(tools) + body.lstrip("\n"))
    )
    out = vault_root / AGENTS_DIR / f"{name}.md"
    if not _write_if_changed(out, text):
        return None
    return f"Hermes: {AGENTS_DIR}/{name}.md ({label})"


def write_entry_skill(vault_root: Path, rendered: str, hpr: str) -> str | None:
    from hyperresearch.core.codex import split_frontmatter

    _, body = split_frontmatter(rendered)
    text = orchestrator_notes(hpr) + translate(body.lstrip("\n"))
    if not _write_if_changed(vault_root / ENTRY_SKILL, text):
        return None
    return f"Hermes: {ENTRY_SKILL} (orchestrator entry)"


def write_step_files(vault_root: Path, rendered_steps: dict[str, str]) -> str | None:
    changed = []
    steps_dir = vault_root / STEPS_DIR
    for name, rendered in rendered_steps.items():
        if _write_if_changed(steps_dir / f"{name}.md", translate(rendered)):
            changed.append(name)
    expected = {f"{n}.md" for n in rendered_steps}
    for child in steps_dir.glob("hyperresearch-*.md"):
        if child.name not in expected:
            child.unlink()
    if not changed:
        return None
    return f"Hermes: {STEPS_DIR}/ ({len(changed)} step files)"


def is_installed(vault_root: Path) -> bool:
    return (vault_root / ENTRY_SKILL).is_file()


# ---------------------------------------------------------------------------
# Spawning subagents
# ---------------------------------------------------------------------------

def hermes_bin() -> str:
    exe = os.environ.get("HPR_HERMES_BIN") or shutil.which("hermes")
    if not exe:
        raise HermesError("`hermes` not found on PATH")
    return exe


def build_chat_cmd(
    tier: Tier,
    toolsets: list[str],
    query_file: Path,
    cfg: HermesConfig,
    max_turns: int | None = None,
    workdir: Path | None = None,
) -> list[str]:
    cmd = [
        hermes_bin(), "chat",
        "--query-file", str(Path(query_file).resolve()),
        "--format", "stream-json",
        "-m", tier.model,
        "-t", ",".join(toolsets),
        "--max-turns", str(max_turns or cfg.max_turns),
        "--ignore-rules",       # no SOUL/AGENTS/memory injection: isolation + cost
        "--source", "tool",     # keep these out of the user's session pickers
    ]
    if tier.provider:
        cmd += ["--provider", tier.provider]
    if workdir is not None:
        # Hermes file tools resolve relative paths against the session's
        # workspace, not the process cwd; pin it to the vault.
        cmd += ["--in", str(Path(workdir).resolve())]
    return cmd + list(cfg.extra_args)


ROLE_ENV = "HPR_HERMES_ROLE"
ORCHESTRATOR_ROLE = "orchestrator"


def chat_env(workdir: Path, role: str) -> dict[str, str]:
    """Environment for a `hermes chat` child.

    TERMINAL_CWD anchors Hermes' file + terminal tools to the vault.
    HPR_HERMES_ROLE tells the hpr CLI who is calling, so the orchestrator
    can be refused the work it must delegate (see guard_orchestrator_fetch).
    """
    env = dict(os.environ)
    env["TERMINAL_CWD"] = str(Path(workdir).resolve())
    env["HPR_VAULT_ROOT"] = str(Path(workdir).resolve())
    env[ROLE_ENV] = role
    _apply_pass_file(env)
    return env


PASS_FILE_ENV = "HPR_CLAUDE_PASS_FILE"


def _apply_pass_file(env: dict[str, str]) -> None:
    """Worker hosts: read the current short-lived Claude pass at every spawn.

    A run outlives one pass, so the dispatching host rewrites the file with a
    fresh pass while the run is going; each new `hermes chat` picks up
    whatever is current. The pass never goes on a command line or into a log.
    """
    path = env.get(PASS_FILE_ENV)
    if not path:
        return
    try:
        tok = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return
    if tok:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = tok


def guard_orchestrator_fetch(json_output: bool = False) -> None:
    """Refuse `hpr fetch` / `fetch-batch` when called by the orchestrator.

    Fetching in the orchestrator's own session runs it on the orchestrator's
    model and grows its context with every page, which every later turn then
    re-reads. Measured on the first light run: ~16M tokens, nearly all of it
    this. Fetchers run on the bulk tier in their own short sessions.
    """
    if os.environ.get(ROLE_ENV) != ORCHESTRATOR_ROLE:
        return
    import json as _json

    import typer

    msg = (
        "The orchestrator does not fetch. Write the URL batch to a message file and "
        "spawn fetchers: `hpr hermes spawn --tag <vault_tag> "
        "--job hyperresearch-fetcher=<msg-file> ...`"
    )
    if json_output:
        print(_json.dumps({"ok": False, "error": {"code": "DELEGATE_FETCH", "message": msg}}))
    else:
        print(f"Error: {msg}", file=sys.stderr)
    raise typer.Exit(2)


def load_agent(vault_root: Path, agent: str) -> tuple[dict, str]:
    import yaml

    path = vault_root / AGENTS_DIR / f"{agent}.md"
    if not path.is_file():
        raise HermesError(f"unknown agent '{agent}' (no {path.relative_to(vault_root)})")
    text = path.read_text(encoding="utf-8")
    end = text.find("\n---", 3)
    meta = yaml.safe_load(text[3:end]) or {}
    body = text[text.find("\n", end + 1) + 1 :]
    return meta, body


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def missing_outputs(vault_root: Path, expects: list[str]) -> list[str]:
    """Expected output files that don't exist or are empty."""
    out = []
    for rel in expects:
        f = vault_root / rel
        try:
            if f.stat().st_size == 0:
                out.append(rel)
        except OSError:
            out.append(rel)
    return out


def write_blocked_reason(vault_root: Path, env: dict[str, str] | None = None) -> str | None:
    """Why Hermes' file tools can't write the vault here, or None if they can.

    Hermes refuses every write outside HERMES_WRITE_SAFE_ROOT. Container
    images set it (the official one: /opt/data), so a vault elsewhere gets
    silent per-call denials: agents improvise around them or skip their
    output. Checked before a run starts, not discovered mid-run.
    """
    env = os.environ if env is None else env
    roots = [r for r in (env.get("HERMES_WRITE_SAFE_ROOT") or "").split(os.pathsep) if r]
    if not roots:
        return None
    v = Path(vault_root).resolve()
    for r in roots:
        try:
            if v.is_relative_to(Path(r).resolve()):
                return None
        except OSError:
            continue
    return (f"HERMES_WRITE_SAFE_ROOT={env.get('HERMES_WRITE_SAFE_ROOT')} excludes the vault {v}; "
            "every agent write would be denied. Add the vault to it (or unset it) before running.")


def _batch_dir(vault_root: Path, batch_id: str) -> Path:
    return vault_root / BATCHES_DIR / batch_id


def create_batch(vault_root: Path, jobs: list[tuple], tag: str | None) -> str:
    """Validate jobs, write prompts + batch.json, return the batch id.

    A job is ``(agent, message_path)`` or ``(agent, message_path, expects)``
    where ``expects`` lists vault-relative files the agent promises to write.
    A job that exits 0 without them is marked ``failed`` (missing_outputs),
    never ``done``: an agent's word is not a handoff; the file is.
    """
    cfg = load_config(vault_root)
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    bdir = _batch_dir(vault_root, batch_id)
    bdir.mkdir(parents=True)
    records = []
    for i, job_spec in enumerate(jobs):
        agent, msg_path = job_spec[0], job_spec[1]
        expects = [str(e) for e in (job_spec[2] if len(job_spec) > 2 else [])]
        meta, body = load_agent(vault_root, agent)
        msg_path = (vault_root / msg_path) if not msg_path.is_absolute() else msg_path
        if not msg_path.is_file():
            raise HermesError(f"message file not found: {msg_path}")
        tier = cfg.tier_for_role(meta.get("role"))
        prompt = bdir / f"{i:02d}-{agent}.prompt.md"
        prompt.write_text(
            body.rstrip() + "\n\n---\n\n# Your assignment\n\n" + msg_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        records.append({
            "index": i,
            "agent": agent,
            "role": meta.get("role"),
            "tier": cfg.roles.get(meta.get("role") or "", "analysis"),
            "model": tier.model,
            "provider": tier.provider,
            "toolsets": meta.get("toolsets") or ["file"],
            "message_file": str(msg_path.relative_to(vault_root)) if msg_path.is_relative_to(vault_root) else str(msg_path),
            "prompt_file": str(prompt.relative_to(vault_root)),
            "log_file": str((bdir / f"{i:02d}-{agent}.log.jsonl").relative_to(vault_root)),
            "expects": expects,
            "status": "queued",
        })
    batch = {"batch_id": batch_id, "tag": tag, "created": _now(), "status": "queued", "jobs": records}
    (bdir / "batch.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
    return batch_id


def _load_batch(vault_root: Path, batch_id: str) -> dict:
    path = _batch_dir(vault_root, batch_id) / "batch.json"
    if not path.is_file():
        raise HermesError(f"unknown batch '{batch_id}'")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_batch(vault_root: Path, batch: dict) -> None:
    path = _batch_dir(vault_root, batch["batch_id"]) / "batch.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    tmp.replace(path)


def _parse_result(log_path: Path) -> dict:
    """Pull the final `result` event (text + tokens) out of a stream-json log."""
    result: dict = {}
    if not log_path.exists():
        return result
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            result = ev
    return result


def mem_available_mb() -> int | None:
    """MemAvailable from /proc/meminfo, in MB (None where unavailable)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def can_start(running: int, cfg: HermesConfig, avail_mb: int | None) -> bool:
    """Start another subagent now? Pure; unit-tested.

    Always allows min_parallel; never exceeds max_parallel; in between,
    requires room for one more agent on top of the reserve. Memory of agents
    that just started isn't in MemAvailable yet, so the caller waits a few
    seconds between starts.
    """
    if running < cfg.min_parallel:
        return True
    if running >= cfg.max_parallel:
        return False
    if avail_mb is None:
        return running < 3
    return avail_mb >= cfg.reserve_mb + cfg.per_agent_mb


def run_batch(vault_root: Path, batch_id: str) -> None:
    """Supervisor loop: run a batch's jobs with the parallel cap. Runs detached."""
    cfg = load_config(vault_root)
    batch = _load_batch(vault_root, batch_id)
    batch["status"] = "running"
    batch["supervisor_pid"] = os.getpid()
    _save_batch(vault_root, batch)

    running: dict[int, tuple[subprocess.Popen, float, IO[str]]] = {}
    queue = [j["index"] for j in batch["jobs"] if j["status"] == "queued"]
    last_start = 0.0
    while queue or running:
        while (
            queue
            and can_start(len(running), cfg, mem_available_mb())
            and (len(running) < cfg.min_parallel or time.monotonic() - last_start > 8)
        ):
            last_start = time.monotonic()
            idx = queue.pop(0)
            job = batch["jobs"][idx]
            tier = Tier(provider=job["provider"], model=job["model"])
            cmd = build_chat_cmd(
                tier, job["toolsets"], vault_root / job["prompt_file"], cfg, workdir=vault_root
            )
            log = open(vault_root / job["log_file"], "w", encoding="utf-8")  # noqa: SIM115
            proc = subprocess.Popen(
                cmd, cwd=vault_root, env=chat_env(vault_root, job["agent"]), stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            job.update(status="running", pid=proc.pid, started=_now())
            running[idx] = (proc, time.monotonic(), log)
            batch["peak_parallel"] = max(batch.get("peak_parallel", 0), len(running))
            _save_batch(vault_root, batch)
        time.sleep(2)
        for idx, (proc, started, log) in list(running.items()):
            job = batch["jobs"][idx]
            timed_out = time.monotonic() - started > cfg.spawn_timeout_s
            if proc.poll() is None and not timed_out:
                continue
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=15)
                except Exception:
                    os.killpg(proc.pid, signal.SIGKILL)
            log.close()
            res = _parse_result(vault_root / job["log_file"])
            code = proc.returncode
            ok = code == 0 and not timed_out and bool(res)
            missing = missing_outputs(vault_root, job.get("expects") or [])
            if missing:
                job["missing_outputs"] = missing
                ok = False
            job.update(
                status="done" if ok else ("timed_out" if timed_out else "failed"),
                exit_code=code,
                finished=_now(),
                tokens=res.get("tokens"),
                duration_ms=res.get("duration_ms"),
                session_id=res.get("session_id"),
                final_text=(res.get("text") or "")[-1500:],
            )
            del running[idx]
            _save_batch(vault_root, batch)
            _ledger(vault_root, batch, job)
    batch["status"] = "done"
    batch["finished"] = _now()
    _save_batch(vault_root, batch)


def _ledger(vault_root: Path, batch: dict, job: dict) -> None:
    """Append one line per finished spawn to the run's token ledger."""
    tag = batch.get("tag")
    if not tag:
        return
    path = vault_root / "research" / "runs" / tag / "temp" / "hermes-spawns.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: job.get(k) for k in ("agent", "role", "tier", "model", "status", "exit_code", "tokens", "duration_ms")}
    row["batch_id"] = batch["batch_id"]
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def start_batch_detached(vault_root: Path, batch_id: str) -> None:
    """Launch the supervisor fully detached so it outlives this CLI call.

    Double fork: the supervisor is re-parented to init at once, so it is not a
    descendant of the caller. Agent terminal tools kill a timed-out command's
    whole process tree; a merely setsid'd child is still in that tree and died
    with it (observed: a 180 s tool timeout killed a running fetcher batch).
    """
    log_path = _batch_dir(vault_root, batch_id) / "supervisor.log"
    argv = [sys.executable, "-m", "hyperresearch.core.hermes", "supervise", str(vault_root), batch_id]
    pid = os.fork()
    if pid == 0:  # intermediate child
        try:
            os.setsid()
            if os.fork() > 0:
                os._exit(0)
            os.chdir(vault_root)
            fd_in = os.open(os.devnull, os.O_RDONLY)
            fd_out = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.dup2(fd_in, 0)
            os.dup2(fd_out, 1)
            os.dup2(fd_out, 2)
            os.execv(sys.executable, argv)
        finally:
            os._exit(1)
    os.waitpid(pid, 0)


def wait_batch(vault_root: Path, batch_id: str, timeout_s: int) -> dict:
    """Block until the batch finishes or `timeout_s` elapses; return a summary."""
    deadline = time.monotonic() + timeout_s
    while True:
        batch = _load_batch(vault_root, batch_id)
        if batch["status"] == "done" or time.monotonic() >= deadline:
            return summarize(batch)
        pid = batch.get("supervisor_pid")
        if pid and batch["status"] == "running" and not _alive(pid):
            batch["status"] = "done"
            for j in batch["jobs"]:
                if j["status"] in ("queued", "running"):
                    j["status"] = "failed"
                    j["final_text"] = "supervisor died"
            _save_batch(vault_root, batch)
            return summarize(batch)
        time.sleep(3)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def summarize(batch: dict) -> dict:
    jobs = [
        {k: j.get(k) for k in ("index", "agent", "tier", "model", "status", "exit_code", "final_text", "log_file")}
        for j in batch["jobs"]
    ]
    counts: dict[str, int] = {}
    for j in batch["jobs"]:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    return {
        "batch_id": batch["batch_id"],
        "status": batch["status"],
        "counts": counts,
        "peak_parallel": batch.get("peak_parallel"),
        "jobs": jobs,
    }


# ---------------------------------------------------------------------------
# Orchestrator runs
# ---------------------------------------------------------------------------

def orchestrator_prompt(query: str, tier_cap: str, budget: float | None, hpr: str) -> str:
    tier_rule = {
        "light": (
            "**Tier cap: light.** In step 1, set `pipeline_tier` to `\"light\"` regardless of "
            "your classification, and note in the scaffold that the cap was applied. Run "
            "steps 1 -> 2 -> 10 -> 15 -> 16 only."
        ),
        "full": "**Tier: full.** In step 1, set `pipeline_tier` to `\"full\"` and run every full-tier step.",
        "auto": "**Tier: auto.** Classify `pipeline_tier` in step 1 as the procedure describes.",
    }[tier_cap]
    budget_rule = f" Pass `--budget {budget}` to `run init`." if budget else ""
    return (
        "Run the hyperresearch pipeline for the research query below.\n\n"
        f"1. Read `{ENTRY_SKILL}` IN FULL with `read_file` (page through it) and follow it exactly.\n"
        f"2. The CLI is `{hpr}` (use this exact path).\n"
        f"3. {tier_rule}{budget_rule}\n"
        "4. The canonical research query is the text between the markers, verbatim.\n\n"
        "<<<RESEARCH_QUERY\n" + query.strip() + "\nRESEARCH_QUERY>>>\n"
    )


def continue_prompt(hpr: str) -> str:
    return (
        "The research run is not finished: the ship gate has not passed. Continue the "
        f"pipeline from where you stopped. Check `{hpr} run resume -j` for the next step, "
        f"re-read `{ENTRY_SKILL}` if you have lost track, and keep going until "
        f"`{hpr} run finish <vault_tag> --json` reports passed, or block the run honestly."
    )


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "supervise":
        run_batch(Path(argv[1]), argv[2])
        return 0
    print("usage: python -m hyperresearch.core.hermes supervise <vault_root> <batch_id>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

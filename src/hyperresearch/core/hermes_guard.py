"""Guards for Hermes research runs: locked files and stray-file cleanup.

Two failure modes from real runs, both handled in code rather than by
asking agents nicely:

* **Locked files.** Agents must never edit the pipeline's own instructions
  or config (the best-documented long-running ICM failure was an agent
  slowly rewriting its own templates). `hpr hermes install` records a
  SHA-256 manifest of every protected file and a pristine copy. A run
  refuses to start if protected files differ from the manifest, and after
  every stage code restores anything an agent changed and logs it.
  Sessions run as the same OS user as this code, so file modes alone can't
  stop them; the check-and-restore is the guarantee, chmod 0444 a speed bump.

* **Stray files.** Helpers left unrequested reports (FETCHER_REPORT*.md,
  RESEARCH_FINDINGS.md, ...) in the run folder and the vault root. After
  each stage, top-level files in the run folder that no step or agent file
  names, and new files at the vault root, are moved to
  `stages/_strays/<stage>/` and logged.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
from pathlib import Path

from hyperresearch.core import hermes

LOCK_FILE = hermes.HERMES_DIR / "lock.json"
PRISTINE_DIR = hermes.HERMES_DIR / ".pristine"


def protected_files(vault_root: Path) -> list[Path]:
    """Relative paths of every file agents must never change."""
    rels: list[Path] = [hermes.ENTRY_SKILL, hermes.CONFIG_FILE, Path(".hyperresearch") / "config.toml"]
    for d in (hermes.STEPS_DIR, hermes.AGENTS_DIR):
        full = vault_root / d
        if full.is_dir():
            rels += sorted(p.relative_to(vault_root) for p in full.glob("*.md"))
    return [r for r in rels if (vault_root / r).is_file()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_lock(vault_root: Path) -> int:
    """Record hashes + a pristine copy of the protected files. Called by install."""
    manifest = {}
    pristine = vault_root / PRISTINE_DIR
    if pristine.exists():
        shutil.rmtree(pristine)
    for rel in protected_files(vault_root):
        src = vault_root / rel
        manifest[str(rel)] = _sha(src)
        dst = pristine / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        src.chmod(src.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    (vault_root / LOCK_FILE).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return len(manifest)


def lock_digest(vault_root: Path) -> str:
    p = vault_root / LOCK_FILE
    return _sha(p) if p.exists() else ""


def unlock_for_install(vault_root: Path) -> None:
    """Make protected files writable again so install can re-render them."""
    for rel in protected_files(vault_root):
        p = vault_root / rel
        p.chmod(p.stat().st_mode | stat.S_IWUSR)


def changed_files(vault_root: Path) -> list[str]:
    """Protected files that differ from (or are missing against) the lock."""
    lock = vault_root / LOCK_FILE
    if not lock.exists():
        return []
    manifest = json.loads(lock.read_text(encoding="utf-8"))
    out = []
    for rel, digest in manifest.items():
        p = vault_root / rel
        if not p.is_file() or _sha(p) != digest:
            out.append(rel)
    return out


def restore(vault_root: Path, rels: list[str]) -> list[str]:
    """Put protected files back from the pristine copy. Returns what was restored."""
    done = []
    for rel in rels:
        src = vault_root / PRISTINE_DIR / rel
        dst = vault_root / rel
        if not src.is_file():
            continue
        if dst.exists():
            dst.chmod(dst.stat().st_mode | stat.S_IWUSR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        dst.chmod(dst.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        done.append(rel)
    return done


# ---------------------------------------------------------------------------
# Stray files
# ---------------------------------------------------------------------------

# Always legitimate at the top of a run folder, whatever the step files say.
_RUN_BASE = {"run.json", "events.jsonl", "query.md", "scaffold.md", "stages", "temp", "shims", "interim"}
_NAME_RE = re.compile(r"research/runs/<vault_tag>/([A-Za-z0-9_.<>*{},\-]+)")


def _pattern_to_regex(name: str) -> str:
    out = ""
    i = 0
    while i < len(name):
        c = name[i]
        if c == "<":
            j = name.find(">", i)
            out += r"[^/]+"
            i = j + 1 if j > 0 else i + 1
            continue
        if c == "{":
            j = name.find("}", i)
            if j > 0:
                out += "(" + "|".join(re.escape(x) for x in name[i + 1 : j].split(",")) + ")"
                i = j + 1
                continue
        out += r"[^/]*" if c == "*" else re.escape(c)
        i += 1
    return out


def allowed_run_names(vault_root: Path) -> re.Pattern:
    """Top-level names the installed step/agent files say a run folder may hold."""
    pats = {re.escape(n) for n in _RUN_BASE}
    for d in (hermes.STEPS_DIR, hermes.AGENTS_DIR):
        for f in (vault_root / d).glob("*.md"):
            for m in _NAME_RE.finditer(f.read_text(encoding="utf-8", errors="replace")):
                first = m.group(1).split("/")[0].rstrip(".,")
                if first:
                    pats.add(_pattern_to_regex(first))
    # Numbered variants of any JSON output (cite-check-findings-1.json etc.)
    pats.add(r"[a-z0-9\-]+-(\d+|[a-z])\.json")
    return re.compile("^(" + "|".join(sorted(pats)) + ")$")


def snapshot_root(vault_root: Path) -> set[str]:
    return {p.name for p in vault_root.iterdir()}


def sweep_strays(vault_root: Path, tag: str, stage_label: str, root_before: set[str],
                 allowed: re.Pattern | None = None) -> list[str]:
    """Move unrequested files out of the run folder and vault root. Returns moved paths."""
    run_dir = vault_root / "research" / "runs" / tag
    dest = run_dir / "stages" / "_strays" / stage_label
    allowed = allowed or allowed_run_names(vault_root)
    moved = []
    candidates = []
    if run_dir.is_dir():
        candidates += [p for p in run_dir.iterdir() if p.is_file() and not allowed.match(p.name)]
    for name in snapshot_root(vault_root) - root_before:
        p = vault_root / name
        if p.is_file() and not name.startswith("."):
            candidates.append(p)
    for p in candidates:
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / p.name
        shutil.move(str(p), target)
        moved.append(str(p.relative_to(vault_root)))
    return moved

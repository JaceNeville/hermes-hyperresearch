"""Publish a finished run into an Obsidian vault.

    hpr hermes publish <vault_tag> [--project "Efforts/Some Project"]

The hyperresearch vault is the working library (SQLite index, claims JSON,
stage logs). An Obsidian vault is where people read. Publishing mirrors the
reader-facing parts of a run into the Obsidian vault, with one home per thing:

* **Sources** -> `<library>/<note-id>.md`, one shared library across runs.
  A source already in the library is left alone; a different source that
  happens to share an id gets a suffixed filename.
* **Report** -> `<project>/<reports_subdir>/<date> <title>.md` when the run
  belongs to a project, else `<reports>/<date> <title>.md`.
* **Run record** -> `<runs>/<vault_tag>/RUN.md` (query, tier, stage table,
  spend, report + source links) plus the scaffold and any interim notes.
  JSON artifacts and session logs stay in the working vault.

Wikilinks in published notes are rewritten to path-qualified links
(`[[<library>/<id>|<id>]]`), so a source called `united-states` can't
resolve to some other note in a large vault.

Writes go through an optional command prefix (e.g. `runuser -u vault --`) for
vaults owned by another user, as a tar stream on stdin. Nothing is
overwritten except files the publisher itself owns (report, run record).
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import tarfile
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from hyperresearch.core import hermes

LINK_RE = re.compile(r"\[\[([^\]|#]+)(#[^\]|]*)?(\|[^\]]*)?\]\]")
_BAD_NAME = re.compile(r'[\\/:*?"<>|#^\[\]]+')


@dataclass
class PublishConfig:
    vault: str = ""
    library: str = "research/sources"
    reports: str = "research/reports"
    runs: str = "research/runs"
    reports_subdir: str = "Research"
    write_prefix: list[str] = field(default_factory=list)
    # Optional: write through another host (ssh destination) and/or at a
    # different path there. Needed when the Obsidian app runs on a machine
    # that mounts the vault over NFS: NFS doesn't send change notifications
    # between clients, so files written from elsewhere stay invisible to the
    # app until it rescans. Writing on the app's own host makes them appear
    # at once. write_prefix then runs on that host (e.g. docker exec ...).
    write_ssh: str = ""
    write_root: str = ""
    prices: dict[str, list[float]] = field(default_factory=dict)  # model -> [in, out, cache_read, cache_write] $/Mtok

    @property
    def enabled(self) -> bool:
        return bool(self.vault)


def load_publish_config(vault_root: Path) -> PublishConfig:
    path = vault_root / hermes.CONFIG_FILE
    data = tomllib.loads(path.read_text(encoding="utf-8")).get("hermes", {}) if path.exists() else {}
    pub = dict(data.get("publish", {}))
    prices = {k: list(v) for k, v in (data.get("prices") or {}).items()}
    cfg = PublishConfig(**{k: v for k, v in pub.items() if k in PublishConfig.__dataclass_fields__}, prices=prices)
    for rel in (cfg.library, cfg.reports, cfg.runs, cfg.reports_subdir):
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise hermes.HermesError(f"publish paths must be relative to the vault: {rel!r}")
    return cfg


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def split_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end > 0:
            try:
                meta = yaml.safe_load(text[4:end]) or {}
            except yaml.YAMLError:
                meta = {}
            body = text[end + 4 :].lstrip("\n")
            return (meta if isinstance(meta, dict) else {}), body
    return {}, text


def join_frontmatter(meta: dict, body: str) -> str:
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=100) + "---\n\n" + body


def rewrite_links(text: str, targets: dict[str, str]) -> str:
    """`[[id]]` -> `[[path/id|id]]` for ids we publish; everything else untouched."""

    def sub(m: re.Match) -> str:
        nid, anchor, alias = m.group(1).strip(), m.group(2) or "", m.group(3)
        if nid not in targets:
            return m.group(0)
        return f"[[{targets[nid]}{anchor}{alias or '|' + nid}]]"

    return LINK_RE.sub(sub, text)


def safe_title(title: str, limit: int = 90) -> str:
    t = _BAD_NAME.sub(" ", title).strip(" .")
    t = re.sub(r"\s+", " ", t)
    return (t[:limit].rstrip() or "Research report")


def report_title(report_text: str, fallback: str) -> str:
    _, body = split_frontmatter(report_text)
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def estimate_cost(tokens: dict, price: list[float] | None) -> float | None:
    if not price:
        return None
    pin, pout, pcr, pcw = [*list(price), 0, 0, 0, 0][:4]
    return (
        int(tokens.get("input") or 0) * pin
        + int(tokens.get("output") or 0) * pout
        + int(tokens.get("cache_read") or 0) * pcr
        + int(tokens.get("cache_write") or 0) * pcw
    ) / 1e6


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


@dataclass
class PublishResult:
    report: str | None
    run_record: str
    sources_new: list[str]
    sources_existing: list[str]
    sources_renamed: dict[str, str]
    interim: list[str]

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _run_json(hpr: str, args: list[str], cwd: Path) -> dict:
    p = subprocess.run([hpr, *args, "--json"], cwd=cwd, capture_output=True, text=True, timeout=300)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {}


def _write_tree(files: dict[str, str], dest_root: Path, prefix: list[str], ssh: str = "") -> None:
    """Write {relpath: text} under dest_root, via `prefix` if set, as one tar stream.

    With `ssh`, the prefix + tar command run on that host (arguments quoted
    for the remote shell) and dest_root is a path on that host.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        dirs = sorted({str(Path(r).parent) for r in files} - {"."})
        for d in dirs:
            ti = tarfile.TarInfo(d)
            ti.type, ti.mode = tarfile.DIRTYPE, 0o755
            tar.addfile(ti)
        for rel, text in files.items():
            data = text.encode("utf-8")
            ti = tarfile.TarInfo(rel)
            ti.size, ti.mode, ti.mtime = len(data), 0o644, int(datetime.now(UTC).timestamp())
            tar.addfile(ti, io.BytesIO(data))
    cmd = [*prefix, "tar", "-C", str(dest_root), "-xf", "-", "--no-same-owner", "--no-same-permissions"]
    if ssh:
        import shlex

        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", ssh, shlex.join(cmd)]
    p = subprocess.run(cmd, input=buf.getvalue(), capture_output=True, cwd="/tmp", timeout=300)
    if p.returncode != 0:
        raise hermes.HermesError(f"publish write failed: {p.stderr.decode(errors='replace')[-500:]}")


def publish_run(vault_root: Path, tag: str, hpr: str, project: str | None = None,
                cfg: PublishConfig | None = None) -> PublishResult:
    cfg = cfg or load_publish_config(vault_root)
    if not cfg.enabled:
        raise hermes.HermesError("publishing is off: set [hermes.publish] vault in .hyperresearch/hermes.toml")
    dest = Path(cfg.vault)
    if not dest.is_dir():
        raise hermes.HermesError(f"Obsidian vault not found: {dest}")
    if project and (project.startswith("/") or ".." in Path(project).parts):
        raise hermes.HermesError("--project must be a path inside the Obsidian vault")
    if project and not (dest / project).is_dir():
        raise hermes.HermesError(f"project folder not found in the vault: {project}")

    run_dir = vault_root / "research" / "runs" / tag
    report_src = vault_root / "research" / "notes" / f"final_report_{tag}.md"
    if not run_dir.is_dir():
        raise hermes.HermesError(f"no run {tag}")
    manifest = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    _, query_body = split_frontmatter((run_dir / "query.md").read_text()) if (run_dir / "query.md").exists() else ({}, "")
    notes = (_run_json(hpr, ["note", "list", "--tag", tag, "--all"], vault_root).get("data")) or []

    files: dict[str, str] = {}
    targets: dict[str, str] = {}
    new, existing, renamed, interim = [], [], {}, []
    # --- sources + interim ----------------------------------------------------
    for n in notes:
        nid = n["id"]
        if nid.startswith("final_report_"):
            continue
        src_path = vault_root / n["path"]
        if not src_path.exists():
            continue
        if n.get("type") == "interim":
            rel = f"{cfg.runs}/{tag}/interim/{nid}.md"
            targets[nid] = rel[:-3]
            interim.append(rel)
            continue
        rel = f"{cfg.library}/{nid}.md"
        there = dest / rel
        if there.exists():
            other, _ = split_frontmatter(there.read_text(encoding="utf-8", errors="replace"))
            if other.get("source") and other.get("source") == split_frontmatter(src_path.read_text())[0].get("source"):
                existing.append(rel)
                targets[nid] = rel[:-3]
                continue
            rel = f"{cfg.library}/{nid}--{tag[-6:]}.md"
            renamed[nid] = rel
        targets[nid] = rel[:-3]
        new.append(rel)

    def body_of(nid_path: Path, extra_meta: dict | None = None) -> str:
        meta, body = split_frontmatter(nid_path.read_text(encoding="utf-8"))
        meta.update(extra_meta or {})
        return join_frontmatter(meta, rewrite_links(body, targets))

    for n in notes:
        rel = targets.get(n["id"])
        if rel and (rel + ".md") in (*new, *interim):
            files[rel + ".md"] = body_of(vault_root / n["path"], {"published_from_run": tag})

    # --- report ---------------------------------------------------------------
    date = (manifest.get("started_at") or datetime.now(UTC).isoformat())[:10]
    report_rel = None
    if report_src.exists():
        text = report_src.read_text(encoding="utf-8")
        title = report_title(text, tag)
        folder = f"{project}/{cfg.reports_subdir}" if project else cfg.reports
        report_rel = f"{folder}/{date} {safe_title(title)}.md"
        # One report per run: never overwrite another run's report.
        existing_report = dest / report_rel
        if existing_report.exists():
            other_meta, _ = split_frontmatter(existing_report.read_text(encoding="utf-8", errors="replace"))
            if tag not in str(other_meta.get("run", "")):
                report_rel = f"{folder}/{date} {safe_title(title)} ({tag[-6:]}).md"
        meta, body = split_frontmatter(text)
        meta = {
            "title": title,
            "type": "research-report",
            "date": date,
            "tier": (manifest.get("profile") or ""),
            "query": (query_body.strip().splitlines() or [""])[0][:300],
            "run": f"[[{cfg.runs}/{tag}/RUN|{tag}]]",
            "sources": len(new) + len(existing),
            "tags": ["research"],
            **{k: v for k, v in meta.items() if k not in ("tags",)},
        }
        files[report_rel] = join_frontmatter(meta, rewrite_links(body, targets))

    # --- run record -----------------------------------------------------------
    for name in ("scaffold.md",):
        if (run_dir / name).exists():
            files[f"{cfg.runs}/{tag}/{name}"] = rewrite_links((run_dir / name).read_text(encoding="utf-8"), targets)
    files[f"{cfg.runs}/{tag}/RUN.md"] = _run_record(
        tag, manifest, query_body, run_dir, cfg, report_rel, new, existing, interim, project
    )
    _write_tree(files, Path(cfg.write_root) if cfg.write_root else dest, cfg.write_prefix, cfg.write_ssh)
    return PublishResult(report_rel, f"{cfg.runs}/{tag}/RUN.md", new, existing, renamed, interim)


def _run_record(tag, manifest, query_body, run_dir, cfg, report_rel, new, existing, interim, project) -> str:
    rows, by_model = [], {}
    ledger = run_dir / "temp" / "hermes-spawns.jsonl"
    total_cost, priced = 0.0, True
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r.get("tokens") or {}
            m = by_model.setdefault(r.get("model", "?"), {"sessions": 0, "tokens": 0, "cost": 0.0})
            m["sessions"] += 1
            m["tokens"] += int(t.get("total") or 0)
            c = estimate_cost(t, cfg.prices.get(r.get("model", "")))
            if c is None:
                priced = False
            else:
                m["cost"] += c
                total_cost += c
            if r.get("role") == "stage":
                rows.append((r["agent"].removeprefix("stage-"), r.get("model", ""),
                             round((r.get("duration_ms") or 0) / 60000, 1), t.get("total"),
                             f"${c:.2f}" if c is not None else ""))
    verify = manifest.get("verify") or {}
    meta = {
        "type": "research-run",
        "run": tag,
        "tier": manifest.get("profile"),
        "status": manifest.get("status"),
        "gate_passed": verify.get("passed"),
        "started": manifest.get("started_at"),
        "finished": manifest.get("updated_at"),
        "project": project or "",
        "tags": ["research-run"],
    }
    if priced and by_model:
        meta["est_cost_usd"] = round(total_cost, 2)
    lines = [f"# Research run `{tag}`", ""]
    if report_rel:
        lines += [f"**Report:** [[{report_rel[:-3]}]]", ""]
    lines += ["## Query", "", *("> " + ln for ln in query_body.strip().splitlines()), ""]
    lines += ["## Stages", "", "| Stage | Model | Minutes | Tokens | Est. cost |", "|---|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows] or ["| (single-session run) | | | | |"]
    lines += ["", "## Spend by model", "", "| Model | Sessions | Tokens | Est. cost |", "|---|---|---|---|"]
    money = lambda v: f"${v['cost']:.2f}" if priced else ""  # noqa: E731
    lines += [f"| {k} | {v['sessions']} | {v['tokens']:,} | {money(v)} |" for k, v in by_model.items()]
    if priced and by_model:
        lines += ["", f"Estimated total: **${total_cost:.2f}** (list prices; not a bill)."]
    lines += ["", f"## Sources ({len(new) + len(existing)})", ""]
    lines += [f"- [[{p[:-3]}]]" + (" (already in library)" if p in existing else "") for p in sorted(new + existing)]
    if interim:
        lines += ["", "## Interim analyses", "", *(f"- [[{p[:-3]}]]" for p in interim)]
    lines += ["", "Working files (claims, stage contracts, logs) stay in the research working vault."]
    return join_frontmatter(meta, "\n".join(lines) + "\n")

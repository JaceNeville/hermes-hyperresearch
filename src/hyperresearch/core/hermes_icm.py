"""ICM mode: each pipeline step runs as its own stage, sequenced by code.

    hpr hermes icm "QUERY" [--tier light|full|auto]

Why: in single-session mode the orchestrator carries every step's procedure,
tool output, and subagent report in one ever-growing context, and every turn
re-reads all of it. Measured on the light benchmark: ~120k tokens per turn on
average, ~12-16M tokens per run, almost all of it re-reading.

ICM (stage folders + load lists + file handoffs) fixes that structurally:

* **Code sequences, models work.** A Python loop walks the tier's steps. No
  model is paid to remember where the run is.
* **One fresh session per stage.** Each stage starts empty, reads only what
  its load list names, writes its outputs to disk, and exits.
* **Per-run structure, built programmatically.** Every run gets its own
  `research/runs/<tag>/stages/NN_name/` tree with a generated CONTEXT.md
  (the stage contract), the session log, and a handoff note. Nothing is
  shared between runs except the source library, so folders can't drift.
* **Code checks each stage's outputs** before moving on, resumes the stage
  once if something is missing, and runs the ship gate itself.

Upstream step files are used unchanged: a stage's CONTEXT.md points at the
step file and overrides only the "return to the orchestrator" plumbing.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from hyperresearch.core import hermes

# ---------------------------------------------------------------------------
# Stage catalogue
# ---------------------------------------------------------------------------

TIER_STEPS = {
    "light": ["1", "2", "10", "15", "16"],
    "full": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "14.5", "15", "16"],
}

STEP_FILES = {
    "1": "hyperresearch-1-decompose",
    "2": "hyperresearch-2-width-sweep",
    "3": "hyperresearch-3-contradiction-graph",
    "4": "hyperresearch-4-loci-analysis",
    "5": "hyperresearch-5-depth-investigation",
    "6": "hyperresearch-6-cross-locus-reconcile",
    "7": "hyperresearch-7-source-tensions",
    "8": "hyperresearch-8-corpus-critic",
    "9": "hyperresearch-9-evidence-digest",
    "10": "hyperresearch-10-triple-draft",
    "11": "hyperresearch-11-synthesize",
    "12": "hyperresearch-12-critics",
    "13": "hyperresearch-13-gap-fetch",
    "14": "hyperresearch-14-patcher",
    "14.5": "hyperresearch-14-5-cite-check",
    "15": "hyperresearch-15-polish",
    "16": "hyperresearch-16-readability-audit",
}

R = "research/runs/{tag}"
REPORT = "research/notes/final_report_{tag}.md"


@dataclass
class Stage:
    step: str
    title: str
    loads: list[str]
    outputs: list[str]  # paths code checks; "@notes" = >=1 vault note tagged with the run
    web: bool = False
    extra: str = ""
    avoid: list[str] = field(default_factory=list)


_SHIMS = R + "/shims/"
_T = R + "/temp/"
_BASE = [R + "/query.md", R + "/scaffold.md", R + "/prompt-decomposition.json"]
_NOTES = "source notes for this run: `note list --tag {tag}` for the list, `note show <id>` only for the ones you need"
_NO_BULK = "full bodies of every source note; read only the notes the step names"
_CRITICS = [R + f"/critic-findings-{c}.json" for c in ("width", "depth", "instruction", "dialectic")]


def _fetch_rule(urls: int) -> str:
    return (
        f"**Fetcher batching (this runtime):** give each `hyperresearch-fetcher` at most **{urls} URLs**, "
        "and launch every fetcher of a wave in **one** `hpr hermes spawn` call. The spawner runs as many "
        "in parallel as memory allows; short batches keep each fetcher's session short, which is where "
        "fetch cost comes from."
    )


STAGES: dict[str, Stage] = {
    "1": Stage(
        "1", "decompose",
        loads=[R + "/query.md"],
        outputs=[R + "/scaffold.md", R + "/prompt-decomposition.json", _T + "coverage-matrix.md", _SHIMS],
        extra="@bootstrap",
    ),
    "2": Stage(
        "2", "width-sweep",
        loads=[*_BASE, _T + "coverage-matrix.md", _SHIMS + "research.md"],
        outputs=["@notes", _T + "coverage-gaps.md"],
        web=True,
        avoid=["full text of fetched source notes (research/notes/*.md); use `note list` and the fetchers' handoffs"],
        extra=_fetch_rule(3),
    ),
    "3": Stage(
        "3", "contradiction-graph",
        loads=[*_BASE, _T + "claims-*.json"],
        outputs=[_T + "contradiction-graph.json", _T + "consensus-claims.json"],
        avoid=[_NO_BULK],
    ),
    "4": Stage(
        "4", "loci-analysis",
        loads=[*_BASE, _SHIMS + "research.md", _T + "coverage-gaps.md", _T + "contradiction-graph.json", _NOTES],
        outputs=[R + "/loci.json"],
        avoid=[_NO_BULK],
    ),
    "5": Stage(
        "5", "depth-investigation",
        loads=[*_BASE, _SHIMS + "research.md", R + "/loci.json", _T + "contradiction-graph.json"],
        outputs=["@interim"],
        web=True,
        avoid=[_NO_BULK, "the investigators' fetched sources; read their interim notes only"],
        extra=_fetch_rule(3),
    ),
    "6": Stage(
        "6", "cross-locus-reconcile",
        loads=[*_BASE, R + "/loci.json", "the interim notes (`note list --tag {tag} --type interim`)", _T + "orchestrator-notes.md"],
        outputs=[R + "/comparisons.md"],
        avoid=[_NO_BULK],
    ),
    "7": Stage(
        "7", "source-tensions",
        loads=[*_BASE, _T + "contradiction-graph.json", R + "/comparisons.md", _NOTES],
        outputs=[_T + "source-tensions.json"],
        avoid=[_NO_BULK],
    ),
    "8": Stage(
        "8", "corpus-critic",
        loads=[*_BASE, _SHIMS + "research.md", R + "/loci.json", R + "/comparisons.md", _T + "source-tensions.json"],
        outputs=[R + "/corpus-critic-gaps.json", _T + "corpus-critic-results.md"],
        web=True,
        avoid=[_NO_BULK],
        extra=_fetch_rule(3),
    ),
    "9": Stage(
        "9", "evidence-digest",
        loads=[*_BASE, _T + "claims-*.json", _T + "contradiction-graph.json", _T + "consensus-claims.json"],
        outputs=[_T + "evidence-digest.md"],
        avoid=[_NO_BULK],
    ),
    "10": Stage(
        "10", "draft",
        loads=[*_BASE, _SHIMS + "drafting.md", "@draft-inputs"],
        outputs=["@draft"],
        extra="@draft-rules",
    ),
    "11": Stage(
        "11", "synthesize",
        loads=[*_BASE, _T + "draft-a.md", _T + "draft-b.md", _T + "draft-c.md", _T + "evidence-digest.md",
               _T + "source-tensions.json", R + "/comparisons.md"],
        outputs=[REPORT, _T + "synthesis-pass1.md"],
        avoid=[_NO_BULK],
        extra="@length",
    ),
    "12": Stage(
        "12", "critics",
        loads=[*_BASE, _SHIMS + "critics.md", REPORT],
        outputs=_CRITICS,
        avoid=[_NO_BULK],
    ),
    "13": Stage(
        "13", "gap-fetch",
        loads=[*_BASE, _SHIMS + "research.md", R + "/critic-findings-*.json", _T + "evidence-digest.md"],
        outputs=[_T + "post-critic-fetch-log.md"],
        web=True,
        avoid=[_NO_BULK],
        extra=_fetch_rule(3),
    ),
    "14": Stage(
        "14", "patcher",
        loads=[*_BASE, _SHIMS + "critics.md", REPORT, R + "/critic-findings-*.json", _T + "evidence-digest.md"],
        outputs=[R + "/patch-log.json"],
        avoid=[_NO_BULK],
    ),
    "14.5": Stage(
        "14.5", "cite-check",
        loads=[*_BASE, REPORT],
        outputs=[R + "/cite-check-pairs.json", R + "/cite-check-findings.json"],
        avoid=[_NO_BULK],
        extra="@cite-precheck",
    ),
    "15": Stage("15", "polish", loads=[R + "/query.md", REPORT, _SHIMS + "polish.md"], outputs=[R + "/polish-log.json"]),
    "16": Stage("16", "readability-audit", loads=[R + "/query.md", REPORT], outputs=[R + "/readability-recommendations.json"]),
}

_DRAFT_INPUTS = {
    "light": ["the 8-15 most relevant source notes (via `note show`)"],
    "full": [_T + "evidence-digest.md", _T + "source-tensions.json", R + "/comparisons.md",
             R + "/loci.json", _T + "orchestrator-notes.md"],
}


# vault_tag -> (low, high) word target, filled in once stage 1 has classified
# the response format. Module-level so render_context stays a pure function
# of its arguments plus this lookup.
_WORD_TARGET: dict[str, tuple[int, int]] = {}

# After these stages code runs the mechanical checks that caused expensive
# end-of-run fixes, and fixes them right there in a small session.
CHECKPOINTS = {"11": ("length-in-range",), "14": ("length-in-range",),
               "14.5": ("quote-integrity",), "10": ("length-in-range",)}


def word_target(vault_root: Path, tag: str, tier: str) -> tuple[int, int] | None:
    try:
        from hyperresearch.core.profiles import resolve_profile

        decomp = json.loads((vault_root / R.format(tag=tag) / "prompt-decomposition.json").read_text())
        fmt = decomp.get("response_format")
        prof = resolve_profile("full" if tier == "full" else "light", vault_root / ".hyperresearch" / "config.toml")
        lo, hi = prof.word_targets[fmt]
        return int(lo), int(hi)
    except Exception:
        return None


def stage_tier(cfg: hermes.HermesConfig, step: str, tier: str) -> str:
    """Tier name for a stage's own session. `[hermes.stages]` accepts
    `"10@light"`-style keys to override one run tier."""
    return cfg.stages.get(f"{step}@{tier}") or cfg.stages.get(step) or cfg.orchestrator_tier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP = set(["a", "an", "the", "of", "to", "for", "and", "or", "in", "on", "at", "by", "with", "what", "how", "does", "do", "is", "are", "be", "as", "that", "this", "from", "vs", "versus", "which", "when", "why", "who", "whom", "current", "evidence", "say", "says", "keep", "it", "practical"])


def slugify(query: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOP]
    return "-".join(words[:4]) or "research"


def _section(skill_text: str, start: str, end: str) -> str:
    """Slice a section out of the installed entry skill, so upstream edits flow through."""
    i = skill_text.find(start)
    if i < 0:
        return ""
    j = skill_text.find(end, i + len(start))
    return skill_text[i : j if j > 0 else None].strip()


def _hpr_json(hpr: str, args: list[str], cwd: Path) -> dict:
    p = subprocess.run([hpr, *args, "--json"], cwd=cwd, capture_output=True, text=True, timeout=600)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": {"message": (p.stdout + p.stderr)[-800:]}}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Stage contract (CONTEXT.md)
# ---------------------------------------------------------------------------


def render_context(stage: Stage, tag: str, tier: str, query: str, hpr: str, skill_text: str, prev_handoffs: str) -> str:
    fmt = lambda p: p.format(tag=tag)  # noqa: E731
    items: list[str] = []
    for p in stage.loads:
        items += _DRAFT_INPUTS["full" if tier == "full" else "light"] if p == "@draft-inputs" else [p]
    loads = "\n".join(f"- `{fmt(p)}`" if p.startswith("research/") else f"- {fmt(p)}" for p in items)
    outputs = []
    for o in stage.outputs:
        outputs.append({
            "@notes": f"- source notes in the vault tagged `{tag}` (fetched by fetcher subagents)",
            "@interim": f"- interim notes (`type: interim`) tagged `{tag}`",
            "@draft": (f"- `{fmt(REPORT)}` (light tier: single draft)" if tier == "light"
                       else f"- `{fmt(R)}/temp/draft-{{a,b,c}}.md`"),
        }.get(o, f"- `{fmt(o)}`"))
    avoid = [
        "`.hyperresearch/hermes/SKILL.md` — the orchestrator entry. Code sequences the run; you don't need it.",
        "other steps' files under `.hyperresearch/hermes/steps/`",
        *(fmt(a) for a in stage.avoid),
    ]
    extra = stage.extra
    if extra in ("@draft-rules", "@length"):
        lo_hi = _WORD_TARGET.get(tag)
        length = (
            f"**Length (checked by code right after this stage):** the final report must land at "
            f"**{lo_hi[0]}-{lo_hi[1]} words**. Aim for about {int((lo_hi[0] + lo_hi[1]) / 2)}."
            if lo_hi else ""
        )
    if extra == "@draft-rules":
        extra = "\n\n".join(x for x in (
            length and length + " Pass this target to every draft-orchestrator; each draft must fit it.",
            "**Draft reading budget (this runtime, overrides the step's 20-50):** give each "
            "draft-orchestrator **12-20** `must_read_note_ids`, and tell it to read "
            f"`research/runs/{tag}/temp/evidence-digest.md` first, then only its must-read notes. "
            "No vault survey, no extra notes. The digest already carries the claims; the notes are for "
            "quoting and detail.",
        ) if x)
    elif extra == "@length":
        extra = length + (" Tell the synthesizer this target explicitly." if length else "")
    if extra == "@cite-precheck":
        extra = (
            "**Steps 14.5.1 and 14.5.2 are already done, by code.** Code ran `citecheck extract`, split "
            "`sampled_for_llm` into batches, ran one `hyperresearch-cite-checker` per batch, and merged "
            f"their findings plus every dangling citation into `research/runs/{tag}/cite-check-findings.json`. "
            "Do NOT re-run extraction or spawn cite-checkers. Start at step 14.5.3 (second patcher pass) "
            "and finish the exit criterion."
        )
    if extra == "@bootstrap":
        extra = (
            "**Before the procedure, finish the bootstrap** (code already minted the vault tag, "
            "initialized the run, and wrote `query.md`). Do these two items from the entry skill:\n\n"
            + _section(skill_text, "4. **Classify modality**", "6. **Seed your plan.**")
        )
    tier_rule = {
        "light": "In this stage's procedure, set `pipeline_tier` to `\"light\"` (operator cap) and note the cap in the scaffold.",
        "full": "In this stage's procedure, set `pipeline_tier` to `\"full\"` (operator choice).",
        "auto": "Classify `pipeline_tier` as the procedure describes.",
    }[tier] if stage.step == "1" else f"Tier for this run: **{tier}**. Follow the {tier}-tier branch of the procedure."

    spawn_contract = _section(skill_text, "## Subagent spawn contract", "\n---")
    return f"""# Stage {stage.step} — {stage.title}

You are ONE stage of a research pipeline that code runs as separate stages (ICM).
This session is fresh: this file is everything you know. Do this stage, then stop.

## Run
- vault_tag: `{tag}`
- run dir: `{fmt(R)}/`
- CLI: `{hpr}` (use this exact path)
- {tier_rule}

## Research query (verbatim — gospel)

{chr(10).join('> ' + line for line in query.strip().splitlines())}

## Load (only these)
{loads}
- the step file: `.hyperresearch/hermes/steps/{STEP_FILES[stage.step]}.md` (read IN FULL, paging as needed)

## Do not load
{chr(10).join('- ' + a for a in avoid)}

## Procedure
Follow the step file. Where it says to return to the entry skill, invoke the next step,
or record the step with `run step`, skip that: code records progress and starts the next
stage. Everything this stage needs from earlier stages is on disk at the paths above.

{extra}

## Outputs (code checks these before the next stage starts)
{chr(10).join(outputs)}

## Tools in this runtime
- Read -> `read_file` (page long files). Write -> `write_file`. Edit -> `patch` (surgical only).
- Shell -> `terminal`.{" Web search -> `web_search` (planning only)." if stage.web else ""}
- You never fetch source pages yourself: `{hpr} fetch` / `fetch-batch` refuse to run here
  (`DELEGATE_FETCH`). Fetching is the fetcher subagents' job.

## Spawning subagents
Write each subagent's full message to its own file under `{fmt(R)}/temp/spawn/`, then launch
all of a step's subagents in ONE command:

```bash
{hpr} hermes spawn --tag {tag} --job AGENT=MSG_FILE --job AGENT=MSG_FILE -j
```

It returns within ~3 minutes. If `"status": "running"`, run `{hpr} hermes wait <batch_id> -j`
until `"status": "done"`. Retry a failed job once; after that note the gap and continue.

{spawn_contract}
## Earlier stages' handoffs
{prev_handoffs or "(none — this is the first stage)"}

## When you're done
Stop once the outputs exist. Your final message is this stage's handoff to the next one:
3-6 lines on what you produced and anything the next stage must know. No report text.
"""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    step: str
    ok: bool
    attempts: int
    missing: list[str]
    session_id: str | None
    tokens: dict
    duration_s: float


def _check_outputs(vault_root: Path, stage: Stage, tag: str, tier: str, hpr: str) -> list[str]:
    missing = []
    for o in stage.outputs:
        if o == "@notes":
            d = _hpr_json(hpr, ["note", "list", "--tag", tag, "--all"], vault_root)
            if not (d.get("ok") and d.get("data")):
                missing.append(f"source notes tagged {tag}")
        elif o == "@interim":
            d = _hpr_json(hpr, ["note", "list", "--tag", tag, "--type", "interim", "--all"], vault_root)
            if not (d.get("ok") and d.get("data")):
                missing.append("interim notes")
        elif o == "@draft":
            paths = [REPORT] if tier == "light" else [R + "/temp/draft-a.md", R + "/temp/draft-b.md", R + "/temp/draft-c.md"]
            missing += [p.format(tag=tag) for p in paths if not (vault_root / p.format(tag=tag)).exists()]
        else:
            p = vault_root / o.format(tag=tag)
            if o.endswith("/"):
                if not (p.is_dir() and any(p.iterdir())):
                    missing.append(o.format(tag=tag))
            elif not p.exists():
                missing.append(o.format(tag=tag))
    return missing


def _run_session(
    vault_root: Path, cfg: hermes.HermesConfig, tier_name: str, toolsets: list[str],
    prompt_file: Path, log_file: Path, resume: str | None = None,
) -> tuple[int, dict]:
    tier = cfg.tier(tier_name)
    cmd = hermes.build_chat_cmd(tier, toolsets, prompt_file, cfg, max_turns=200, workdir=vault_root)
    if resume:
        cmd += ["--resume", resume]
    with open(log_file, "a", encoding="utf-8") as log:
        code = subprocess.call(
            cmd, cwd=vault_root, env=hermes.chat_env(vault_root, hermes.ORCHESTRATOR_ROLE),
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
    return code, hermes._parse_result(log_file)


def _ledger(vault_root: Path, tag: str, row: dict) -> None:
    path = vault_root / R.format(tag=tag) / "temp" / "hermes-spawns.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _write_index(vault_root: Path, tag: str, tier: str, rows: list[dict], sealed: bool) -> None:
    lines = [
        f"# Run {tag}",
        "",
        f"- tier: {tier}",
        f"- status: {'SEALED — do not reuse this folder' if sealed else 'in progress'}",
        f"- updated: {_now()}",
        "",
        "| Stage | Status | Model | Minutes | Tokens |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['stage']} | {r['status']} | {r.get('model', '')} | {r.get('minutes', '')} | {r.get('tokens', '')} |")
    (vault_root / R.format(tag=tag) / "stages" / "RUN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_icm(vault_root: Path, query: str, tier: str, hpr: str, echo=print,
            resume_tag: str | None = None) -> dict:
    from hyperresearch.core import hermes_guard as _g

    changed = _g.changed_files(vault_root)
    if changed:
        raise hermes.HermesError(
            "protected pipeline files changed since install: " + ", ".join(changed)
            + ". Review them; if the change is intended re-run `hpr hermes install`, otherwise "
            "`hpr hermes install` also restores the originals."
        )
    blocked = hermes.write_blocked_reason(vault_root)
    if blocked:
        raise hermes.HermesError(blocked)
    cfg = hermes.load_config(vault_root)
    skill_text = (vault_root / hermes.ENTRY_SKILL).read_text(encoding="utf-8")

    # --- bootstrap, in code -------------------------------------------------
    done_before: set[str] = set()
    if resume_tag:
        # Resume a stopped run (budget stop, crash): finished steps are skipped,
        # their handoffs reload, spend so far still counts against the cap.
        tag = resume_tag
        run_dir = vault_root / R.format(tag=tag)
        stages_dir = run_dir / "stages"
        manifest = run_dir / "run.json"
        if not manifest.exists() or not (stages_dir / "query.txt").exists():
            raise hermes.HermesError(f"no resumable ICM run at {run_dir}")
        done_before = {k for k, v in (json.loads(manifest.read_text()).get("steps") or {}).items()
                       if isinstance(v, dict) and v.get("status") == "done"}
        query = (stages_dir / "query.txt").read_text(encoding="utf-8")
        echo(f"resume {tag}: {len(done_before)} step(s) already done")
    else:
        tag = _hpr_json(hpr, ["vault-tag", slugify(query)], vault_root)["data"]["vault_tag"]
        run_dir = vault_root / R.format(tag=tag)
        stages_dir = run_dir / "stages"
        stages_dir.mkdir(parents=True, exist_ok=True)
        qfile = stages_dir / "query.txt"
        qfile.write_text(query.strip() + "\n", encoding="utf-8")
        init = _hpr_json(hpr, ["run", "init", tag, "--profile", "full" if tier == "full" else "light",
                               "--query-file", str(qfile)], vault_root)
        if not init.get("ok"):
            raise hermes.HermesError(f"run init failed: {init.get('error')}")
        echo(f"run {tag}: {run_dir.relative_to(vault_root)}")

    from hyperresearch.core import hermes_guard

    allowed = hermes_guard.allowed_run_names(vault_root)
    steps = list(TIER_STEPS["light" if tier in ("light", "auto") else "full"])
    rows: list[dict] = []
    handoffs: list[str] = []
    results: list[StageResult] = []
    i = 0
    while i < len(steps):
        step = steps[i]
        stage = STAGES[step]
        sdir = stages_dir / f"{int(float(step)):02d}{'_5' if step.endswith('.5') else ''}_{stage.title}"
        sdir.mkdir(exist_ok=True)
        if step in done_before:
            prev = sdir / "handoff.md"
            handoffs.append(f"### Stage {step} — {stage.title}\n"
                            + (prev.read_text(encoding="utf-8").strip() if prev.exists() else ""))
            rows.append({"stage": f"{step} {stage.title}", "status": "done (earlier)"})
            if step == "1":
                wt = word_target(vault_root, tag, tier)
                if wt:
                    _WORD_TARGET[tag] = wt
            i += 1
            continue
        ctx = render_context(stage, tag, tier, query, hpr, skill_text, "\n\n".join(handoffs[-3:]))
        (sdir / "CONTEXT.md").write_text(ctx, encoding="utf-8")
        tier_name = stage_tier(cfg, step, tier)
        toolsets = ["file", "terminal"] + (["web"] if stage.web else [])
        log = sdir / "session.log.jsonl"
        _hpr_json(hpr, ["run", "step", tag, step, "--status", "running"], vault_root)
        if step == "14.5":
            pre = cite_precheck(vault_root, cfg, tag, hpr, sdir, echo)
            rows.append(pre)
            _write_index(vault_root, tag, tier, rows, sealed=False)
            if pre.get("missing_batches"):
                # A patch pass on partial findings looks finished but isn't: stop, resumable.
                echo(f"  cite-check batches {pre['missing_batches']} produced no findings after a retry; stopping")
                _hpr_json(hpr, ["run", "block", tag, "--on", "stage-14.5-cite-checkers"], vault_root)
                return _summary(vault_root, tag, tier, results, None, hpr)
        echo(f"  stage {step} {stage.title} ({cfg.tier(tier_name).model}) ...")
        root_before = hermes_guard.snapshot_root(vault_root)
        t0 = time.monotonic()
        code, res = _run_session(vault_root, cfg, tier_name, toolsets, sdir / "CONTEXT.md", log)
        tokens = dict(res.get("tokens") or {})
        missing = _check_outputs(vault_root, stage, tag, tier, hpr)
        attempts = 1
        if missing and res.get("session_id"):
            fix = sdir / "resume.md"
            fix.write_text(
                "Code checked this stage's outputs and these are missing:\n"
                + "\n".join(f"- {m}" for m in missing)
                + "\n\nFinish the stage so they exist, then stop with your handoff.\n",
                encoding="utf-8",
            )
            code, res2 = _run_session(vault_root, cfg, tier_name, toolsets, fix, log, resume=res["session_id"])
            for k, v in (res2.get("tokens") or {}).items():
                tokens[k] = int(tokens.get(k) or 0) + int(v or 0)
            res = res2 or res
            missing = _check_outputs(vault_root, stage, tag, tier, hpr)
            attempts = 2
        dur = time.monotonic() - t0
        _guard_after(vault_root, tag, sdir.name, root_before, allowed, echo)
        handoff = (res.get("text") or "").strip()[-1500:]
        (sdir / "handoff.md").write_text(handoff + "\n", encoding="utf-8")
        handoffs.append(f"### Stage {step} — {stage.title}\n{handoff}")
        ok = not missing
        _ledger(vault_root, tag, {
            "agent": f"stage-{step}-{stage.title}", "role": "stage", "tier": tier_name,
            "model": cfg.tier(tier_name).model, "status": "done" if ok else "failed",
            "exit_code": code, "tokens": tokens, "duration_ms": int(dur * 1000),
        })
        results.append(StageResult(step, ok, attempts, missing, res.get("session_id"), tokens, dur))
        rows.append({"stage": f"{step} {stage.title}", "status": "done" if ok else f"MISSING {missing}",
                     "model": cfg.tier(tier_name).model, "minutes": round(dur / 60, 1),
                     "tokens": tokens.get("total")})
        _write_index(vault_root, tag, tier, rows, sealed=False)
        echo(f"    {'ok' if ok else 'MISSING ' + str(missing)} in {dur / 60:.1f} min, {tokens.get('total')} tokens")
        if not ok:
            _hpr_json(hpr, ["run", "block", tag, "--on", f"stage-{step}-outputs"], vault_root)
            return _summary(vault_root, tag, tier, results, None, hpr)
        _hpr_json(hpr, ["run", "step", tag, step, "--status", "done"], vault_root)
        if step == "1":
            wt = word_target(vault_root, tag, tier)
            if wt:
                _WORD_TARGET[tag] = wt
        if step in CHECKPOINTS and not (step == "10" and tier == "full"):
            fixed = _checkpoint(vault_root, cfg, tag, step, CHECKPOINTS[step], hpr, stages_dir, skill_text, echo)
            if fixed:
                rows.append(fixed)
                _write_index(vault_root, tag, tier, rows, sealed=False)
        spent = _spent(vault_root, tag)
        if cfg.max_cost_usd and spent is not None and spent > cfg.max_cost_usd:
            echo(f"  budget: ~${spent:.2f} > max_cost_usd ${cfg.max_cost_usd:.2f}; stopping")
            _hpr_json(hpr, ["run", "block", tag, "--on", "budget"], vault_root)
            rows.append({"stage": "BUDGET STOP", "status": f"~${spent:.2f} > ${cfg.max_cost_usd:.2f}"})
            _write_index(vault_root, tag, tier, rows, sealed=False)
            return _summary(vault_root, tag, tier, results, None, hpr)

        if step == "1" and tier == "auto":
            try:
                decomp = json.loads((run_dir / "prompt-decomposition.json").read_text())
                chosen = decomp.get("pipeline_tier", "light")
            except Exception:
                chosen = "light"
            if chosen == "full":
                steps, tier = list(TIER_STEPS["full"]), "full"
            else:
                tier = "light"
        i += 1

    # --- ship gate, in code -------------------------------------------------
    gate = None
    for rnd in range(3):
        _hpr_json(hpr, ["sources", "retractions", "--tag", tag], vault_root)
        gate = _hpr_json(hpr, ["run", "finish", tag], vault_root)
        if _gate_passed(gate):
            break
        failed = _gate_failures(gate)
        echo(f"  gate round {rnd + 1}: failed {failed}")
        gdir = stages_dir / "99_gate-fix"
        gdir.mkdir(exist_ok=True)
        gctx = gdir / f"CONTEXT-{rnd + 1}.md"
        gctx.write_text(
            f"# Gate fix — round {rnd + 1}\n\nRun `{tag}`. The ship gate failed these checks:\n\n"
            f"```json\n{json.dumps(failed, indent=2)[:4000]}\n```\n\n"
            f"Run `{hpr} run finish {tag} --json` to see details, fix the REPORT "
            f"(`{REPORT.format(tag=tag)}`) with surgical `patch` edits, and stop once it passes.\n\n"
            "**If `grounding` failed:** read `research/runs/" + tag + "/grounding.json`. Each finding is a "
            "number, quote or named source in a cited sentence that the cited note does not contain. For each: "
            "open the cited note; if it states the fact differently, correct the sentence to match the note "
            "exactly; if another vault note states it, cite that note instead; otherwise delete the number/"
            "quote/attribution or the whole sentence. Never invent a replacement and never add a citation "
            "you have not opened. A report that says less is acceptable; a report that says something its "
            "sources don't is not.\n\n"
            + _section(skill_text, "**The gate's verdict is final.", "Ship only after")
            + "\n",
            encoding="utf-8",
        )
        t0 = time.monotonic()
        code, res = _run_session(vault_root, cfg, cfg.orchestrator_tier, ["file", "terminal"], gctx,
                                 gdir / "session.log.jsonl")
        _ledger(vault_root, tag, {
            "agent": f"stage-gate-fix-{rnd + 1}", "role": "stage", "tier": cfg.orchestrator_tier,
            "model": cfg.tier(cfg.orchestrator_tier).model, "status": "done", "exit_code": code,
            "tokens": res.get("tokens"), "duration_ms": int((time.monotonic() - t0) * 1000),
        })
    status = _hpr_json(hpr, ["run", "status", tag], vault_root).get("data", {}).get("status")
    rows.append({"stage": "ship gate", "status": status})
    _write_index(vault_root, tag, tier, rows, sealed=status == "done")
    return _summary(vault_root, tag, tier, results, gate, hpr)


def _guard_after(vault_root: Path, tag: str, label: str, root_before: set[str], allowed, echo) -> None:
    """After every stage: restore protected files an agent changed; move strays aside."""
    from hyperresearch.core import hermes_guard

    changed = hermes_guard.changed_files(vault_root)
    events = []
    if changed:
        restored = hermes_guard.restore(vault_root, changed)
        echo(f"    guard: restored {len(restored)} protected file(s) changed during {label}: {restored}")
        events.append({"event": "protected-restored", "stage": label, "files": restored})
    strays = hermes_guard.sweep_strays(vault_root, tag, label, root_before, allowed)
    if strays:
        echo(f"    guard: moved {len(strays)} unrequested file(s) to stages/_strays/{label}/")
        events.append({"event": "strays-moved", "stage": label, "files": strays})
    if events:
        log = vault_root / R.format(tag=tag) / "stages" / "guard.jsonl"
        with open(log, "a", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps({"at": _now(), **e}) + "\n")


def cite_batches(n_pairs: int, size: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) index ranges of at most `size` pairs."""
    size = max(1, size)
    return [(i, min(i + size, n_pairs) - 1) for i in range(0, n_pairs, size)]


def cite_precheck(vault_root: Path, cfg: hermes.HermesConfig, tag: str, hpr: str, sdir: Path, echo) -> dict:
    """Step 14.5.1-14.5.2 in code: extract, batch, spawn checkers, merge findings."""
    run_dir = vault_root / R.format(tag=tag)
    t0 = time.monotonic()
    ext = _hpr_json(hpr, ["citecheck", "extract", tag, "--sample-rate", "1.0"], vault_root)
    pairs_path = run_dir / "cite-check-pairs.json"
    if not ext.get("ok") or not pairs_path.exists():
        raise hermes.HermesError(f"citecheck extract failed: {ext.get('error')}")
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    sampled = pairs.get("sampled_for_llm") or []
    findings = [{
        "verdict": "unsupported", "severity": "critical", "sentence": d.get("sentence", ""),
        "cited_note_id": d.get("note_id") or d.get("citation"), "correct_note_id": None,
        "evidence": "Citation resolves to no vault note (dangling).", "suggested_fix": "swap citation or delete sentence",
    } for d in (pairs.get("dangling") or [])]
    from hyperresearch.core import grounding

    report = vault_root / REPORT.format(tag=tag)
    gres = grounding.check_file(vault_root, report)
    grounding.write_findings(run_dir / "grounding.json", gres)
    for f in gres.findings:
        if f.kind == "dangling":
            continue
        findings.append({
            "verdict": "unsupported", "severity": "critical", "sentence": f.sentence,
            "cited_note_id": f.cited[0] if f.cited else None, "correct_note_id": None,
            "evidence": f"Deterministic grounding check: {f.kind} {f.missing} not found in the cited source(s) {f.cited}.",
            "suggested_fix": "swap to a note that contains it, or remove/soften the unsupported "
                             f"{f.kind}; never replace it with a new unsourced one",
        })
    batches = cite_batches(len(sampled), cfg.cite_batch)
    echo(f"  stage 14.5 pre-check (code): {len(sampled)} pairs -> {len(batches)} cite-checker batch(es) "
         f"of <= {cfg.cite_batch}; grounding {gres.cited_sentences} sentences, {len(gres.findings)} flagged; "
         f"{len(findings)} findings before checkers")
    query = (run_dir / "query.md").read_text(encoding="utf-8") if (run_dir / "query.md").exists() else ""
    jobs = []
    for n, (a, b) in enumerate(batches, 1):
        out = f"{R.format(tag=tag)}/cite-check-findings-{n}.json"
        (run_dir / f"cite-check-findings-{n}.json").unlink(missing_ok=True)
        msg = sdir / f"cite-checker-{n}.md"
        msg.write_text(
            f"RESEARCH QUERY (verbatim, gospel):\n> {query.strip()}\n\nQUERY FILE: {R.format(tag=tag)}/query.md\n\n"
            "PIPELINE POSITION: You are step 14.5 (cite-checker) of the hyperresearch V8 pipeline. "
            "You verify citation-sentence bindings. You do not edit the report.\n\n"
            f"YOUR INPUTS:\n- pairs_file: {R.format(tag=tag)}/cite-check-pairs.json\n"
            f"- your_range: sampled_for_llm[{a}..{b}] (inclusive; {b - a + 1} pairs)\n"
            f"- findings_path: {out}\n- vault_tag: {tag}\n",
            encoding="utf-8",
        )
        jobs.append(("hyperresearch-cite-checker", msg, [out]))

    def _read(n: int) -> list | None:
        try:
            part = json.loads((run_dir / f"cite-check-findings-{n}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return part if isinstance(part, list) else None

    failed: list[int] = []
    if jobs:
        todo = list(range(1, len(jobs) + 1))
        for attempt in (1, 2):  # one retry for batches that left no valid findings file
            batch_id = hermes.create_batch(vault_root, [jobs[n - 1] for n in todo], tag)
            hermes.start_batch_detached(vault_root, batch_id)
            summary = hermes.wait_batch(vault_root, batch_id, cfg.spawn_timeout_s + 120)
            todo = [n for n in todo if _read(n) is None]
            echo(f"    checkers (attempt {attempt}): {summary['counts']}, peak parallel "
                 f"{summary.get('peak_parallel')}" + (f"; no findings file from batch(es) {todo}" if todo else ""))
            if not todo:
                break
        failed = todo
        for n in range(1, len(jobs) + 1):
            findings += _read(n) or []
    (run_dir / "cite-check-findings.json").write_text(json.dumps(findings, indent=1) + "\n", encoding="utf-8")
    return {"stage": "14.5 pre-check (code)", "status": f"{len(batches)} batches, {len(findings)} findings"
            + (f", batches {failed} missing" if failed else ""), "model": "code + cite-checkers",
            "minutes": round((time.monotonic() - t0) / 60, 1), "tokens": "", "missing_batches": failed}


def _gate_passed(gate: dict | None) -> bool:
    data = (gate or {}).get("data") or {}
    return bool(data.get("passed") or (data.get("verify") or {}).get("passed"))


def _gate_failures(gate: dict | None) -> object:
    data = (gate or {}).get("data") or {}
    return (
        data.get("failed_checks")
        or (data.get("verify") or {}).get("failed_checks")
        or [c for c in (data.get("verify") or {}).get("checks", []) if not c.get("ok")]
        or (gate or {}).get("error")
    )


def _spent(vault_root: Path, tag: str) -> float | None:
    """Estimated spend so far from the run ledger and [hermes.prices]; None if unpriced."""
    from hyperresearch.core import hermes_publish

    prices = hermes_publish.load_publish_config(vault_root).prices
    ledger = vault_root / R.format(tag=tag) / "temp" / "hermes-spawns.jsonl"
    if not prices or not ledger.exists():
        return None
    total = 0.0
    for line in ledger.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = hermes_publish.estimate_cost(r.get("tokens") or {}, prices.get(r.get("model", "")))
        if c is None:
            return None
        total += c
    return total


def _checkpoint(vault_root, cfg, tag, step, checks, hpr, stages_dir, skill_text, echo) -> dict | None:
    """Run named mechanical checks now; fix failures in one small, focused session."""
    if not (vault_root / REPORT.format(tag=tag)).exists():
        return None
    v = _hpr_json(hpr, ["run", "verify", tag], vault_root).get("data") or {}
    failed = [c for c in v.get("checks", []) if c.get("name") in checks and not c.get("ok")]
    if not failed:
        return None
    detail = failed
    if any(c["name"] == "quote-integrity" for c in failed):
        lint = _hpr_json(hpr, ["lint", "--rule", "quote-integrity", "--audit-file", REPORT.format(tag=tag)], vault_root)
        detail = failed + [{"quote": i.get("message", "")[:300]} for i in (lint.get("data") or {}).get("issues", [])]
    echo(f"    checkpoint after {step}: {[c['name'] for c in failed]}; fixing")
    d = stages_dir / f"{int(float(step)):02d}{'_5' if step.endswith('.5') else ''}_checkpoint"
    d.mkdir(exist_ok=True)
    lo_hi = _WORD_TARGET.get(tag)
    ctx = d / "CONTEXT.md"
    ctx.write_text(
        f"# Checkpoint after stage {step}\n\nRun `{tag}`. Report: `{REPORT.format(tag=tag)}`.\n"
        f"Code ran mechanical checks and these failed:\n\n```json\n{json.dumps(detail, indent=1)[:5000]}\n```\n\n"
        "Fix only these, then stop:\n\n"
        + ("- **Length:** spawn ONE `hyperresearch-synthesizer` compression pass "
           f"(`{hpr} hermes spawn --tag {tag} --job hyperresearch-synthesizer=<msg file> -j`) with the report path, "
           f"the target ({lo_hi[0]}-{lo_hi[1]} words, aim {int(sum(lo_hi) / 2)}) and the rule: cut repetition and "
           "weaker supporting evidence, keep every load-bearing claim and citation, keep all H2s. Then confirm "
           "with `wc -w`.\n" if lo_hi and any(c["name"] == "length-in-range" for c in failed) else "")
        + ("- **Quotes:** for each flagged quote, `hpr search` the phrase; if the source has the exact words, "
           "copy them verbatim; otherwise remove the quotation marks and keep the claim as plain prose. "
           "Surgical `patch` edits only.\n" if any(c["name"] == "quote-integrity" for c in failed) else "")
        + f"\nWhen done, `{hpr} run verify {tag} --json` must show these checks ok.\n",
        encoding="utf-8",
    )
    t0 = time.monotonic()
    code, res = _run_session(vault_root, cfg, cfg.orchestrator_tier, ["file", "terminal"], ctx, d / "session.log.jsonl")
    _ledger(vault_root, tag, {
        "agent": f"stage-{step}-checkpoint", "role": "stage", "tier": cfg.orchestrator_tier,
        "model": cfg.tier(cfg.orchestrator_tier).model, "status": "done", "exit_code": code,
        "tokens": res.get("tokens"), "duration_ms": int((time.monotonic() - t0) * 1000),
    })
    after = _hpr_json(hpr, ["run", "verify", tag], vault_root).get("data") or {}
    still = [c["name"] for c in after.get("checks", []) if c.get("name") in checks and not c.get("ok")]
    echo(f"    checkpoint {'ok' if not still else 'still failing ' + str(still)}")
    return {"stage": f"{step} checkpoint", "status": "fixed" if not still else f"still {still}",
            "model": cfg.tier(cfg.orchestrator_tier).model,
            "minutes": round((time.monotonic() - t0) / 60, 1), "tokens": (res.get("tokens") or {}).get("total")}


def _summary(vault_root: Path, tag: str, tier: str, results: list[StageResult], gate: dict | None, hpr: str) -> dict:
    report = vault_root / REPORT.format(tag=tag)
    esc = _hpr_json(hpr, ["escalation", "list", "--status", "queued", "--tag", tag], vault_root)
    return {
        "vault_tag": tag,
        "tier": tier,
        "stages": [
            {"step": r.step, "ok": r.ok, "attempts": r.attempts, "missing": r.missing,
             "minutes": round(r.duration_s / 60, 1), "tokens": r.tokens}
            for r in results
        ],
        "gate_passed": _gate_passed(gate),
        "report": str(report) if report.exists() else None,
        "queued_escalations": (esc.get("data") if esc.get("ok") else None),
    }

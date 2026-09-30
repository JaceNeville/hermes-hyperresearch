"""`hpr hermes ...` — run the pipeline on Hermes Agent (fork addition).

    hpr hermes install [PATH]            render prompts + write default model tiers
    hpr hermes spawn --job AGENT=MSG ... launch subagents (parallel, per-role models)
    hpr hermes wait BATCH_ID             block until a spawn batch finishes
    hpr hermes run "QUERY"               run the whole pipeline end to end
    hpr hermes models                    show the role -> tier -> model table
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import typer

from hyperresearch.cli._output import console, output
from hyperresearch.models.output import error, success

app = typer.Typer(no_args_is_help=True)

# Stay under the Hermes terminal tool's default 180 s foreground timeout, so a
# wait never gets its command killed. The batch runs on regardless.
WAIT_CAP_S = 150


def _fail(msg: str, code: str, json_output: bool) -> NoReturn:
    if json_output:
        output(error(msg, code), json_mode=True)
    else:
        console.print(f"[red]Error:[/] {msg}")
    raise typer.Exit(1)


def _vault(json_output: bool, path: Path | None = None):
    from hyperresearch.core.vault import Vault, VaultError

    try:
        return Vault.discover(path)
    except VaultError as e:
        _fail(str(e), "NO_VAULT", json_output)


@app.command("install")
def install(
    path: str = typer.Argument(".", help="Project directory (vault root)."),
    profile: str | None = typer.Option(None, "--profile", help="Scale gear to render (default: persisted gear)."),
    json_output: bool = typer.Option(False, "--json", "-j"),
) -> None:
    """Install the pipeline for Hermes Agent under .hyperresearch/hermes/."""
    from hyperresearch.core.agent_docs import _resolve_executable
    from hyperresearch.core.config import VaultConfig
    from hyperresearch.core.hermes import HermesError, load_config
    from hyperresearch.core.hooks import install_hermes
    from hyperresearch.core.profiles import ProfileError, resolve_profile
    from hyperresearch.core.vault import Vault, VaultError

    root = Path(path).resolve()
    try:
        vault = Vault.discover(root)
        created = False
    except VaultError:
        # platforms=() : no CLAUDE.md / AGENTS.md written into the project.
        vault = Vault.init(root, platforms=())
        created = True
    cfg_path = vault.root / ".hyperresearch" / "config.toml"
    gear = profile or VaultConfig.load(cfg_path).pipeline_profile
    try:
        resolve_profile(gear, cfg_path if cfg_path.exists() else None)
    except ProfileError as e:
        _fail(str(e), "UNKNOWN_PROFILE", json_output)
    actions = install_hermes(vault.root, hpr_path=_resolve_executable(), profile=gear)
    try:
        load_config(vault.root)
    except HermesError as e:
        _fail(f"hermes.toml: {e}", "BAD_CONFIG", json_output)
    data = {"vault": str(vault.root), "created": created, "gear": gear, "actions": actions}
    if json_output:
        output(success(data, vault=str(vault.root)), json_mode=True)
        return
    console.print(f"[green]Hermes install[/] ({'new vault' if created else 'existing vault'}): {vault.root}")
    for a in actions or ["[dim]already up to date[/]"]:
        console.print(f"  {a}")


@app.command("models")
def models(json_output: bool = typer.Option(False, "--json", "-j")) -> None:
    """Show which model each subagent role runs on."""
    from hyperresearch.core.hermes import HermesError, load_config

    vault = _vault(json_output)
    try:
        cfg = load_config(vault.root)
    except HermesError as e:
        _fail(str(e), "BAD_CONFIG", json_output)
    rows = [
        {"role": r, "tier": t, "provider": cfg.tier(t).provider, "model": cfg.tier(t).model}
        for r, t in sorted(cfg.roles.items())
    ]
    orch = cfg.tier(cfg.orchestrator_tier)
    data = {
        "default_tier": cfg.default_tier,
        "orchestrator": {"tier": cfg.orchestrator_tier, "model": orch.model, "provider": orch.provider},
        "max_parallel": cfg.max_parallel,
        "roles": rows,
    }
    if json_output:
        output(success(data, vault=str(vault.root)), json_mode=True)
        return
    console.print(f"default run tier: [bold]{cfg.default_tier}[/]   parallel: {cfg.max_parallel}")
    console.print(f"orchestrator: {cfg.orchestrator_tier} -> {orch.model}")
    for r in rows:
        console.print(f"  {r['role']:<24} {r['tier']:<10} {r['model']}")


def _parse_jobs(jobs: list[str], json_output: bool) -> list[tuple[str, Path]]:
    parsed = []
    for spec in jobs:
        if "=" not in spec:
            _fail(f"--job must be AGENT=MESSAGE_FILE, got '{spec}'", "BAD_JOB", json_output)
        agent, msg = spec.split("=", 1)
        parsed.append((agent.strip(), Path(msg.strip())))
    return parsed


@app.command("spawn")
def spawn(
    job: list[str] = typer.Option(..., "--job", help="AGENT=MESSAGE_FILE. Repeat for parallel jobs."),
    tag: str | None = typer.Option(None, "--tag", help="Run vault_tag (for the token ledger)."),
    wait: int = typer.Option(WAIT_CAP_S, "--wait", help=f"Seconds to block (max {WAIT_CAP_S}); 0 = return at once."),
    json_output: bool = typer.Option(False, "--json", "-j"),
) -> None:
    """Launch subagents as separate Hermes processes, each on its role's model tier."""
    from hyperresearch.core import hermes

    vault = _vault(json_output)
    try:
        batch_id = hermes.create_batch(vault.root, _parse_jobs(job, json_output), tag)
        hermes.start_batch_detached(vault.root, batch_id)
        summary = hermes.wait_batch(vault.root, batch_id, min(max(wait, 0), WAIT_CAP_S))
    except hermes.HermesError as e:
        _fail(str(e), "SPAWN_ERROR", json_output)
    _emit_batch(summary, vault, json_output)


@app.command("wait")
def wait_cmd(
    batch_id: str = typer.Argument(...),
    wait: int = typer.Option(WAIT_CAP_S, "--wait", help=f"Seconds to block (max {WAIT_CAP_S})."),
    json_output: bool = typer.Option(False, "--json", "-j"),
) -> None:
    """Block until a spawn batch finishes (or the wait cap elapses)."""
    from hyperresearch.core import hermes

    vault = _vault(json_output)
    try:
        summary = hermes.wait_batch(vault.root, batch_id, min(max(wait, 0), WAIT_CAP_S))
    except hermes.HermesError as e:
        _fail(str(e), "SPAWN_ERROR", json_output)
    _emit_batch(summary, vault, json_output)


def _emit_batch(summary: dict, vault, json_output: bool) -> None:
    if json_output:
        output(success(summary, vault=str(vault.root)), json_mode=True)
        return
    console.print(f"batch {summary['batch_id']}: [bold]{summary['status']}[/] {summary['counts']} peak parallel {summary.get('peak_parallel')}")
    for j in summary["jobs"]:
        console.print(f"  [{j['index']}] {j['agent']} ({j['model']}): {j['status']}")
    if summary["status"] != "done":
        console.print(f"[dim]still running — `hpr hermes wait {summary['batch_id']}`[/]")


# ---------------------------------------------------------------------------
# End-to-end run: one orchestrator session, supervised until the gate passes
# ---------------------------------------------------------------------------


def _newest_run(vault, since: float) -> dict | None:
    from hyperresearch.core.runs import latest_run_tag, load_manifest

    tag = latest_run_tag(vault)
    if not tag:
        return None
    manifest = load_manifest(vault, tag)
    run_dir = vault.run_dir(tag)
    if (run_dir / "run.json").stat().st_mtime < since:
        return None
    return {"tag": tag, "manifest": manifest}


def _session_id(log_path: Path) -> str | None:
    from hyperresearch.core.hermes import _parse_result

    return _parse_result(log_path).get("session_id")


@app.command("intake")
def intake_cmd(
    question: str = typer.Argument(None, help="Rough research question (omit with --id to answer questions)."),
    intake_id: str | None = typer.Option(None, "--id", help="Continue an intake that asked questions."),
    answers: str = typer.Option("", "--answers", help="Answers to the intake's questions, free text."),
    json_output: bool = typer.Option(False, "--json", "-j"),
) -> None:
    """Scope a research question before any research runs: build the prompt or ask up to 3 questions."""
    from hyperresearch.core import hermes, hermes_intake

    vault = _vault(json_output)
    if not (question or intake_id):
        _fail("give a rough question, or --id with --answers", "NO_QUESTION", json_output)
    try:
        res = hermes_intake.run_intake(vault.root, question or "", answers, intake_id)
    except hermes.HermesError as e:
        _fail(str(e), "INTAKE_ERROR", json_output)
    _emit_intake(res, vault, json_output)
    if res.status != "ready":
        raise typer.Exit(3)


def _emit_intake(res, vault, json_output: bool) -> None:
    d = res.as_dict()
    if json_output:
        output(success(d, vault=str(vault.root)), json_mode=True)
        return
    console.print(f"intake {res.intake_id}: [bold]{res.status}[/] (suggested tier: {res.tier or '-'})")
    if res.status == "ready":
        console.print(res.prompt)
        console.print(f"[dim]start it: hpr hermes icm --from-intake {res.intake_id}[/]")
    elif res.status == "needs_input":
        for i, q in enumerate(res.questions, 1):
            console.print(f"  {i}. {q}")
        console.print(f"[dim]answer: hpr hermes intake --id {res.intake_id} --answers \"...\"[/]")
    else:
        console.print(f"[red]{res.error}[/] (see {res.path})")
    for a in res.assumptions:
        console.print(f"  [dim]assumed: {a}[/]")


@app.command("icm")
def icm(
    query: str = typer.Argument(None, help="Research query (or use --query-file)."),
    query_file: Path | None = typer.Option(None, "--query-file", help="Read the query from a file."),
    intake: bool = typer.Option(False, "--intake", help="Scope the query first; stop with questions if it needs them."),
    from_intake: str | None = typer.Option(None, "--from-intake", help="Run the finished prompt of this intake."),
    tier: str | None = typer.Option(None, "--tier", help="light | full | auto (default: hermes.toml default_tier)."),
    publish: bool = typer.Option(False, "--publish", help="Publish to the Obsidian vault when the gate passes."),
    project: str | None = typer.Option(None, "--project", help="Obsidian project folder the report belongs to."),
    resume: str | None = typer.Option(None, "--resume", help="Continue a stopped run (vault tag); finished steps are skipped."),
    json_output: bool = typer.Option(False, "--json", "-j"),
) -> None:
    """Run the pipeline ICM-style: code sequences stages, each a fresh session."""
    from hyperresearch.core import hermes, hermes_icm
    from hyperresearch.core.agent_docs import _resolve_executable

    vault = _vault(json_output)
    if not hermes.is_installed(vault.root):
        _fail("not installed for Hermes; run `hpr hermes install` first", "NOT_INSTALLED", json_output)
    if query_file:
        query = query_file.read_text(encoding="utf-8")
    intake_tier = None
    if from_intake or intake:
        from hyperresearch.core import hermes_intake

        try:
            res = (hermes_intake.load_ready(vault.root, from_intake) if from_intake
                   else hermes_intake.run_intake(vault.root, query or ""))
        except hermes.HermesError as e:
            _fail(str(e), "INTAKE_ERROR", json_output)
        if res.status != "ready":
            _emit_intake(res, vault, json_output)
            raise typer.Exit(3)
        query, intake_tier = res.prompt, res.tier or None
    if resume:
        query = query or "(resumed)"
    if not query or not query.strip():
        _fail("empty research query", "NO_QUERY", json_output)
    try:
        cfg = hermes.load_config(vault.root)
        tier_cap = tier or intake_tier or cfg.default_tier
        if tier_cap not in ("light", "full", "auto"):
            _fail("--tier must be light, full, or auto", "BAD_TIER", json_output)
        echo = (lambda *_: None) if json_output else (lambda m: console.print(m))
        data = hermes_icm.run_icm(vault.root, query, tier_cap, _resolve_executable(), echo=echo,
                                  resume_tag=resume)
    except hermes.HermesError as e:
        _fail(str(e), "ICM_ERROR", json_output)
    data["spend"] = _spawn_spend(vault.root, data["vault_tag"])
    ok = data["gate_passed"]
    if ok and (publish or project):
        from hyperresearch.core import hermes_publish

        try:
            data["published"] = hermes_publish.publish_run(
                vault.root, data["vault_tag"], _resolve_executable(), project=project
            ).as_dict()
        except hermes.HermesError as e:
            data["published"] = {"error": str(e)}
    if json_output:
        output(success(data, vault=str(vault.root)) if ok else error(json.dumps(data), "RUN_NOT_DONE"), json_mode=True)
    else:
        console.print(json.dumps(data, indent=2))
    if not ok:
        raise typer.Exit(1)


@app.command("publish")
def publish_cmd(
    vault_tag: str = typer.Argument(..., help="Run to publish."),
    project: str | None = typer.Option(None, "--project", help="Obsidian project folder the report belongs to."),
    json_output: bool = typer.Option(False, "--json", "-j"),
) -> None:
    """Publish a finished run (report, sources, run record) into the Obsidian vault."""
    from hyperresearch.core import hermes, hermes_publish
    from hyperresearch.core.agent_docs import _resolve_executable

    vault = _vault(json_output)
    try:
        res = hermes_publish.publish_run(vault.root, vault_tag, _resolve_executable(), project=project)
    except hermes.HermesError as e:
        _fail(str(e), "PUBLISH_ERROR", json_output)
    d = res.as_dict()
    if json_output:
        output(success(d, vault=str(vault.root)), json_mode=True)
    else:
        console.print(f"report: {d['report']}\nrun record: {d['run_record']}")
        console.print(f"sources: {len(d['sources_new'])} new, {len(d['sources_existing'])} already in library")


@app.command("run")
def run(
    query: str = typer.Argument(None, help="Research query (or use --query-file)."),
    query_file: Path | None = typer.Option(None, "--query-file", help="Read the query from a file."),
    tier: str | None = typer.Option(None, "--tier", help="light | full | auto (default: hermes.toml default_tier)."),
    budget: float | None = typer.Option(None, "--budget", help="USD spend ceiling recorded in the run manifest."),
    max_resumes: int = typer.Option(3, "--max-resumes", help="Times to resume the orchestrator if it stops early."),
    json_output: bool = typer.Option(False, "--json", "-j"),
) -> None:
    """Run the full pipeline under a supervised Hermes orchestrator session."""
    from hyperresearch.core import hermes
    from hyperresearch.core.agent_docs import _resolve_executable

    vault = _vault(json_output)
    if not hermes.is_installed(vault.root):
        _fail("not installed for Hermes; run `hpr hermes install` first", "NOT_INSTALLED", json_output)
    if query_file:
        query = query_file.read_text(encoding="utf-8")
    if not query or not query.strip():
        _fail("empty research query", "NO_QUERY", json_output)
    try:
        cfg = hermes.load_config(vault.root)
    except hermes.HermesError as e:
        _fail(str(e), "BAD_CONFIG", json_output)
    tier_cap = tier or cfg.default_tier
    if tier_cap not in ("light", "full", "auto"):
        _fail("--tier must be light, full, or auto", "BAD_TIER", json_output)

    hpr = _resolve_executable()
    orch = cfg.tier(cfg.orchestrator_tier)
    logs = vault.root / hermes.LOGS_DIR
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    started = time.time()

    prompt_path = logs / f"{stamp}-orchestrator.prompt.md"
    prompt_path.write_text(hermes.orchestrator_prompt(query, tier_cap, budget, hpr), encoding="utf-8")

    session: str | None = None
    attempts = []
    run_info = None
    for attempt in range(max_resumes + 1):
        log_path = logs / f"{stamp}-orchestrator-{attempt}.log.jsonl"
        if attempt == 0:
            cmd = hermes.build_chat_cmd(orch, hermes.ORCHESTRATOR_TOOLSETS, prompt_path, cfg, max_turns=500, workdir=vault.root)
        else:
            cont = logs / f"{stamp}-continue-{attempt}.md"
            cont.write_text(hermes.continue_prompt(hpr), encoding="utf-8")
            cmd = hermes.build_chat_cmd(orch, hermes.ORCHESTRATOR_TOOLSETS, cont, cfg, max_turns=500, workdir=vault.root)
            if session:
                cmd += ["--resume", session]
        if not json_output:
            console.print(f"[dim]orchestrator attempt {attempt} ({orch.model}) -> {log_path.relative_to(vault.root)}[/]")
        with open(log_path, "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, cwd=vault.root, env=hermes.chat_env(vault.root, hermes.ORCHESTRATOR_ROLE), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        session = _session_id(log_path) or session
        run_info = _newest_run(vault, started)
        status = run_info["manifest"]["status"] if run_info else None
        attempts.append({"attempt": attempt, "exit_code": code, "run_status": status, "log": str(log_path.relative_to(vault.root))})
        if status in ("done", "blocked", "failed"):
            break

    data = {"attempts": attempts, "tier_cap": tier_cap}
    if run_info:
        tag = run_info["tag"]
        report = vault.root / "research" / "notes" / f"final_report_{tag}.md"
        data.update(
            vault_tag=tag,
            status=run_info["manifest"]["status"],
            report=str(report) if report.exists() else None,
            spend=_spawn_spend(vault.root, tag),
        )
    ok = bool(run_info) and run_info["manifest"]["status"] == "done"
    if json_output:
        output(success(data, vault=str(vault.root)) if ok else error(json.dumps(data), "RUN_NOT_DONE"), json_mode=True)
    else:
        console.print(json.dumps(data, indent=2))
    if not ok:
        raise typer.Exit(1)


def _spawn_spend(vault_root: Path, tag: str) -> dict:
    """Roll up subagent token usage by model from the run's spawn ledger."""
    path = vault_root / "research" / "runs" / tag / "temp" / "hermes-spawns.jsonl"
    by_model: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            m = by_model.setdefault(row.get("model") or "?", {"spawns": 0, "input": 0, "output": 0, "cache_read": 0})
            m["spawns"] += 1
            t = row.get("tokens") or {}
            for k in ("input", "output", "cache_read"):
                m[k] += int(t.get(k) or 0)
    return by_model

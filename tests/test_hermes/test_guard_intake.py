"""Guards (locked files, stray sweep), code-driven cite-check batching, intake."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from hyperresearch.core import hermes, hermes_guard, hermes_icm, hermes_intake


@pytest.fixture()
def hvault(tmp_path: Path) -> Path:
    from hyperresearch.core.hooks import install_hermes
    from hyperresearch.core.vault import Vault

    Vault.init(tmp_path, platforms=())
    install_hermes(tmp_path, hpr_path="/opt/hpr", profile="full")
    return tmp_path


# --- locked files -----------------------------------------------------------


def test_install_locks_pipeline_files(hvault: Path):
    lock = json.loads((hvault / hermes_guard.LOCK_FILE).read_text())
    assert str(hermes.ENTRY_SKILL) in lock and str(hermes.CONFIG_FILE) in lock
    assert any(k.startswith(str(hermes.STEPS_DIR)) for k in lock)
    assert any(k.startswith(str(hermes.AGENTS_DIR)) for k in lock)
    mode = (hvault / hermes.ENTRY_SKILL).stat().st_mode
    assert not mode & stat.S_IWUSR
    assert hermes_guard.changed_files(hvault) == []


def test_changed_file_detected_and_restored(hvault: Path):
    step = next((hvault / hermes.STEPS_DIR).glob("*.md"))
    original = step.read_text()
    step.chmod(0o644)
    step.write_text(original + "\nagent was here\n")
    rel = str(step.relative_to(hvault))
    assert hermes_guard.changed_files(hvault) == [rel]
    assert hermes_guard.restore(hvault, [rel]) == [rel]
    assert step.read_text() == original
    assert hermes_guard.changed_files(hvault) == []


def test_run_refuses_when_protected_files_changed(hvault: Path):
    cfg = hvault / hermes.CONFIG_FILE
    cfg.chmod(0o644)
    cfg.write_text(cfg.read_text() + "\n# edited\n")
    with pytest.raises(hermes.HermesError, match="protected pipeline files changed"):
        hermes_icm.run_icm(hvault, "q", "light", "/opt/hpr", echo=lambda *_: None)


def test_reinstall_relocks(hvault: Path):
    from hyperresearch.core.hooks import install_hermes

    cfg = hvault / hermes.CONFIG_FILE
    cfg.chmod(0o644)
    cfg.write_text(cfg.read_text() + "\n# intended edit\n")
    install_hermes(hvault, hpr_path="/opt/hpr", profile="full")
    assert hermes_guard.changed_files(hvault) == []
    assert "intended edit" in cfg.read_text()


# --- stray files ------------------------------------------------------------


def test_strays_moved_real_outputs_kept(hvault: Path):
    tag = "t-abc123"
    run = hvault / "research" / "runs" / tag
    (run / "temp").mkdir(parents=True)
    keep = ["run.json", "scaffold.md", "patch-log.json", "cite-check-findings.json",
            "cite-check-findings-2.json", "critic-findings-width.json", "loci-a.json", "comparisons.md"]
    stray = ["FETCHER_REPORT_BATCH3.md", "RESEARCH_FINDINGS.md", "step8-fetcher-report.md"]
    for n in keep + stray:
        (run / n).write_text("x")
    before = hermes_guard.snapshot_root(hvault)
    (hvault / "PHASE_2_SUMMARY.md").write_text("x")
    moved = hermes_guard.sweep_strays(hvault, tag, "08_corpus-critic", before)
    assert sorted(Path(m).name for m in moved) == sorted([*stray, "PHASE_2_SUMMARY.md"])
    for n in keep:
        assert (run / n).exists(), n
    assert (run / "stages" / "_strays" / "08_corpus-critic" / "RESEARCH_FINDINGS.md").exists()


# --- cite-check batching ----------------------------------------------------


def test_cite_batches():
    assert hermes_icm.cite_batches(0, 30) == []
    assert hermes_icm.cite_batches(30, 30) == [(0, 29)]
    assert hermes_icm.cite_batches(89, 30) == [(0, 29), (30, 59), (60, 88)]
    assert hermes.load_config(Path("/nonexistent")).cite_batch == 30


def test_cite_stage_told_precheck_done(hvault: Path):
    skill = (hvault / hermes.ENTRY_SKILL).read_text()
    ctx = hermes_icm.render_context(hermes_icm.STAGES["14.5"], "t-x", "full", "q", "/opt/hpr", skill, "")
    assert "already done, by code" in ctx and "14.5.3" in ctx


def test_cite_precheck_spawns_one_checker_per_batch(hvault: Path, monkeypatch):
    tag = "t-cc0001"
    run = hvault / "research" / "runs" / tag
    run.mkdir(parents=True)
    (run / "query.md").write_text("the question")
    (hvault / "research" / "notes" / f"final_report_{tag}.md").write_text("A claim of 30 years [[n]].\n")
    sampled = [{"sentence": f"s{i}", "note_id": "n"} for i in range(65)]
    (run / "cite-check-pairs.json").write_text(json.dumps(
        {"summary": {}, "sampled_for_llm": sampled, "dangling": [{"sentence": "d", "note_id": None, "citation": "[[x]]"}]}))
    monkeypatch.setattr(hermes_icm, "_hpr_json", lambda *a, **k: {"ok": True})
    seen = {}

    def fake_create(vault_root, jobs, t):
        seen["jobs"] = jobs
        for n in range(1, len(jobs) + 1):
            (run / f"cite-check-findings-{n}.json").write_text(json.dumps([{"verdict": "unsupported", "b": n}]))
        return "b1"

    monkeypatch.setattr(hermes, "create_batch", fake_create)
    monkeypatch.setattr(hermes, "start_batch_detached", lambda *a: None)
    monkeypatch.setattr(hermes, "wait_batch", lambda *a: {"counts": {"done": 3}, "peak_parallel": 3})
    cfg = hermes.load_config(hvault)
    sdir = run / "stages" / "14_5_cite-check"
    sdir.mkdir(parents=True)
    row = hermes_icm.cite_precheck(hvault, cfg, tag, "/opt/hpr", sdir, lambda *_: None)
    assert len(seen["jobs"]) == 3
    assert "sampled_for_llm[60..64]" in (sdir / "cite-checker-3.md").read_text()
    merged = json.loads((run / "cite-check-findings.json").read_text())
    # 1 dangling + 1 grounding (cited note "n" missing -> dangling, not double-counted) + 3 batches
    assert len(merged) == 4 and merged[0]["severity"] == "critical"
    assert (run / "grounding.json").exists()
    assert row["status"].startswith("3 batches")


# --- intake -----------------------------------------------------------------


def test_intake_prompt_uses_structure_file():
    p = hermes_intake.build_prompt("how often clean carpet", "IDENTITY: x\nTASK: y", "", "my-file.md")
    assert "IDENTITY: x" in p and "my-file.md" in p and "at most 3" in p
    assert "answers" not in p.split("## The rough question")[1].split("## How to decide")[0]
    p2 = hermes_intake.build_prompt("q", "S", "Residential, Missouri", "f")
    assert "Residential, Missouri" in p2


def _fake_chat(reply: dict | str):
    text = reply if isinstance(reply, str) else json.dumps(reply)

    def call(cmd, stdout=None, **kw):
        stdout.write(json.dumps({"type": "result", "text": text, "tokens": {"total": 10}}) + "\n")
        return 0
    return call


def test_intake_ready_and_needs_input(hvault: Path, monkeypatch):
    monkeypatch.setattr(hermes_intake.subprocess, "call", _fake_chat(
        {"status": "needs_input", "prompt": "", "questions": ["Which market?", "b", "c", "d"], "tier": "full"}))
    r = hermes_intake.run_intake(hvault, "rough q")
    assert r.status == "needs_input" and len(r.questions) == 3 and r.prompt == ""
    monkeypatch.setattr(hermes_intake.subprocess, "call", _fake_chat(
        {"status": "ready", "prompt": "TASK: do it", "questions": [], "assumptions": ["US"], "tier": "light"}))
    r2 = hermes_intake.run_intake(hvault, "", answers="Residential", intake_id=r.intake_id)
    assert r2.status == "ready" and r2.prompt == "TASK: do it" and r2.tier == "light"
    idir = hvault / r2.path
    assert (idir / "prompt.md").read_text().strip() == "TASK: do it"
    assert "Residential" in (idir / "pass-2.md").read_text()
    assert hermes_intake.load_ready(hvault, r.intake_id).status == "ready"


def test_intake_ready_with_questions_is_not_ready(hvault: Path, monkeypatch):
    monkeypatch.setattr(hermes_intake.subprocess, "call", _fake_chat(
        {"status": "ready", "prompt": "TASK", "questions": ["still unsure?"]}))
    assert hermes_intake.run_intake(hvault, "q").status == "needs_input"


def test_intake_bad_reply_is_error(hvault: Path, monkeypatch):
    monkeypatch.setattr(hermes_intake.subprocess, "call", _fake_chat("I think the scope is fine."))
    assert hermes_intake.run_intake(hvault, "q").status == "error"


def test_intake_structure_file_config(hvault: Path, tmp_path: Path):
    f = tmp_path / "structure.md"
    f.write_text("MY STRUCTURE")
    cfg = hvault / hermes.CONFIG_FILE
    cfg.chmod(0o644)
    cfg.write_text(cfg.read_text().replace('structure_file = ""', f'structure_file = "{f}"'))
    text, src = hermes_intake._structure_text(hvault, hermes.load_config(hvault))
    assert text == "MY STRUCTURE" and src == str(f)
    assert os.path.exists(f)


def test_gate_output_is_never_a_stray(tmp_path):
    from hyperresearch.core import hermes_guard
    assert hermes_guard.allowed_run_names(tmp_path).match("grounding.json")


def test_resume_skips_done_steps_and_reuses_handoffs(hvault: Path, monkeypatch):
    """--resume: finished steps don't re-run; their handoffs feed the next stage."""
    tag = "resume-me-abc123"
    run_dir = hvault / hermes_icm.R.format(tag=tag)
    stages = run_dir / "stages"
    stages.mkdir(parents=True)
    (stages / "query.txt").write_text("the original question\n")
    steps = hermes_icm.TIER_STEPS["light"]
    done = steps[:-1]
    (run_dir / "run.json").write_text(json.dumps(
        {"status": "running", "steps": {s: {"status": "done"} for s in done}}))
    first = hermes_icm.STAGES[done[0]]
    d = stages / f"{int(float(done[0])):02d}_{first.title}"
    d.mkdir()
    (d / "handoff.md").write_text("EARLIER-HANDOFF")

    ran, contexts = [], []
    monkeypatch.setattr(hermes_icm, "_hpr_json", lambda hpr, args, cwd: {"ok": True, "data": {}})
    monkeypatch.setattr(hermes_icm, "word_target", lambda *a: None)

    def fake_session(vault_root, cfg, tier_name, toolsets, prompt_file, log, resume=None):
        ran.append(prompt_file.parent.name)
        contexts.append(prompt_file.read_text())
        return 0, {"text": "handoff", "tokens": {}, "session_id": "s"}

    monkeypatch.setattr(hermes_icm, "_run_session", fake_session)
    monkeypatch.setattr(hermes_icm, "_check_outputs", lambda *a: [])
    monkeypatch.setattr(hermes_icm, "_checkpoint", lambda *a: None)
    monkeypatch.setattr(hermes_icm, "_spent", lambda *a: None)
    monkeypatch.setattr(hermes_icm, "_gate_passed", lambda g: True)
    monkeypatch.setattr(hermes_icm, "_summary", lambda *a: {"vault_tag": a[1]})

    out = hermes_icm.run_icm(hvault, "(resumed)", "light", "/opt/hpr", echo=lambda *_: None, resume_tag=tag)
    assert out["vault_tag"] == tag
    last = hermes_icm.STAGES[steps[-1]]
    assert ran == [f"{int(float(steps[-1])):02d}_{last.title}"]
    assert "the original question" in contexts[0]


def test_resume_refuses_unknown_run(hvault: Path):
    with pytest.raises(hermes.HermesError, match="no resumable ICM run"):
        hermes_icm.run_icm(hvault, "(resumed)", "full", "/opt/hpr", echo=lambda *_: None, resume_tag="nope-000000")

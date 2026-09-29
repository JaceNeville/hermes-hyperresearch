"""Tests for ICM mode and memory-aware spawning (fork addition)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperresearch.core import hermes, hermes_icm
from hyperresearch.core.hooks import install_hermes
from hyperresearch.core.vault import Vault


@pytest.fixture
def hvault(tmp_path: Path) -> Path:
    Vault.init(tmp_path, platforms=())
    install_hermes(tmp_path, hpr_path="/opt/hpr")
    return tmp_path


def test_every_tier_step_has_a_stage_and_step_file(hvault: Path):
    for tier, steps in hermes_icm.TIER_STEPS.items():
        for step in steps:
            assert step in hermes_icm.STAGES, (tier, step)
            f = hvault / hermes.STEPS_DIR / f"{hermes_icm.STEP_FILES[step]}.md"
            assert f.is_file(), f


def test_stage_models_configured(hvault: Path):
    cfg = hermes.load_config(hvault)
    model = lambda step, tier: cfg.tier(hermes_icm.stage_tier(cfg, step, tier)).model  # noqa: E731
    assert model("10", "light") == "claude-opus-5-5"  # light: the stage writes the report
    assert model("10", "full") == "claude-sonnet-5"   # full: Opus draft-orchestrators do the writing
    assert model("2", "light") == "claude-sonnet-5"
    assert cfg.tier(cfg.roles["synthesizer"]).model == "claude-opus-5-5"


def test_full_stages_list_existing_outputs_in_step_exit_criteria(hvault: Path):
    """Every file a stage's code check demands is named by the step's own exit criterion."""
    for step in hermes_icm.TIER_STEPS["full"]:
        text = (hvault / hermes.STEPS_DIR / f"{hermes_icm.STEP_FILES[step]}.md").read_text()
        for out in hermes_icm.STAGES[step].outputs:
            if out.startswith("@") or out.endswith("/"):
                continue
            name = out.replace("research/runs/{tag}/", "").replace("research/notes/", "")
            name = name.replace("{tag}", "<vault_tag>")
            assert name.split("/")[-1] in text, (step, out)


def test_full_draft_stage_loads_digest_not_raw_notes(hvault: Path):
    skill = (hvault / hermes.ENTRY_SKILL).read_text()
    ctx = hermes_icm.render_context(hermes_icm.STAGES["10"], "t", "full", "q", "/opt/hpr", skill, "")
    assert "temp/evidence-digest.md" in ctx
    assert "8-15 most relevant" not in ctx


def test_context_is_self_contained(hvault: Path):
    skill = (hvault / hermes.ENTRY_SKILL).read_text()
    for step in hermes_icm.TIER_STEPS["full"]:
        ctx = hermes_icm.render_context(
            hermes_icm.STAGES[step], "t-abc123", "full", "What is X?", "/opt/hpr", skill, ""
        )
        assert "> What is X?" in ctx
        assert "research/runs/t-abc123/" in ctx
        assert "{tag}" not in ctx
        assert hermes_icm.STEP_FILES[step] in ctx
        assert not hermes.LEFTOVER_RE.search(ctx), step


def test_stage1_carries_bootstrap_from_installed_skill(hvault: Path):
    skill = (hvault / hermes.ENTRY_SKILL).read_text()
    ctx = hermes_icm.render_context(hermes_icm.STAGES["1"], "t", "light", "q", "/opt/hpr", skill, "")
    assert "Classify modality" in ctx and "Write the scaffold" in ctx
    assert 'set `pipeline_tier` to `"light"`' in ctx


def test_fetch_stage_forces_small_parallel_batches(hvault: Path):
    skill = (hvault / hermes.ENTRY_SKILL).read_text()
    ctx = hermes_icm.render_context(hermes_icm.STAGES["2"], "t", "light", "q", "/opt/hpr", skill, "")
    assert "at most **3 URLs**" in ctx
    assert "`web_search`" in ctx


def test_slugify():
    assert hermes_icm.slugify(
        "What does current evidence say about how often residential carpet should be cleaned?"
    ) == "about-often-residential-carpet"
    assert hermes_icm.slugify("???") == "research"


@pytest.mark.parametrize(
    ("running", "avail", "expected"),
    [
        (0, 100, True),     # min_parallel always allowed
        (1, 100, False),    # no memory headroom
        (1, 1000, True),    # 700 reserve + 260 per agent fits
        (1, 900, False),
        (5, 99999, True),
        (6, 99999, False),  # max_parallel cap
        (2, None, True),    # no /proc/meminfo: fall back to 3
        (3, None, False),
    ],
)
def test_can_start(hvault: Path, running, avail, expected):
    cfg = hermes.load_config(hvault)
    assert hermes.can_start(running, cfg, avail) is expected


def test_mem_available_reads_proc():
    if not os.path.exists("/proc/meminfo"):
        pytest.skip("no /proc")
    assert (hermes.mem_available_mb() or 0) > 0


def test_supervisor_survives_caller_tree_kill(hvault: Path, monkeypatch, tmp_path: Path):
    """The detached supervisor must not be a descendant of the spawning process."""
    import subprocess
    import sys
    import time

    fake = tmp_path / "fake-hermes"
    fake.write_text(
        "#!" + sys.executable + "\nimport json,time\ntime.sleep(3)\n"
        "print(json.dumps({'type':'result','text':'ok','tokens':{'total':1}}))\n"
    )
    fake.chmod(0o755)
    (hvault / "m.md").write_text("x")
    monkeypatch.setattr(hermes, "hermes_bin", lambda: str(fake))
    batch_id = hermes.create_batch(hvault, [("hyperresearch-fetcher", Path("m.md"))], tag=None)
    # Launch from a throwaway process that we kill (tree and all) right away.
    code = (
        f"import sys; sys.path[:0] = {sys.path!r}\n"
        "from pathlib import Path\n"
        "from hyperresearch.core import hermes\n"
        f"hermes.start_batch_detached(Path({str(hvault)!r}), {batch_id!r})\n"
        "import time; time.sleep(60)\n"
    )
    monkeypatch.setenv("HPR_HERMES_BIN", str(fake))
    p = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    time.sleep(2)
    os.killpg(p.pid, 9)
    p.wait()
    summary = hermes.wait_batch(hvault, batch_id, 30)
    assert summary["status"] == "done"
    assert summary["counts"] == {"done": 1}


def test_vault_pin(hvault: Path, tmp_path_factory, monkeypatch):
    from hyperresearch.core.vault import VaultError

    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("HPR_VAULT_ROOT", str(hvault))
    assert Vault.discover().root == hvault.resolve()
    monkeypatch.setenv("HPR_VAULT_ROOT", str(elsewhere))
    with pytest.raises(VaultError):
        Vault.discover()


def test_init_refused_when_pinned(hvault: Path, monkeypatch, tmp_path_factory):
    from typer.testing import CliRunner

    from hyperresearch.cli import app

    d = tmp_path_factory.mktemp("stray")
    monkeypatch.chdir(d)
    monkeypatch.setenv("HPR_VAULT_ROOT", str(hvault))
    r = CliRunner().invoke(app, ["init", "."])
    assert r.exit_code == 2
    assert not (d / ".hyperresearch").exists()


def test_gate_readout_matches_run_finish_shape():
    ok = {"ok": True, "data": {"manifest": {}, "verify": {"passed": True, "checks": []}}}
    bad = {"ok": True, "data": {"verify": {"passed": False, "checks": [{"name": "x", "ok": False}]}}}
    assert hermes_icm._gate_passed(ok)
    assert not hermes_icm._gate_passed(bad)
    assert hermes_icm._gate_failures(bad) == [{"name": "x", "ok": False}]


def test_strip_markdown_keeps_prose_between_angle_brackets():
    from hyperresearch.core.note import strip_markdown

    t = "Clean when traffic is < 5 people and pets > 2 per home. <b>bold</b> <br/> <div class='x'>y</div>"
    out = strip_markdown(t)
    assert "< 5 people and pets > 2 per home" in out
    assert "<b>" not in out and "<br/>" not in out and "<div" not in out and "bold" in out


def test_trailing_slash_is_same_source(tmp_path: Path):
    import sqlite3

    from hyperresearch.core.fetcher import existing_live_note_for_url

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE sources (url TEXT, note_id TEXT)")
    conn.execute("INSERT INTO sources VALUES ('https://x.org/PMC1', 'n1'), ('https://y.org/a/', 'n2')")
    assert existing_live_note_for_url(conn, "https://x.org/PMC1/")["note_id"] == "n1"
    assert existing_live_note_for_url(conn, "https://y.org/a")["note_id"] == "n2"
    assert existing_live_note_for_url(conn, "https://x.org/PMC1?q=/") is None


def test_draft_stage_rules(hvault: Path):
    skill = (hvault / hermes.ENTRY_SKILL).read_text()
    hermes_icm._WORD_TARGET["t-x"] = (2000, 5000)
    ctx = hermes_icm.render_context(hermes_icm.STAGES["10"], "t-x", "full", "q", "/opt/hpr", skill, "")
    assert "2000-5000 words" in ctx and "**12-20**" in ctx and "evidence-digest.md" in ctx
    ctx11 = hermes_icm.render_context(hermes_icm.STAGES["11"], "t-x", "full", "q", "/opt/hpr", skill, "")
    assert "2000-5000 words" in ctx11
    ctx145 = hermes_icm.render_context(hermes_icm.STAGES["14.5"], "t-x", "full", "q", "/opt/hpr", skill, "")
    assert "already done, by code" in ctx145


def test_budget_config(hvault: Path):
    assert hermes.load_config(hvault).max_cost_usd == 0

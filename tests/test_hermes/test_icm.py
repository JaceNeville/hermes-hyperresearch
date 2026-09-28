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
    assert cfg.tier(cfg.stages["10"]).model == "claude-opus-5-5"  # the writing stage
    assert cfg.tier(cfg.stages["2"]).model == "claude-sonnet-5"


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
    assert "at most **5 URLs**" in ctx
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

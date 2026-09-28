"""Tests for the Hermes Agent port (fork addition)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hyperresearch.core import hermes
from hyperresearch.core.hooks import install_hermes
from hyperresearch.core.vault import Vault


@pytest.fixture
def hvault(tmp_path: Path) -> Path:
    Vault.init(tmp_path, platforms=())
    install_hermes(tmp_path, hpr_path="/opt/hpr")
    return tmp_path


def _all_installed(root: Path) -> list[Path]:
    return sorted((root / hermes.HERMES_DIR).rglob("*.md"))


def test_install_layout(hvault: Path):
    assert (hvault / hermes.ENTRY_SKILL).is_file()
    assert (hvault / hermes.CONFIG_FILE).is_file()
    steps = sorted(p.name for p in (hvault / hermes.STEPS_DIR).glob("*.md"))
    assert "hyperresearch-1-decompose.md" in steps
    assert "hyperresearch-16-readability-audit.md" in steps
    agents = sorted(p.stem for p in (hvault / hermes.AGENTS_DIR).glob("*.md"))
    assert "hyperresearch-fetcher" in agents
    assert "hyperresearch-browser-fetcher" not in agents  # no browser lane yet


def test_install_writes_nothing_outside_hyperresearch_dir(hvault: Path):
    for name in (".claude", ".codex", ".agents", "CLAUDE.md", "AGENTS.md"):
        assert not (hvault / name).exists(), name


def test_no_codex_vocabulary_survives(hvault: Path):
    """Upstream wording the translation table misses must fail here, not ship."""
    leftovers = []
    for path in _all_installed(hvault):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = hermes.LEFTOVER_RE.search(line)
            if m:
                leftovers.append(f"{path.relative_to(hvault)}:{n}: {m.group(0)}")
    assert not leftovers, "\n".join(leftovers[:20])


def test_hpr_path_resolved(hvault: Path):
    text = (hvault / hermes.ENTRY_SKILL).read_text(encoding="utf-8")
    assert "/opt/hpr" in text
    assert "{hpr_path}" not in text


def test_install_is_idempotent(hvault: Path):
    assert install_hermes(hvault, hpr_path="/opt/hpr") == []


def test_agent_frontmatter_roles_and_tool_locks(hvault: Path):
    meta, body = hermes.load_agent(hvault, "hyperresearch-patcher")
    assert meta["role"] == "patcher"
    assert meta["toolsets"] == ["file"]  # Read+Edit only: no shell, no web
    assert "Hermes runtime notes" in body

    meta, _ = hermes.load_agent(hvault, "hyperresearch-fetcher")
    assert meta["role"] == "fetcher"
    assert set(meta["toolsets"]) == {"file", "terminal", "web"}


def test_every_agent_role_has_a_tier(hvault: Path):
    cfg = hermes.load_config(hvault)
    for path in (hvault / hermes.AGENTS_DIR).glob("*.md"):
        meta, _ = hermes.load_agent(hvault, path.stem)
        assert meta["role"] in cfg.roles, path.stem


def test_default_tiering(hvault: Path):
    cfg = hermes.load_config(hvault)
    assert cfg.default_tier == "light"
    assert cfg.roles["fetcher"] == "bulk"
    assert cfg.roles["synthesizer"] == "synthesis"


def test_config_rejects_dangling_tier(hvault: Path):
    path = hvault / hermes.CONFIG_FILE
    path.write_text(path.read_text().replace('fetcher = "bulk"', 'fetcher = "nope"'))
    with pytest.raises(hermes.HermesError, match="unknown tier"):
        hermes.load_config(hvault)


def test_config_has_no_secrets():
    """The shipped config names models only; no key/token fields or values."""
    import re
    import tomllib

    data = tomllib.loads(hermes.DEFAULT_CONFIG_TOML)

    def keys(d, prefix=""):
        for k, v in d.items():
            yield prefix + k
            if isinstance(v, dict):
                yield from keys(v, prefix + k + ".")

    assert not [k for k in keys(data) if re.search(r"key|token|secret|password", k, re.I)]
    assert not re.search(r"(sk-|ghp_|github_pat_|xox[bp]-)[A-Za-z0-9]", hermes.DEFAULT_CONFIG_TOML)


def test_translate_examples():
    assert hermes.translate("use `apply_patch` on it") == "use `patch` on it"
    assert hermes.translate("custom agent") == "subagent"
    assert hermes.translate("Custom agent:") == "Subagent:"
    assert hermes.translate("see .codex/agents/hyperresearch-fetcher.toml") == (
        "see .hyperresearch/hermes/agents/hyperresearch-fetcher.md"
    )


def test_chat_cmd_shape(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(hermes, "hermes_bin", lambda: "hermes")
    cfg = hermes.load_config(tmp_path)
    cmd = hermes.build_chat_cmd(cfg.tier("bulk"), ["file", "terminal"], tmp_path / "q.md", cfg)
    assert cmd[:2] == ["hermes", "chat"]
    assert cmd[cmd.index("-m") + 1] == "claude-haiku-4-5"
    assert cmd[cmd.index("-t") + 1] == "file,terminal"
    assert "--ignore-rules" in cmd
    assert cmd[cmd.index("--source") + 1] == "tool"
    assert "--in" not in cmd
    cmd = hermes.build_chat_cmd(cfg.tier("bulk"), ["file"], tmp_path / "q.md", cfg, workdir=tmp_path)
    assert cmd[cmd.index("--in") + 1] == str(tmp_path.resolve())
    assert hermes.chat_env(tmp_path)["TERMINAL_CWD"] == str(tmp_path.resolve())


def test_spawn_batch_end_to_end_with_fake_hermes(hvault: Path, monkeypatch, tmp_path: Path):
    """Supervisor runs jobs with the parallel cap and records results + ledger."""
    fake = tmp_path / "fake-hermes"
    fake.write_text(
        "#!" + sys.executable + "\n"
        "import json, sys\n"
        "q = open(sys.argv[sys.argv.index('--query-file') + 1]).read()\n"
        "m = sys.argv[sys.argv.index('-m') + 1]\n"
        "print(json.dumps({'type': 'result', 'session_id': 's1', 'exit_code': 0,\n"
        "  'text': 'ok ' + m + ' ' + str(len(q)), 'tokens': {'input': 10, 'output': 5}}))\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr(hermes, "hermes_bin", lambda: str(fake))

    msg = hvault / "msg.md"
    msg.write_text("fetch these URLs")
    batch_id = hermes.create_batch(
        hvault,
        [("hyperresearch-fetcher", Path("msg.md")), ("hyperresearch-synthesizer", Path("msg.md"))],
        tag="t-abc123",
    )
    hermes.run_batch(hvault, batch_id)  # in-process for the test
    summary = hermes.summarize(hermes._load_batch(hvault, batch_id))
    assert summary["status"] == "done"
    assert summary["counts"] == {"done": 2}
    models = {j["agent"]: j["model"] for j in summary["jobs"]}
    assert models["hyperresearch-fetcher"] == "claude-haiku-4-5"
    assert models["hyperresearch-synthesizer"] == "claude-opus-5-5"

    ledger = hvault / "research/runs/t-abc123/temp/hermes-spawns.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert {r["model"] for r in rows} == {"claude-haiku-4-5", "claude-opus-5-5"}


def test_unknown_agent_rejected(hvault: Path):
    (hvault / "m.md").write_text("x")
    with pytest.raises(hermes.HermesError, match="unknown agent"):
        hermes.create_batch(hvault, [("hyperresearch-nope", Path("m.md"))], tag=None)


def test_orchestrator_prompt_tier_cap():
    p = hermes.orchestrator_prompt("What is X?", "light", None, "/opt/hpr")
    assert "Tier cap: light" in p
    assert "What is X?" in p
    assert "--budget" not in p
    assert "--budget 5.0" in hermes.orchestrator_prompt("q", "full", 5.0, "/opt/hpr")

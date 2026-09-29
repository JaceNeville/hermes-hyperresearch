"""Tests for publishing a run into an Obsidian vault (fork addition)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperresearch.core import hermes
from hyperresearch.core import hermes_publish as hp


def test_rewrite_links_only_touches_published_ids():
    t = "See [[a]], [[a|Alias]], [[a#Sec]], [[other]]."
    out = hp.rewrite_links(t, {"a": "research/sources/a"})
    assert out == (
        "See [[research/sources/a|a]], [[research/sources/a|Alias]], "
        "[[research/sources/a#Sec|a]], [[other]]."
    )


def test_frontmatter_roundtrip():
    meta, body = hp.split_frontmatter("---\ntitle: X\ntags: [a]\n---\n\nBody\n")
    assert meta == {"title": "X", "tags": ["a"]} and body == "Body\n"
    assert hp.split_frontmatter(hp.join_frontmatter(meta, body)) == (meta, body)
    assert hp.split_frontmatter("no fm") == ({}, "no fm")


def test_safe_title():
    assert hp.safe_title('How often? A/B: "test" #1') == "How often A B test 1"
    assert hp.safe_title("") == "Research report"


def test_cost_estimate():
    assert hp.estimate_cost({"input": 1e6, "output": 1e6, "cache_read": 1e6}, [3, 15, 0.3, 3.75]) == pytest.approx(18.3)
    assert hp.estimate_cost({"input": 5}, None) is None


def test_config_rejects_escaping_paths(tmp_path: Path):
    (tmp_path / ".hyperresearch").mkdir()
    (tmp_path / ".hyperresearch" / "hermes.toml").write_text('[hermes.publish]\nvault = "/v"\nlibrary = "../x"\n')
    with pytest.raises(hermes.HermesError):
        hp.load_publish_config(tmp_path)


def _fake_run(root: Path, tag: str) -> None:
    rd = root / "research" / "runs" / tag
    (rd / "temp").mkdir(parents=True)
    (root / "research" / "notes").mkdir(parents=True)
    (rd / "run.json").write_text(json.dumps({"profile": "light", "status": "done", "started_at": "2026-09-28T21:00:00",
                                             "verify": {"passed": True}}))
    (rd / "query.md").write_text("---\nvault_tag: x\n---\n\nHow often should carpet be cleaned?\n")
    (rd / "scaffold.md").write_text("scaffold [[src-a]]")
    (rd / "temp" / "hermes-spawns.jsonl").write_text(
        json.dumps({"agent": "stage-1-decompose", "role": "stage", "model": "m", "duration_ms": 60000,
                    "tokens": {"input": 1000000, "total": 1000000}}) + "\n")
    for nid, src in (("src-a", "https://a.example/1"), ("src-b", "https://b.example/2")):
        (root / "research" / "notes" / f"{nid}.md").write_text(f"---\nid: {nid}\nsource: {src}\n---\n\nBody of {nid} [[src-a]]\n")
    (root / "research" / "notes" / f"final_report_{tag}.md").write_text(
        "# How Often: Carpet?\n\nCleaning per [[src-a]] and [[src-b]].\n")


def test_publish_run_layout(tmp_path: Path, monkeypatch):
    work, obs, tag = tmp_path / "work", tmp_path / "obs", "carpet-abc123"
    work.mkdir()
    (obs / "Efforts" / "Proj").mkdir(parents=True)
    _fake_run(work, tag)
    # the library already holds src-b from an earlier run, and a different src-a
    (obs / "research" / "sources").mkdir(parents=True)
    (obs / "research" / "sources" / "src-b.md").write_text("---\nsource: https://b.example/2\n---\nold\n")
    (obs / "research" / "sources" / "src-a.md").write_text("---\nsource: https://other.example/\n---\nunrelated\n")
    notes = [{"id": "src-a", "path": "research/notes/src-a.md", "type": "note"},
             {"id": "src-b", "path": "research/notes/src-b.md", "type": "note"},
             {"id": f"final_report_{tag}", "path": f"research/notes/final_report_{tag}.md", "type": "note"}]
    monkeypatch.setattr(hp, "_run_json", lambda *a, **k: {"data": notes})
    cfg = hp.PublishConfig(vault=str(obs), prices={"m": [2, 0, 0, 0]})
    res = hp.publish_run(work, tag, "hpr", project="Efforts/Proj", cfg=cfg)

    assert res.report == "Efforts/Proj/Research/2026-09-28 How Often Carpet.md"
    assert res.sources_existing == ["research/sources/src-b.md"]
    assert res.sources_renamed == {"src-a": "research/sources/src-a--abc123.md"}
    # untouched: existing library notes
    assert "unrelated" in (obs / "research/sources/src-a.md").read_text()
    assert "old" in (obs / "research/sources/src-b.md").read_text()
    report = (obs / res.report).read_text()
    assert "[[research/sources/src-a--abc123|src-a]]" in report
    assert "[[research/sources/src-b|src-b]]" in report
    assert "type: research-report" in report
    rec = (obs / "research/runs" / tag / "RUN.md").read_text()
    assert "est_cost_usd: 2.0" in rec and "[[Efforts/Proj/Research/2026-09-28 How Often Carpet]]" in rec
    assert (obs / "research/runs" / tag / "scaffold.md").exists()
    assert not list((obs / "research/runs" / tag).glob("*.json"))


def test_publish_requires_vault(tmp_path: Path):
    with pytest.raises(hermes.HermesError, match="publishing is off"):
        hp.publish_run(tmp_path, "t", "hpr", cfg=hp.PublishConfig())


def test_publish_rejects_escaping_project(tmp_path: Path):
    with pytest.raises(hermes.HermesError):
        hp.publish_run(tmp_path, "t", "hpr", project="../etc", cfg=hp.PublishConfig(vault=str(tmp_path)))


def test_second_run_with_same_title_gets_its_own_report(tmp_path: Path, monkeypatch):
    obs = tmp_path / "obs"
    obs.mkdir()
    notes = [{"id": "src-a", "path": "research/notes/src-a.md", "type": "note"}]
    monkeypatch.setattr(hp, "_run_json", lambda *a, **k: {"data": notes})
    reports = []
    for tag in ("carpet-aaaaaa", "carpet-bbbbbb"):
        work = tmp_path / tag
        work.mkdir()
        _fake_run(work, tag)
        res = hp.publish_run(work, tag, "hpr", cfg=hp.PublishConfig(vault=str(obs)))
        reports.append(res.report)
    assert reports[0] != reports[1] and reports[1].endswith("(bbbbbb).md")
    # re-publishing the same run updates its own report in place
    res = hp.publish_run(tmp_path / "carpet-aaaaaa", "carpet-aaaaaa", "hpr", cfg=hp.PublishConfig(vault=str(obs)))
    assert res.report == reports[0]
    assert len(list((obs / "research/reports").iterdir())) == 2

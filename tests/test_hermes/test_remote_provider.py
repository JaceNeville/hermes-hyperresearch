"""Tests for the remote-browser fetch provider (fork addition)."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from hyperresearch.web import remote_provider as rp
from hyperresearch.web.base import WebResult


def _cfg(**kw):
    return rp.RemoteFetchConfig(host="ws", **kw)


def test_command_quotes_hostile_url():
    url = "https://example.com/a?b=c&d='x';rm -rf ~"
    with pytest.raises(rp.RemoteFetchError):  # whitespace rejected outright
        rp.build_remote_command(url, _cfg())
    url = "https://example.com/a?b=c&d='x';$(id)`id`"
    cmd = rp.build_remote_command(url, _cfg())
    assert shlex.quote(url) in cmd
    assert cmd.rstrip().endswith(shlex.quote(url) + " 2>/dev/null")


@pytest.mark.parametrize(
    "bad", ["file:///etc/passwd", "javascript:alert(1)", "https://a\nb", "https://a b", "ftp://x"]
)
def test_command_rejects_non_http(bad):
    with pytest.raises(rp.RemoteFetchError):
        rp.build_remote_command(bad, _cfg())


def test_command_uses_throwaway_profile_and_blocks_tailnet_names():
    cmd = rp.build_remote_command("https://example.com", _cfg())
    assert "mktemp -d" in cmd and 'rm -rf "$D"' in cmd
    assert "--user-data-dir=\"$D\"" in cmd
    assert "*.ts.net ~NOTFOUND" in cmd
    assert "HeadlessChrome" not in cmd


def test_config_from_hermes_toml(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HPR_REMOTE_FETCH_HOST", raising=False)
    (tmp_path / ".hyperresearch").mkdir()
    (tmp_path / ".hyperresearch" / "hermes.toml").write_text(
        '[hermes.remote_fetch]\nhost = "box"\nmode = "always"\nmax_parallel = 2\n'
    )
    c = rp.load_remote_config(tmp_path)
    assert (c.host, c.mode, c.max_parallel) == ("box", "always", 2)
    monkeypatch.setenv("HPR_REMOTE_FETCH_HOST", "other")
    assert rp.load_remote_config(tmp_path).host == "other"


@pytest.mark.parametrize("host", ["-oProxyCommand=x", "a b", "a;b", ""])
def test_config_rejects_unsafe_host(tmp_path: Path, monkeypatch, host):
    monkeypatch.setenv("HPR_REMOTE_FETCH_HOST", host)
    if not host:
        monkeypatch.delenv("HPR_REMOTE_FETCH_HOST")
    with pytest.raises(rp.RemoteFetchError):
        rp.load_remote_config(tmp_path)


class _FakeLocal:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, 0

    def fetch(self, url):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.result

    def _extract(self, html):
        return "T", "words " * 500


def _provider(local, mode="fallback"):
    p = rp.RemoteBrowserProvider.__new__(rp.RemoteBrowserProvider)
    from hyperresearch.core.config import FetchSettings

    p._settings, p._gates, p._cfg, p._local = FetchSettings(), None, _cfg(mode=mode), local
    p.remote_calls = []
    p._remote = lambda url: (p.remote_calls.append(url), WebResult(url=url, title="R", content="r"))[1]
    return p


def test_fallback_uses_local_when_good(monkeypatch):
    monkeypatch.setattr("hyperresearch.web.safe_http.check_url", lambda *a, **k: None)
    good = WebResult(url="https://x.org", title="t", content="word " * 400)
    p = _provider(_FakeLocal(result=good))
    assert p.fetch("https://x.org/page") is good
    assert p.remote_calls == []


def test_fallback_goes_remote_on_403_or_thin(monkeypatch):
    monkeypatch.setattr("hyperresearch.web.safe_http.check_url", lambda *a, **k: None)
    p = _provider(_FakeLocal(exc=RuntimeError("HTTP 403")))
    assert p.fetch("https://x.org/a").title == "R"
    thin = WebResult(url="https://x.org", title="Just a moment...", content="checking your browser")
    p = _provider(_FakeLocal(result=thin))
    assert p.fetch("https://x.org/b").title == "R"


def test_always_mode_skips_local(monkeypatch):
    monkeypatch.setattr("hyperresearch.web.safe_http.check_url", lambda *a, **k: None)
    local = _FakeLocal(result=WebResult(url="u", title="t", content="w " * 900))
    p = _provider(local, mode="always")
    p.fetch("https://x.org/c")
    assert local.calls == 0 and p.remote_calls == ["https://x.org/c"]


def test_remote_ssrf_checked_before_ssh(monkeypatch):
    from hyperresearch.web.safe_http import SafeHTTPError

    def deny(*a, **k):
        raise SafeHTTPError("private")

    monkeypatch.setattr("hyperresearch.web.safe_http.check_url", deny)
    p = _provider(_FakeLocal(exc=RuntimeError("HTTP 403")))
    with pytest.raises(SafeHTTPError):
        p.fetch("https://10.0.0.1/admin")
    assert p.remote_calls == []


@pytest.mark.parametrize("host,first", [("local", "sh"), ("my-box", "ssh")])
def test_local_host_runs_browser_without_ssh(monkeypatch, host, first):
    import subprocess

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=b"<html>" + b"x" * 400 + b"</html>", stderr=b"")

    monkeypatch.setattr(rp.subprocess, "run", fake_run)
    p = rp.RemoteBrowserProvider.__new__(rp.RemoteBrowserProvider)
    p._cfg = rp.RemoteFetchConfig(host=host)
    p._local = _FakeLocal()
    rp.RemoteBrowserProvider._remote(p, "https://x.org/d")
    assert seen["cmd"][0] == first
    assert ("my-box" in seen["cmd"]) == (host != "local")

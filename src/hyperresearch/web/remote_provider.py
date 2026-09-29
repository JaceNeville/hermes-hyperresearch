"""Remote-browser web provider (fork addition).

Runs a real headless Chromium on ANOTHER machine over SSH and returns the
rendered page. For agent hosts too small to run browsers locally.

    [hermes.remote_fetch]            # in .hyperresearch/hermes.toml
    host = "my-workstation"          # any `ssh` destination (alias, user@host)
    mode = "fallback"                # "fallback": local first, browser on failure
                                     # "always":   every HTML page via the browser
    chromium = "chromium"            # browser binary on the remote host
    timeout_s = 90
    max_parallel = 4                 # concurrent remote browsers (flock slots)

Or set HPR_REMOTE_FETCH_HOST in the environment (overrides the file).
`host = "local"` runs the same throwaway browser on this machine, no ssh
(for worker containers that ship Chromium; set `chromium` to its path).
Select it with `[web] provider = "remote"` in config.toml or `--provider remote`.

Safety:
* The URL is SSRF-checked on this host first (same gate as the builtin lane).
* Each fetch gets a throwaway browser profile, deleted afterwards: no
  cookies, no logins, no history. The remote user's own browser is never used.
* The remote command is built with shlex quoting; the URL is validated to
  plain http(s) with no whitespace or control characters before quoting.
* Tailnet (*.ts.net), *.local, and localhost names are unresolvable inside
  the remote browser, so a redirect can't pivot into the remote's network by
  name. A redirect to a private IP *literal* is not blocked; keep the remote
  host on a network where that doesn't matter, or leave mode = "fallback".

No host names, addresses, or credentials live in this module or the repo.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hyperresearch.web.base import WebResult

# Pages the local lane returns with fewer words than this get a browser retry.
THIN_WORDS = 150

_URL_OK = re.compile(r"^https?://[^\s\x00-\x1f\x7f]+$", re.IGNORECASE)

_RESOLVER_RULES = (
    "MAP *.ts.net ~NOTFOUND, MAP *.local ~NOTFOUND, MAP *.internal ~NOTFOUND, "
    "MAP localhost ~NOTFOUND, MAP *.localhost ~NOTFOUND"
)


class RemoteFetchError(RuntimeError):
    pass


@dataclass
class RemoteFetchConfig:
    host: str
    mode: str = "fallback"
    chromium: str = "chromium"
    timeout_s: int = 90
    max_parallel: int = 4
    ssh_options: tuple[str, ...] = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")


def _find_hermes_toml(start: Path) -> Path | None:
    for d in (start, *start.parents):
        p = d / ".hyperresearch" / "hermes.toml"
        if p.is_file():
            return p
    return None


def load_remote_config(start: Path | None = None) -> RemoteFetchConfig:
    section: dict = {}
    path = _find_hermes_toml((start or Path.cwd()).resolve())
    if path:
        section = tomllib.loads(path.read_text(encoding="utf-8")).get("hermes", {}).get("remote_fetch", {})
    host = os.environ.get("HPR_REMOTE_FETCH_HOST") or section.get("host")
    if not host:
        raise RemoteFetchError(
            "remote fetch has no host: set [hermes.remote_fetch] host in "
            ".hyperresearch/hermes.toml or HPR_REMOTE_FETCH_HOST"
        )
    if host.startswith("-") or not re.fullmatch(r"[\w.@-]+", host):
        raise RemoteFetchError("remote fetch host must be a plain ssh destination")
    mode = section.get("mode", "fallback")
    if mode not in ("fallback", "always"):
        raise RemoteFetchError("remote_fetch.mode must be 'fallback' or 'always'")
    chromium = section.get("chromium", "chromium")
    if not re.fullmatch(r"[\w./-]+", chromium):
        raise RemoteFetchError("remote_fetch.chromium must be a plain path or command name")
    return RemoteFetchConfig(
        host=host,
        mode=mode,
        chromium=chromium,
        timeout_s=int(section.get("timeout_s", 90)),
        max_parallel=max(1, int(section.get("max_parallel", 4))),
    )


def build_remote_command(url: str, cfg: RemoteFetchConfig) -> str:
    """The shell command run on the remote host. Pure; unit-tested."""
    if not _URL_OK.match(url):
        raise RemoteFetchError(f"refusing to hand a malformed URL to the remote browser: {url!r}")
    browser = [
        cfg.chromium,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-sync",
        "--mute-audio",
        '--user-data-dir="$D"',
        # Headless Chromium announces itself as "HeadlessChrome"; bot walls
        # (Cloudflare et al.) challenge that UA. Present the ordinary one.
        '--user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/$V.0.0.0 Safari/537.36"',
        "--disable-blink-features=AutomationControlled",
        f"--host-resolver-rules={shlex.quote(_RESOLVER_RULES)}",
        "--virtual-time-budget=15000",
        "--dump-dom",
        "--",
        shlex.quote(url),
    ]
    slots = " ".join(str(i) for i in range(cfg.max_parallel))
    # Take the first free flock slot (caps concurrent browsers on the remote),
    # use a throwaway profile, always clean it up.
    return (
        'set -e; L="${XDG_RUNTIME_DIR:-/tmp}/hpr-remote-fetch"; mkdir -p "$L"; '
        f"got=; for s in {slots}; do exec 9>\"$L/slot$s\"; if flock -n 9; then got=1; break; fi; "
        "exec 9>&-; done; "
        '[ -n "$got" ] || { exec 9>"$L/slot0"; flock 9; }; '
        f"V=$({shlex.quote(cfg.chromium)} --version 2>/dev/null | grep -oE '[0-9]+' | head -1); "
        '[ -n "$V" ] || V=140; '
        'D=$(mktemp -d); trap \'rm -rf "$D"\' EXIT; '
        f"timeout {int(cfg.timeout_s)} " + " ".join(browser) + " 2>/dev/null"
    )


class RemoteBrowserProvider:
    """Headless Chromium on a remote host, with an optional local-first lane."""

    name = "remote"

    def __init__(self, settings=None, gates=None, config: RemoteFetchConfig | None = None):
        from hyperresearch.core.config import FetchSettings
        from hyperresearch.web.builtin import BuiltinProvider

        self._settings = settings or FetchSettings()
        self._gates = gates
        self._cfg = config or load_remote_config()
        self._local = BuiltinProvider(settings=self._settings)

    def fetch(self, url: str) -> WebResult:
        from hyperresearch.web.pdf import is_pdf_url
        from hyperresearch.web.safe_http import check_url

        # PDFs never need a browser.
        if is_pdf_url(url) or self._cfg.mode == "fallback":
            try:
                result = self._local.fetch(url)
                if self._good_enough(url, result):
                    return result
            except Exception:
                if is_pdf_url(url):
                    raise
        check_url(url, allow_private_hosts=self._settings.allow_private_hosts)
        return self._remote(url)

    def _good_enough(self, url: str, result: WebResult) -> bool:
        if result.raw_bytes is not None:  # PDF lane
            return True
        if len((result.content or "").split()) < THIN_WORDS:
            return False
        return not (result.looks_like_login_wall(url, self._gates) or result.looks_like_junk(self._gates))

    def _remote(self, url: str) -> WebResult:
        remote = build_remote_command(url, self._cfg)
        # host = "local": the browser is on this machine (e.g. a worker
        # container that ships Chromium). Same command, no ssh hop.
        cmd = (["sh", "-c", remote] if self._cfg.host == "local"
               else ["ssh", *self._cfg.ssh_options, self._cfg.host, remote])
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=self._cfg.timeout_s + 30, stdin=subprocess.DEVNULL
            )
        except subprocess.TimeoutExpired as e:
            raise RemoteFetchError(f"remote browser timed out fetching {url}") from e
        html = proc.stdout.decode("utf-8", errors="replace")
        if proc.returncode != 0 or len(html) < 200:
            raise RemoteFetchError(
                f"remote browser failed for {url} (exit {proc.returncode}, {len(html)} bytes)"
            )
        title, content = self._local._extract(html)
        return WebResult(
            url=url,
            title=title,
            content=content,
            raw_html=html,
            fetched_at=datetime.now(UTC),
            metadata={"fetched_via": "remote-browser"},
        )

    def search(self, query: str, max_results: int = 5) -> list[WebResult]:
        raise NotImplementedError("remote provider does not search; use your agent's web search")

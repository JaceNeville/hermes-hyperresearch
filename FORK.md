# hermes-hyperresearch

A downstream fork of [jordan-gibbs/hyperresearch](https://github.com/jordan-gibbs/hyperresearch)
that runs the research pipeline natively on [Hermes Agent](https://github.com/NousResearch/hermes-agent)
instead of Claude Code or Codex.

Upstream's README is kept unchanged below this file's scope so upstream merges stay clean.
Everything fork-specific is documented here.

## What this fork changes

| Area | Upstream | This fork |
|---|---|---|
| Orchestrator | Claude Code / Codex skills + subagents | Hermes skill + `delegate_task` subagents |
| Models | Per-agent defaults in profiles | Tiered routing (bulk / analysis / synthesis) set in one config block |
| Default tier | `full` | `light`; `full` / `premier` must be requested explicitly |
| Fetching | Local crawl4ai, 8–12 parallel | Optional **remote fetch** over SSH for low-resource agent hosts |

## Quick start

```bash
pip install -e .                     # or: uv pip install -e .
hpr hermes install ~/research-vault  # renders prompts, writes .hyperresearch/hermes.toml
cd ~/research-vault
hpr hermes models                    # role -> tier -> model table
hpr hermes run "your research question"            # light tier (default)
hpr hermes run --tier full "your research question" # all 16 steps
```

`hpr hermes install` writes only under `.hyperresearch/`; it never creates
`CLAUDE.md`, `AGENTS.md`, `.claude/`, or `.codex/`.

## How it runs

- **Orchestrator:** one `hermes chat` session reads `.hyperresearch/hermes/SKILL.md`
  and walks the step files. `hpr hermes run` supervises it and resumes it (up to
  `--max-resumes`) if it stops before the ship gate passes.
- **Subagents:** `hpr hermes spawn --job AGENT=MESSAGE_FILE ...` launches each
  subagent as its own `hermes chat` process on the model tier for its role, with
  toolsets locked to what the role is allowed (the patcher gets no shell or web).
  Parallelism, timeouts, and per-run token ledgers are handled by the spawner.
- **Prompts:** rendered from upstream's templates (Codex branch) and passed through
  a small vocabulary table in `core/hermes.py`. A test fails if any Codex-specific
  wording survives, so upstream prompt changes can't silently ship broken.

## Model tiers

`.hyperresearch/hermes.toml` maps roles to three tiers:

| Tier | Default model | Roles |
|---|---|---|
| bulk | claude-haiku-4-5 | fetcher |
| analysis | claude-sonnet-5 | orchestrator, analysts, critics of the corpus, patcher, polish |
| synthesis | claude-opus-5-5 | draft orchestrator, synthesizer, report critics |

Swap models by editing the tier; re-run `hpr hermes install` to re-render.

## Status

Working: install, spawn, supervised end-to-end runs, remote fetch.

## Remote fetch (low-resource hosts)

Small agent hosts can't run headless browsers, and plain HTTP gets 403s from
bot-walled sites. The `remote` web provider runs Chromium on another machine
over SSH and returns the rendered page. Everything else stays local.

```toml
# config.toml
[web]
provider = "remote"

# .hyperresearch/hermes.toml  (keep host names out of anything you commit)
[hermes.remote_fetch]
host = "my-workstation"   # any ssh destination; or env HPR_REMOTE_FETCH_HOST
mode = "fallback"         # local HTTP first, browser only on 403/thin/junk
max_parallel = 4          # concurrent browsers on the remote (flock slots)
```

- Each fetch uses a throwaway browser profile, deleted afterwards. The remote
  user's own browser, cookies, and logins are never touched.
- URLs are SSRF-checked locally before they're sent; the remote command is
  shell-quoted; tailnet/`.local`/`localhost` names are unresolvable inside
  the remote browser.
- PDFs and clean pages never leave the local lane.

## Secret hygiene

- No credentials in the repo. Config refers to secrets by env var name only.
- `.githooks/pre-commit` runs gitleaks on staged changes
  (`git config core.hooksPath .githooks` once per clone).
- `.github/workflows/secret-scan.yml` scans full history on every push.
- GitHub secret scanning + push protection are enabled on the repo.
- `.gitleaksignore` lists reviewed upstream false positives only
  (a test fixture key and a public star-history chart token).
- The PyPI publish workflow is guarded to run only on the upstream repo.

## Tracking upstream

```bash
git remote add upstream https://github.com/jordan-gibbs/hyperresearch
git fetch upstream
git log --oneline main..upstream/main   # what's new upstream
```

## License

MIT, same as upstream. Copyright for upstream code remains with Jordan Gibbs.

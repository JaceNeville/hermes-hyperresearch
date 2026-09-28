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

## Status

Work in progress. See the commit log.

## Remote fetch (planned)

Hosts with little RAM can't run 8–12 headless browsers. Remote fetch runs the
fetch step on another machine over SSH and returns the page content; the vault,
indexing, and pipeline stay on the agent host. Off by default. Host details are
read from a local config file that is git-ignored, never from the repo.

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

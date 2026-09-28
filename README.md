# hermes-hyperresearch

Deep research on [Hermes Agent](https://github.com/NousResearch/hermes-agent). One question in, a cited, adversarially checked report out, with every source kept in a searchable library.

This is a downstream fork of Jordan Gibbs's [hyperresearch](https://github.com/jordan-gibbs/hyperresearch), which runs the same pipeline on Claude Code and OpenAI Codex. The research method is his. This fork adds a Hermes runtime built around three ideas:

- **Code runs the pipeline; models do the work.** A small Python loop sequences the steps. Each step runs as its own fresh Hermes session with only the files it needs.
- **Every role gets the cheapest model that can do it.** Fetching runs on a small model, analysis on a mid-tier one, and writing plus critique on the best one. The routing lives in one config file.
- **Output lands where people read.** Finished runs publish into an Obsidian vault: the report goes in the project folder that asked for it, sources go in one shared library, and each run gets a sealed run record.

Upstream's full documentation, covering the 16-step method, source ranking, scholarly search and the vault format, is kept at [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md). Everything there still applies.

---

## Quick start

```bash
git clone https://github.com/JaceNeville/hermes-hyperresearch && cd hermes-hyperresearch
uv pip install -e .                       # or: pip install -e .

hpr hermes install ~/research             # writes only under ~/research/.hyperresearch/
cd ~/research
hpr hermes models                         # which model each role runs on

hpr hermes icm "your research question"               # light tier: 5 stages, ~20 min
hpr hermes icm --tier full "your research question"   # all 17 stages, critics, 3 drafts
```

Requirements: Python 3.11+, the `hermes` CLI on `PATH`, and provider API keys in the environment Hermes already uses. This repo never stores keys.

---

## How a run works (ICM mode)

`hpr hermes icm` treats the pipeline as a set of stage folders. ICM stands for Interpretable Context Methodology: each stage has its own instructions, an explicit list of what to load and what to skip, and hands off to the next stage through files.

```
research/runs/<vault_tag>/
├── query.md, scaffold.md, run.json …    # the run's working files
└── stages/
    ├── RUN.md                           # stage table: status, model, minutes, tokens
    ├── 01_decompose/
    │   ├── CONTEXT.md                   # generated stage contract
    │   ├── session.log.jsonl            # the stage's Hermes session
    │   └── handoff.md                   # what the next stage needs to know
    ├── 02_width-sweep/ …
    └── 99_gate-fix/                     # only if the ship gate needed fixes
```

For each stage, code does the following:

1. Builds `CONTEXT.md` from the upstream step file: the verbatim query, a load list, a do-not-load list, the expected outputs, and the last few handoffs.
2. Starts a fresh `hermes chat` session on that stage's model.
3. Checks that the expected outputs exist. If any are missing, it resumes the session once.
4. Records tokens and time.

After the last stage, code runs the ship gate itself, sends failures to a short fix session, and marks the run folder sealed.

A new folder tree is generated for every run and never reused. Runs don't share working folders, so several can run at once without colliding.

**Why this matters.** In a single long session, every tool call re-sends everything that came before. On our benchmark, the one-session orchestrator averaged about 120k tokens per turn, and nearly all of that was re-reading. Fresh stages avoid that pile-up.

### Tiers

| Tier | Stages | What you get |
|---|---|---|
| `light` | 1, 2, 10, 15, 16 | Decompose, width sweep, single draft, polish, readability |
| `full` | 1–16 plus 14.5 | Adds contradiction graph, loci and depth investigation, corpus critic, evidence digest, three competing drafts and a synthesis, four critics, a gap fetch, a patcher, and a cite check |
| `auto` | decided by stage 1 | Stage 1 classifies the question and picks `light` or `full` |

### Measured (light tier, same question, list-price estimates)

| Version | Est. cost | Minutes | Sources | Cited | Words |
|---|---|---|---|---|---|
| Single session, orchestrator fetches inline | $8.64 | 24 | 18 | 16 | 2,625 |
| Single session, fetching delegated | $11.13 | 31 | 29 | 21 | 2,831 |
| **ICM stages** | **$6.40** | **20** | **31** | **29** | **3,205** |

All three passed the ship gate. Costs are estimated from token counts at list prices and are not a bill.

The older single-session runner is still available as `hpr hermes run`.

---

## Models

`.hyperresearch/hermes.toml` defines three tiers and maps every role and stage onto them:

| Tier | Default | Used for |
|---|---|---|
| `bulk` | claude-haiku-4-5 | fetchers, browser fetcher |
| `analysis` | claude-sonnet-5 | stage sessions, analysts, corpus critic, patcher, polish, cite check |
| `synthesis` | claude-opus-5-5 | draft orchestrators, synthesizer, the four report critics; light-tier drafting |

```toml
[hermes.tiers.analysis]
provider = "anthropic"
model = "claude-sonnet-5"

[hermes.stages]            # a stage's own session; "<step>@<tier>" overrides one tier
"10@light" = "synthesis"

[hermes.roles]             # subagents a stage spawns
fetcher = "bulk"
synthesizer = "synthesis"
```

Swap a model by editing its tier, then run `hpr hermes install` again. Any Hermes provider works.

## Subagents

Stages spawn subagents with `hpr hermes spawn`. Each subagent runs as its own `hermes chat` process with:

- **The model for its role.** Not inherited from the parent, which is why this fork doesn't use Hermes's built-in `delegate_task`.
- **Locked tools.** The patcher and polish auditor get file tools only: no shell, no web.
- **A pinned vault.** `HPR_VAULT_ROOT` is set, so a stray `cd` can't write into, or create, another vault.
- **A wall-clock cap and a token ledger** for every run.

Parallelism is memory-aware. Another subagent starts only while free memory stays above a reserve (`reserve_mb` + `per_agent_mb`), between `min_parallel` and `max_parallel`. Batches run under a fully detached supervisor, so a spawning session that times out doesn't take its subagents with it.

**Fetch delegation is enforced, not suggested.** Stage and orchestrator sessions run with `HPR_HERMES_ROLE=orchestrator`, and `hpr fetch` / `fetch-batch` refuse to run under that role. Fetching belongs to the cheap fetcher subagents, which get small batches of 3 URLs each so their sessions stay short.

---

## Publishing to Obsidian

```bash
hpr hermes icm --tier full --publish "question"                   # publish when the gate passes
hpr hermes icm --project "Efforts/Some Project" "question"        # report goes to that project
hpr hermes publish <vault_tag> [--project "Efforts/Some Project"] # publish a finished run later
```

```toml
[hermes.publish]
vault = "/path/to/Obsidian Vault"
library = "research/sources"     # one shared library, one note per source
reports = "research/reports"     # reports for runs without --project
runs = "research/runs"           # one sealed run record per run
reports_subdir = "Research"      # with --project P, reports land in P/Research/
write_prefix = []                # e.g. ["runuser", "-u", "vault", "--"] for a vault owned by another user

[hermes.prices]                  # optional, $/Mtok [input, output, cache_read, cache_write]
"claude-sonnet-5" = [3, 15, 0.30, 3.75]
```

What gets written:

- **Report:** `<project>/Research/<date> <title>.md`, with frontmatter for query, tier, source count, and a link to the run record.
- **Sources:** `<library>/<note-id>.md`. A source already in the library is reused, not overwritten. A different source that happens to share an id gets a suffixed filename.
- **Run record:** `<runs>/<vault_tag>/RUN.md`, with the query, a stage table (model, minutes, tokens, estimated cost), spend by model, and links to every source. The scaffold and any interim analyses sit alongside it.

Wikilinks are rewritten to path-qualified form (`[[research/sources/x|x]]`), so they resolve to the right note even in a large vault. Claims JSON, stage contracts and session logs stay in the working vault.

---

## Remote fetch for small hosts

Agent hosts are often small VPSes that can't run a headless browser, and plain HTTP gets 403s from bot-walled sites. The `remote` web provider runs Chromium on another machine over SSH and returns the rendered page.

```toml
# .hyperresearch/config.toml
[web]
provider = "remote"

# .hyperresearch/hermes.toml
[hermes.remote_fetch]
host = "my-workstation"   # any ssh destination, or env HPR_REMOTE_FETCH_HOST
mode = "fallback"         # local HTTP first; browser only on 403, thin, or junk pages
max_parallel = 4          # concurrent browsers on the remote
```

Safety guarantees:

- **Throwaway profiles.** Every fetch uses a fresh browser profile that is deleted afterwards. The remote user's own browser, cookies and logins are never touched.
- **Checked URLs.** Each URL is checked locally against private and internal addresses before it's sent, and only plain http(s) is allowed.
- **No local-network lookups.** The remote command is shell-quoted, and tailnet, `.local` and `localhost` names can't resolve inside the remote browser.
- **Local stays local.** PDFs and clean pages never leave the local fetch path.

---

## Commands added by this fork

| Command | What it does |
|---|---|
| `hpr hermes install [PATH]` | Render prompts for Hermes; write `.hyperresearch/hermes.toml` if absent |
| `hpr hermes models` | Show role → tier → model |
| `hpr hermes icm QUERY [--tier] [--publish] [--project]` | ICM run: staged, code-sequenced |
| `hpr hermes run QUERY [--tier]` | Single-session run (older runner) |
| `hpr hermes publish TAG [--project]` | Publish a finished run to Obsidian |
| `hpr hermes spawn --job AGENT=MSG_FILE …` | Launch subagents in parallel (used by stages) |
| `hpr hermes wait BATCH` | Wait on a spawn batch |
| `hpr hermes status` | Install and config health |

Everything else (`hpr fetch`, `note`, `search`, `scholar`, `run`, `sources`, `lint` and so on) is upstream's CLI, unchanged.

---

## Secret hygiene

This repo is public, so nothing secret goes in it.

- Config refers to secrets by environment-variable name only. Host names, vault paths and prices live in your local `.hyperresearch/hermes.toml`, which is git-ignored.
- `.githooks/pre-commit` runs gitleaks on staged changes. Enable it once per clone with `git config core.hooksPath .githooks`.
- `.github/workflows/secret-scan.yml` scans the full history on every push.
- `.gitleaksignore` holds reviewed upstream false positives only.
- The PyPI publish workflow runs only on the upstream repository.

## Tracking upstream

```bash
git remote add upstream https://github.com/jordan-gibbs/hyperresearch
git config merge.ours.driver true   # once per clone: keeps this README on merges
git fetch upstream
git log --oneline main..upstream/main
git merge upstream/main
```

The Hermes layer renders upstream's own prompts through a small translation table. The step files aren't copied or edited, so upstream improvements to the method flow straight through. A test fails if upstream wording changes in a way the table doesn't cover. Upstream's README lives in `docs/UPSTREAM_README.md`; refresh it from `upstream/main` when you merge.

## Status

- **Working:** Hermes install, subagent spawning, light-tier ICM runs, Obsidian publishing, remote fetch.
- **Being validated:** full-tier ICM runs.
- **Planned:** worker hosts, meaning subagents dispatched to other machines over SSH for more parallel capacity.

## License

MIT, same as upstream. Upstream code is copyright Jordan Gibbs; the Hermes runtime additions are copyright their contributors. See [LICENSE](LICENSE).

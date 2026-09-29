"""Intake: turn a rough research question into a well-formed research prompt.

Runs BEFORE stage 1 so the scope is right up front and the run itself can go
hands-off. The prompt shape comes from the user's own prompt-structure file
(IDENTITY / TASK / CONTEXT / CONSTRAINTS / OUTPUT FORMAT). One small session
maps the rough question onto that structure and either

* returns a finished research prompt (status "ready"), or
* returns at most three clarifying questions (status "needs_input") that
  must be answered before any research money is spent.

Answers come back via `hpr hermes intake --answers ...` (or the icm command's
`--answers`), and the intake runs again with them. Code, not the model,
decides what happens next: a run starts only from a "ready" intake.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hyperresearch.core import hermes

INTAKE_DIR = Path("research") / "intake"
MAX_QUESTIONS = 3

# Used when [hermes.intake] structure_file is unset or unreadable. Mirrors the
# five-part structure; kept short on purpose.
DEFAULT_STRUCTURE = """\
IDENTITY: who the answer is for and from what perspective.
TASK: the precise research question, with an action verb.
CONTEXT: background the researcher needs — what the reader already knows,
  why they are asking, the situation the answer will be used in.
CONSTRAINTS: scope limits — geography, time window, sources to prefer or
  avoid, what is out of scope.
OUTPUT FORMAT: the shape of the report — length, structure, what it must
  end with (recommendation, ranked list, decision table ...).
"""


@dataclass
class IntakeResult:
    status: str                      # "ready" | "needs_input" | "error"
    intake_id: str
    prompt: str = ""                 # finished research prompt when ready
    questions: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    tier: str = ""                   # suggested tier: light | full
    path: str = ""                   # intake folder, relative to the vault
    tokens: dict = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _structure_text(vault_root: Path, cfg: hermes.HermesConfig) -> tuple[str, str]:
    path = (cfg.intake_structure_file or "").strip()
    if path:
        p = Path(path) if Path(path).is_absolute() else vault_root / path
        try:
            return p.read_text(encoding="utf-8"), str(p)
        except OSError:
            pass
    return DEFAULT_STRUCTURE, "built-in default"


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:5]
    return "-".join(words) or "question"


def build_prompt(rough: str, structure: str, answers: str, structure_src: str) -> str:
    answers_block = (
        f"\n## The requester's answers to your earlier questions\n\n{answers.strip()}\n"
        if answers.strip() else ""
    )
    return f"""# Research intake

You prepare research questions for an automated deep-research pipeline that
runs unattended for one to three hours and costs real money. Your one job:
make sure the scope is right BEFORE it starts. You do not research anything.
Do not browse, fetch, or search. Do not create files.

## The requester's prompt structure (from {structure_src})

Build the research prompt with THIS structure — every part it defines, in its
order, with its labels.

{structure.strip()}

## The rough question

{rough.strip()}
{answers_block}
## How to decide

Map the rough question (and any answers) onto every part of the structure.
For each part, either fill it from what the requester said or, where a
sensible default exists, fill it and record the default as an assumption.

Ask a clarifying question ONLY when a part can't be filled confidently AND a
wrong guess would send the research in a different direction (wrong
audience, wrong geography or market, wrong decision being made, ambiguous
subject). Never ask about things with a safe default (length, citation
style, tone). Ask at most {MAX_QUESTIONS} questions, each answerable in one line.
If answers were given above, use them and do not ask again about the same
point.

Also pick a tier: "light" (a focused practical question, ~20 minutes) or
"full" (contested, multi-sided, or high-stakes, 2+ hours).

## Output

Reply with ONLY this JSON object, no prose before or after:

{{"status": "ready" | "needs_input",
  "prompt": "<the finished research prompt, using the structure's labels; empty if needs_input>",
  "questions": ["<only if needs_input; max {MAX_QUESTIONS}>"],
  "assumptions": ["<defaults you chose that the requester may want to know>"],
  "tier": "light" | "full"}}
"""


def _parse(text: str) -> dict:
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("intake reply had no JSON object")
    return json.loads(m.group(0))


def run_intake(vault_root: Path, rough: str, answers: str = "", intake_id: str | None = None) -> IntakeResult:
    """Run one intake pass. Re-run with the same intake_id and `answers` after questions."""
    cfg = hermes.load_config(vault_root)
    structure, src = _structure_text(vault_root, cfg)
    if intake_id:
        idir = vault_root / INTAKE_DIR / intake_id
        if not idir.is_dir():
            raise hermes.HermesError(f"unknown intake: {intake_id}")
        rough = rough or (idir / "rough.md").read_text(encoding="utf-8")
    else:
        intake_id = time.strftime("%Y%m%d-%H%M%S") + "-" + _slug(rough)
        idir = vault_root / INTAKE_DIR / intake_id
        idir.mkdir(parents=True)
        (idir / "rough.md").write_text(rough.strip() + "\n", encoding="utf-8")
    n = len(list(idir.glob("pass-*.md"))) + 1
    if answers.strip():
        (idir / f"answers-{n}.md").write_text(answers.strip() + "\n", encoding="utf-8")
    all_answers = "\n\n".join(p.read_text(encoding="utf-8") for p in sorted(idir.glob("answers-*.md")))
    prompt_file = idir / f"pass-{n}.md"
    prompt_file.write_text(build_prompt(rough, structure, all_answers, src), encoding="utf-8")

    tier = cfg.tier(cfg.intake_tier)
    cmd = hermes.build_chat_cmd(tier, ["file"], prompt_file, cfg, max_turns=4, workdir=vault_root)
    log = idir / f"pass-{n}.log.jsonl"
    with open(log, "w", encoding="utf-8") as f:
        subprocess.call(cmd, cwd=vault_root, env=hermes.chat_env(vault_root, hermes.ORCHESTRATOR_ROLE),
                        stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=600)
    res = hermes._parse_result(log)
    rel = str(idir.relative_to(vault_root))
    try:
        data = _parse(res.get("text", ""))
    except (ValueError, json.JSONDecodeError) as e:
        return IntakeResult("error", intake_id, path=rel, tokens=res.get("tokens") or {}, error=str(e))

    questions = [q for q in (data.get("questions") or []) if str(q).strip()][:MAX_QUESTIONS]
    prompt = str(data.get("prompt") or "").strip()
    status = "ready" if data.get("status") == "ready" and prompt and not questions else "needs_input"
    if status == "needs_input" and not questions:
        questions = ["The intake couldn't build a complete prompt. What should this research decide or answer?"]
    result = IntakeResult(
        status=status, intake_id=intake_id, prompt=prompt if status == "ready" else "",
        questions=questions if status == "needs_input" else [],
        assumptions=[str(a) for a in (data.get("assumptions") or [])],
        tier=str(data.get("tier")) if data.get("tier") in ("light", "full") else "",
        path=rel, tokens=res.get("tokens") or {},
    )
    (idir / "intake.json").write_text(json.dumps(result.as_dict(), indent=1) + "\n", encoding="utf-8")
    if status == "ready":
        (idir / "prompt.md").write_text(prompt + "\n", encoding="utf-8")
    return result


def load_ready(vault_root: Path, intake_id: str) -> IntakeResult:
    p = vault_root / INTAKE_DIR / intake_id / "intake.json"
    if not p.exists():
        raise hermes.HermesError(f"no intake result for {intake_id}")
    return IntakeResult(**json.loads(p.read_text(encoding="utf-8")))

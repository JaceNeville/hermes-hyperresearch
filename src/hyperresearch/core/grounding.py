"""Deterministic grounding check: do cited sentences say only what their sources say?

Runs over EVERY cited sentence of a report (no sampling, no model), against
the full text of the cited source notes. It catches the specific ways reports
drift from their sources:

* numbers   — a number in a cited sentence that appears in none of its cited
              notes ("30-year" when the source says 28; "n=1,485" when the
              source never gives it)
* quotes    — quoted text (4+ words) that is not verbatim in a cited note
* attribution — "EPA says / according to CDC ..." where the named body does
              not appear in any cited note

It can't judge meaning ("cheapest" vs "most important"); the cite-checker
agents do that. What it guarantees is cheap, complete, and final: it runs
after the last stage that edits prose, so nothing slips in afterwards.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hyperresearch.core.patterns import WIKI_LINK_RE

# The report's own source list ("## Sources", "## H. Source List", "## References").
_SOURCES_HEAD = re.compile(
    r"^##\s+(?:[A-Z0-9]{1,3}[.)]\s+)?(?:Sources?|References?)(?:\s+(?:List|with\s+dates|cited))?\s*:?\s*$", re.M | re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[*A-Z0-9])")
# a trailing "x" is a multiplier ("2-26x annually"), not part of a word
_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\d.]\d|[\d]|[a-wyz_])")
_QUOTE = re.compile(r"[\"“]([^\"”]{1,400})[\"”]")   # pair every quote; length filter later
# "<Name> says|found|..." and "according to <Name>"
_ATTR_VERBS = r"(?:says|said|states|stated|found|finds|reports|reported|recommends|recommended|requires|required|advises|advised|concludes|concluded|calls|called|notes|noted|warns|warned|estimates|estimated)"
_ATTR_BEFORE = re.compile(r"\b([A-Z][A-Za-z&.\-]*(?: [A-Z][A-Za-z&.\-]*){0,3})(?:'s)? (?:\w+ )?" + _ATTR_VERBS + r"\b")
_ATTR_AFTER = re.compile(r"\baccording to (?:the )?([A-Z][A-Za-z&.\-]*(?: [A-Z][A-Za-z&.\-]*){0,3})")
_ATTR_SKIP = {"The", "This", "That", "It", "One", "A", "An", "Each", "Every", "Most", "Some", "No", "None",
              "Both", "Neither", "Either", "Its", "Their", "His", "Her", "Our", "Your", "Another", "Such",
              "These", "Those", "Which", "What", "Who", "Also", "But", "And", "Or", "If", "When", "So"}


@dataclass
class Finding:
    kind: str                 # number | quote | attribution | dangling
    sentence: str
    missing: list[str]
    cited: list[str]
    line: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroundingResult:
    cited_sentences: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict:
        return {"ok": self.ok, "cited_sentences": self.cited_sentences,
                "by_kind": self.by_kind(), "findings": [f.as_dict() for f in self.findings]}

    def by_kind(self) -> dict:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.kind] = out.get(f.kind, 0) + 1
        return out


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).lower()


def _strip_markup(sentence: str) -> str:
    s = WIKI_LINK_RE.sub(" ", sentence)
    s = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", " ", s)       # [3] / [3, 7] markers
    s = re.sub(r"`[^`]*`", " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    return s


def _numbers(sentence: str) -> list[str]:
    s = _strip_markup(sentence)
    s = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", s)           # list bullets / "1." numbering
    s = re.sub(r"\b[A-Z]{2,}[- ]?\d+\b", " ", s)           # standard names: S100, CRI 205, IICRC S500
    out = []
    for m in _NUM.finditer(s):
        n = m.group(1).replace(",", "")
        if n in {"0", "1", "2"}:                           # too common to be meaningful
            continue
        out.append(n)
    return list(dict.fromkeys(out))


_WORDNUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
            "twenty": "20", "thirty": "30", "fifty": "50", "hundred": "100"}


def _source_numbers(body: str) -> set[str]:
    b = _fold(body)
    nums = {m.group(1).replace(",", "") for m in _NUM.finditer(b)}
    for w, d in _WORDNUM.items():
        if re.search(rf"\b{w}\b", b):
            nums.add(d)
    # "0.5" == ".5", "12.0" == "12"
    more = set()
    for n in nums:
        if "." in n:
            more.add(n.rstrip("0").rstrip("."))
    return nums | more


_STOP = set(["about", "above", "after", "again", "against", "also", "among", "because", "been", "before", "being", "below", "between", "both", "could", "does", "doing", "during", "each", "from", "further", "have", "having", "here", "into", "itself", "just", "more", "most", "much", "must", "only", "other", "over", "same", "should", "since", "some", "such", "than", "that", "their", "them", "then", "there", "these", "they", "this", "those", "through", "under", "until", "very", "were", "what", "when", "where", "which", "while", "will", "with", "would", "your", "years", "year", "months", "month", "times", "percent"])


_UNITS = ("year", "month", "week", "day", "hour", "minute", "percent", "%", "times", "trial", "home",
          "patient", "participant", "carpet", "people", "sq", "square", "foot", "feet", "gallon", "degree")


def _near_context(n: str, sentence: str, src_text: str) -> bool:
    """Unit-aware check. When the sentence gives the number a unit ("30 years",
    "25%", "n=1,485 patients"), the sources must carry the same number with the
    same unit; a stray "30" elsewhere in a long source doesn't count. Numbers
    without a unit only need to appear."""
    plain = _fold(_strip_markup(sentence)).replace(",", "")
    src = _digits(src_text.replace(",", ""))
    num = re.escape(n)
    units = set()
    for m in re.finditer(rf"(?<![\w.]){num}(?![\w])\s*(?:-|to|or|\u2013)?\s*(?:\d[\d.]*)?\s*-?\s*([a-z%]+)", plain):
        u = m.group(1)
        for known in _UNITS:
            if u.startswith(known):
                units.add(known)
    if not units:
        return True
    for u in units:
        alts = [re.escape(u), *{"%": ["percent"], "percent": ["%"], "times": ["x\\b", "\u00d7"]}.get(u, [])]
        pat = rf"(?<![\w.]){num}(?![\d])[^.;\n]{{0,24}}?(?:{'|'.join(alts)})"
        if re.search(pat, src):
            continue
        if u in _COUNT_UNITS and re.search(rf"\bn\s*=\s*{num}(?![\w])", src):
            continue
        if u == "home" and re.search(rf"(?<![\w.]){num}(?![\w])[^.;\n]{{0,40}}?(?:household|house|residence|dwelling)", src):
            continue
        return False
    return True


_COUNT_UNITS = {"trial", "home", "patient", "participant", "people", "carpet"}


def _digits(text: str) -> str:
    """'six to nine months' -> '6 to 9 months' so unit checks see the numbers."""
    return re.sub(r"\b(" + "|".join(_WORDNUM) + r")\b", lambda m: _WORDNUM[m.group(1)], text)


def _cited_ids(sentence: str) -> list[str]:
    return [m.group(1).strip() for m in WIKI_LINK_RE.finditer(sentence)]


def _sentences(report: str) -> list[tuple[int, str]]:
    body = _SOURCES_HEAD.split(report, maxsplit=1)[0]
    out = []
    in_code = False
    for ln, line in enumerate(body.splitlines(), 1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("|"):
            cells = [c for c in line.strip().strip("|").split("|")]
            # A table row is one claim unit: numbers anywhere in the row need its cites.
            if "[[" in line and not re.match(r"^\s*\|?\s*:?-{3}", line):
                out.append((ln, " | ".join(c.strip() for c in cells)))
            continue
        for s in _SENT_SPLIT.split(line):
            if "[[" in s:
                out.append((ln, s.strip()))
    return out


def load_note_bodies(vault_root: Path, ids: set[str]) -> dict[str, str | None]:
    notes_dir = vault_root / "research" / "notes"
    out: dict[str, str | None] = {}
    for i in ids:
        p = notes_dir / f"{i}.md"
        if not p.exists():
            hits = list((vault_root / "research").rglob(f"{i}.md"))
            p = hits[0] if hits else p
        if not p.exists():
            out[i] = None
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if text.startswith("---"):
            parts = text.split("---", 2)
            fm, body = (parts[1], parts[2]) if len(parts) == 3 else ("", text)
            keep = re.findall(r"^(?:title|source_domain|source|summary):\s*((?:.*)(?:\n[ \t]+.*)*)", fm, re.M)
            text = " ".join(keep) + "\n" + body
        out[i] = text
    return out


def check_report(report_text: str, bodies: dict[str, str | None]) -> GroundingResult:
    sents = _sentences(report_text)
    folded: dict[str, str] = {k: _fold(v) for k, v in bodies.items() if v}
    nums_cache: dict[str, set[str]] = {k: _source_numbers(v) for k, v in bodies.items() if v}
    res = GroundingResult(cited_sentences=len(sents))
    for ln, s in sents:
        ids = _cited_ids(s)
        dangling = [i for i in ids if not bodies.get(i)]
        if dangling:
            res.findings.append(Finding("dangling", s, dangling, ids, ln))
        live = [i for i in ids if bodies.get(i)]
        if not live:
            continue
        src_nums = set().union(*(nums_cache[i] for i in live))
        src_text = " ".join(folded[i] for i in live)
        missing = [n for n in _numbers(s) if n not in src_nums or not _near_context(n, s, src_text)]
        if missing:
            res.findings.append(Finding("number", s, missing, live, ln))
        bad_quotes = []
        for q in _QUOTE.findall(_strip_markup(s)):
            if len(q) < 12:
                continue
            fq = _fold(q).strip(" .,;:")
            if len(fq.split()) >= 4 and fq not in src_text:
                # tolerate an ellipsis inside the quote: every part must be verbatim
                parts = [p.strip(" .,;:") for p in re.split(r"\.\.\.|…|\[\.\.\.\]", fq) if p.strip(" .,;:")]
                if not parts or not all(p in src_text for p in parts):
                    bad_quotes.append(q)
        if bad_quotes:
            res.findings.append(Finding("quote", s, bad_quotes, live, ln))
        bad_attr = []
        plain = _strip_markup(s)
        for m in [*_ATTR_BEFORE.finditer(plain), *_ATTR_AFTER.finditer(plain)]:
            words = m.group(1).strip(" .").split()
            while words and words[0] in _ATTR_SKIP:
                words = words[1:]
            name = " ".join(words)
            if len(name) < 3:
                continue
            # A lone title-case word opening a sentence or table cell is usually
            # an ordinary word ("Documentation required"), not a named source.
            pre = plain[: m.start(1) + (len(m.group(1)) - len(m.group(1).lstrip()))]
            at_start = (not pre.strip(" |*-") or pre.rstrip().endswith("|")) and m.group(1).split()[0] == name.split()[0]
            if at_start and " " not in name and not name.isupper():
                continue
            key = _fold(name)
            # accept a match on the full name or on any 3+ letter acronym/word of it
            tokens = [t for t in re.findall(r"[a-z0-9&]+", key) if len(t) >= 3]
            if key not in src_text and not any(re.search(rf"\b{re.escape(t)}\b", src_text) for t in tokens):
                bad_attr.append(name)
        if bad_attr:
            res.findings.append(Finding("attribution", s, list(dict.fromkeys(bad_attr)), live, ln))
    return res


_NUM_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\](?!\()")
_SRC_LINE = re.compile(r"^\s*(?:[-*]\s*)?\[(\d+)\]\s*(.*)$")
_URL = re.compile(r"https?://[^\s)>\]]+")


def _norm_url(u: str) -> str:
    u = u.strip().rstrip(".,;)").lower()
    u = re.sub(r"^https?://(www\.)?", "", u)
    u = u.split("#", 1)[0]
    return u.rstrip("/")


def _url_index(vault_root: Path) -> dict[str, str]:
    """source URL (normalized) -> note id, from every research note's frontmatter."""
    out: dict[str, str] = {}
    notes = vault_root / "research" / "notes"
    if not notes.is_dir():
        return out
    for p in notes.glob("*.md"):
        try:
            head = p.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        if not head.startswith("---"):
            continue
        m = re.search(r"^source:\s*['\"]?(\S+?)['\"]?\s*$", head.split("---", 2)[1] if head.count("---") >= 2 else "", re.M)
        if m:
            out.setdefault(_norm_url(m.group(1)), p.stem)
    return out


def resolve_numbered_citations(text: str, url_to_id: dict[str, str]) -> tuple[str, list[str]]:
    """Rewrite inline-style `[N]` / `[7, 12]` markers into `[[note-id]]` links so the
    grounding check covers numbered-citation reports too. N is looked up in the
    report's own `## Sources` list (`[N] Title. URL`), then URL -> note id.
    A number with no Sources entry, no URL, or no matching note becomes a link to
    `unresolved-citation-N`, which the check reports as dangling (fail closed).
    Returns (rewritten text, list of unresolved numbers)."""
    parts = _SOURCES_HEAD.split(text, maxsplit=1)
    if len(parts) < 2:
        body, sources = text, ""
    else:
        body, sources = parts[0], parts[1]
    num_to_id: dict[str, str] = {}
    for line in sources.splitlines():
        m = _SRC_LINE.match(line)
        if not m:
            continue
        u = _URL.search(m.group(2))
        if u and _norm_url(u.group(0)) in url_to_id:
            num_to_id[m.group(1)] = url_to_id[_norm_url(u.group(0))]
    unresolved: list[str] = []

    def sub(m: re.Match) -> str:
        links = []
        for n in re.split(r"\s*,\s*", m.group(1)):
            nid = num_to_id.get(n)
            if not nid:
                unresolved.append(n)
                nid = f"unresolved-citation-{n}"
            links.append(f"[[{nid}]]")
        return " ".join(links)

    new_body = _NUM_CITE.sub(sub, body)
    head = _SOURCES_HEAD.search(text)
    if head is None:
        return new_body, list(dict.fromkeys(unresolved))
    return new_body + text[head.start():], list(dict.fromkeys(unresolved))


def check_file(vault_root: Path, report_path: Path) -> GroundingResult:
    text = report_path.read_text(encoding="utf-8-sig")
    if _NUM_CITE.search(_SOURCES_HEAD.split(text, maxsplit=1)[0]):
        text, _ = resolve_numbered_citations(text, _url_index(vault_root))
    ids = set()
    for _, s in _sentences(text):
        ids.update(_cited_ids(s))
    res = check_report(text, load_note_bodies(vault_root, ids))
    if res.cited_sentences == 0:
        # A report the gate can't read is not "grounded". Fail closed so it
        # can't publish as verified (a [N]-cited report once passed with 0 checked).
        res.findings.append(Finding(
            "vacuous", "(whole report)",
            ["no cited sentences found: citations missing or in a format the check can't read"],
            [], 0))
    return res


def write_findings(path: Path, result: GroundingResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

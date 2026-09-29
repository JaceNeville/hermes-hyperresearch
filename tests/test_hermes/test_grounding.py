"""Deterministic grounding check: numbers, quotes, attributions vs cited notes."""

from __future__ import annotations

from hyperresearch.core import grounding as g

SRC = {
    "stainmaster": "Premium lines extend texture retention from 25 to 28 years with a qualifying pad. "
                   "Clean every 18 months.",
    "cochrane": "Symptom scores SMD -0.06 (95% CI -0.16 to 0.05). 55 trials were included.",
    "rct": "Randomized controlled trial (N=247: 125 intervention, 122 sham) over 40 weeks.",
    "cri204": "Heavy: 2-6x annually. Severe: 2-26x annually.",
    "localpro": "For homes with pets, cleaning every six to nine months is a good baseline.",
    "aafa": "AAFA (aafa.org). Vacuum carpets weekly. The EPA says to dry within 24-48 hours.",
    "study": "Allergen fell 94 percent on average in 20 homes.",
}


def run(sentence: str) -> g.GroundingResult:
    return g.check_report(sentence + "\n", SRC)


def test_invented_number_is_caught():
    r = run("Premium lines extend coverage from 25 to 30 years [[stainmaster]].")
    assert [f.kind for f in r.findings] == ["number"] and r.findings[0].missing == ["30"]


def test_number_present_but_different_unit_is_caught():
    # "18" is in the source as months; "18 years" is not supported
    r = run("Coverage runs 18 years [[stainmaster]].")
    assert r.findings and r.findings[0].missing == ["18"]


def test_supported_numbers_pass():
    assert run("Coverage goes from 25 to 28 years [[stainmaster]].").ok
    assert run("A trial of 247 patients over 40 weeks found nothing [[rct]].").ok
    assert run("CRI allows up to 26 deep cleans a year in severe traffic [[cri204]].").ok
    assert run("Pet homes: every 6-9 months [[localpro]].").ok
    assert run("It removed 94% of allergen [[study]].").ok


def test_sample_size_not_in_source_is_caught():
    r = run("Symptoms did not improve (n=1,485) [[cochrane]].")
    assert r.findings and "1485" in r.findings[0].missing


def test_fabricated_quote_is_caught_and_real_quote_passes():
    r = run('AAFA says to vacuum "once or twice a week" [[aafa]].')
    assert any(f.kind == "quote" for f in r.findings)
    assert run('AAFA says "vacuum carpets weekly" [[aafa]].').ok


def test_attribution_must_appear_in_source():
    assert run("The EPA says to dry carpets fast [[aafa]].").ok
    r = run("The CDC recommends drying carpets fast [[aafa]].")
    assert [f.kind for f in r.findings] == ["attribution"] and r.findings[0].missing == ["CDC"]


def test_dangling_citation():
    r = run("Something is true [[no-such-note]].")
    assert r.findings[0].kind == "dangling"


def test_uncited_sentences_and_sources_section_ignored():
    text = "An uncited claim of 99 years.\n\n## Sources\n\n- 12345 [[stainmaster]] 77 years\n"
    r = g.check_report(text, SRC)
    assert r.ok and r.cited_sentences == 0


def test_table_row_checked_as_one_unit():
    ok = "| Stainmaster | 18 months | [[stainmaster]] |\n"
    bad = "| Stainmaster | 24 months | [[stainmaster]] |\n"
    assert g.check_report(ok, SRC).ok
    assert not g.check_report(bad, SRC).ok


def test_standard_names_and_small_numbers_not_flagged():
    assert run("IICRC S100 and CRI 205 both say clean every 18 months [[stainmaster]].").ok

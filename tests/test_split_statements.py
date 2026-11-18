"""Tests for the statement splitting step (corpus/split_statements.py)."""

from corpus.split_statements import keep_paragraph, split_document, split_paragraphs

# Example statements of more than 10 words, so they pass the length limits.
CLAIM = "Net zero targets will cost every household in Britain thousands of pounds a year."
REPLY = "Independent analysts say the cost of reaching net zero has been overstated."


def test_joins_paragraph_broken_across_a_page():
    text = "The government wants to build new wind farms\nacross the North Sea by 2030.\nA new paragraph starts here."
    assert split_paragraphs(text) == [
        "The government wants to build new wind farms across the North Sea by 2030.",
        "A new paragraph starts here.",
    ]


def test_tidies_whitespace_and_skips_blank_lines():
    text = "\n\n   Too   many    spaces\there.  \n\n   \n"
    assert split_paragraphs(text) == ["Too many spaces here."]


def test_drops_headings():
    assert not keep_paragraph("WHY NET ZERO WILL COST BRITISH FAMILIES MORE THAN THE GOVERNMENT ADMITS")
    assert keep_paragraph(CLAIM)


def test_keeps_only_10_to_300_words():
    assert not keep_paragraph("Far too short to carry a claim.")
    assert not keep_paragraph(" ".join(["word"] * 301))
    assert keep_paragraph(" ".join(["Word"] * 10))
    assert keep_paragraph(" ".join(["word"] * 300))


def test_split_document_numbers_statements_and_fills_fields():
    doc = {
        "doc_id": "GB7",
        "source": "GB News",
        "date": "2024-03-22",  # GB News records use "date" before renaming
        "title": "Example article",
        "text": f"THE COST OF NET ZERO\n{CLAIM}\nToo short.\n{REPLY}",
    }
    statements = split_document(doc)

    # The heading and the short line are dropped, and numbering skips them.
    assert [s["statement_id"] for s in statements] == ["GB7_0", "GB7_1"]
    assert [s["statement"] for s in statements] == [CLAIM, REPLY]
    assert all(s["date_published"] == "2024-03-22" for s in statements)
    assert all(s["source"] == "GB News" for s in statements)

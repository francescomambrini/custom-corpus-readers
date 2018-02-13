import pytest
from corpusreader import CommentedConllCorpusReader, WebAnnoTsvCorpusReader


@pytest.fixture
def cite():
    return WebAnnoTsvCorpusReader("samples", "citation_sample.tsv",
                                      ["ignore", "ignore", "words", "ne"])

def test_commented():

    corp = CommentedConllCorpusReader("samples", "citation_sample.tsv",
                                      ["ignore", "ignore", "words", "ne"])
    assert len(corp.sents()) == 236


def test_tab_cite():
    corp = WebAnnoTsvCorpusReader("samples", "citation_sample.tsv",
                                  ["ignore", "ignore", "words", "ne"])
    tab = corp.iob_words(columns=["words", "ne"])

    assert tab[0] == ("Chapter", "_")


def test_iob_conversion_cite():
    corp = WebAnnoTsvCorpusReader("samples", "citation_sample.tsv",
                                  ["ignore", "ignore", "words", "ne"])
    tab = corp.iob_words(columns=["words", "ne"], convert2iob=["ne"])

    assert tab[0] == ("Chapter", "O")


def test_iob_sent_cite(cite):
    c = cite.iob_sents(columns=["words", "ne"], convert2iob=["ne"])
    assert c[1][115][1] == "B-DATE"
import pytest

from lextrace.chunking import fixed_windows, make_chunks
from lextrace.config import ChunkingConfig

from .conftest import DOCS

LONG = ("ARTICLE I DEFINITIONS\n1.1 \"Content\" means the audiovisual works listed in Schedule A. " * 3
        + "\n\n2. LICENSE GRANT\n2.1 Licensor grants Licensee a non-exclusive license to stream the Content. "
        + "word " * 300 + "\n\n3. TERM\nThe term ends on December 31, 2030.\n")


@pytest.mark.parametrize("strategy", ["fixed", "paragraph", "clause"])
@pytest.mark.parametrize("size,overlap", [(120, 0), (200, 50), (500, 0)])
def test_chunks_index_the_original_text(strategy, size, overlap):
    cfg = ChunkingConfig(strategy=strategy, size=size, overlap=overlap, min_size=20)
    for doc_id, text in list(DOCS.items()) + [("long.txt", LONG)]:
        chunks = make_chunks(doc_id, text, cfg)
        assert chunks
        for c in chunks:
            assert text[c.start:c.end] == c.text
            assert c.end - c.start <= size
            assert c.text == c.text.strip()


@pytest.mark.parametrize("strategy", ["fixed", "paragraph", "clause"])
def test_chunks_cover_all_non_whitespace(strategy):
    cfg = ChunkingConfig(strategy=strategy, size=150, overlap=0, min_size=20)
    covered = set()
    for c in make_chunks("long.txt", LONG, cfg):
        covered.update(range(c.start, c.end))
    assert all(i in covered for i, ch in enumerate(LONG) if not ch.isspace())


def test_fixed_windows_overlap_and_word_boundaries():
    text = " ".join(f"w{i:03d}" for i in range(200))
    spans = fixed_windows(text, 0, len(text), 100, 30)
    assert all(text[s] != " " and text[e - 1] != " " for s, e in spans)
    assert all(b[0] < a[1] for a, b in zip(spans, spans[1:]))  # consecutive windows overlap
    assert spans[-1][1] == len(text)


def test_paragraph_chunks_do_not_split_short_paragraphs():
    cfg = ChunkingConfig(strategy="paragraph", size=120, min_size=10)
    text = DOCS["fx/gamma_delta_distribution.txt"]
    texts = [c.text for c in make_chunks("d", text, cfg)]
    assert any(t.endswith("Delta may not sublicense the Films to any third-party platform.") for t in texts)


def test_clause_chunker_splits_at_headings():
    cfg = ChunkingConfig(strategy="clause", size=800, min_size=10)
    chunks = make_chunks("long.txt", LONG, cfg)
    assert any(c.text.startswith("3. TERM") for c in chunks)


def test_clause_chunks_differ_from_paragraph_chunks():
    text = "1. Term. Two years.\n\n2. Law. New York.\n\nThe parties agree.\n\n3. Notices. In writing."
    para = [c.text for c in make_chunks("d", text, ChunkingConfig(strategy="paragraph", size=800, min_size=5))]
    clause = [c.text for c in make_chunks("d", text, ChunkingConfig(strategy="clause", size=800, min_size=5))]
    assert len(para) == 1  # paragraphs are packed up to the size limit
    assert clause == ["1. Term. Two years.", "2. Law. New York.\n\nThe parties agree.", "3. Notices. In writing."]


def test_invalid_chunk_config():
    from lextrace.config import ConfigError

    with pytest.raises(ConfigError):
        ChunkingConfig(strategy="fixed", size=100, overlap=100).validate()
    with pytest.raises(ConfigError):
        ChunkingConfig(strategy="sentences").validate()

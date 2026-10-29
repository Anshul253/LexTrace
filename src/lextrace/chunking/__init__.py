"""Chunkers turn a contract into passages. Every chunk is a [start, end) range over the ORIGINAL text and
`chunk.text == document[start:end]` holds by construction (asserted in `make_chunks`).

Strategies
  fixed     - windows of `size` characters with `overlap`, end snapped back to whitespace so words are not cut.
  paragraph - split on blank lines; merge short paragraphs up to `size`; long paragraphs fall back to fixed windows.
  clause    - split before section headings ("12.", "12.3", "Section 4", "ARTICLE V", "(a)" at line start), then
              the same merge / split rules. Aims to align chunks with the units lawyers cite."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from ..config import ChunkingConfig

Span = Tuple[int, int]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str


_BLANK_LINE = re.compile(r"\n[ \t\f\v]*\n\s*")
_HEADING = re.compile(
    r"\n[ \t]*(?=(?:\d{1,2}(?:\.\d{1,2})*\.?[ \t]+\S|section[ \t]+\d|article[ \t]+[ivxlc\d]+|\([a-z0-9]{1,3}\)[ \t]+\S"
    r"|[A-Z][A-Z ,&/\-]{3,60}\n))",
    re.I,
)


def _trim(text: str, s: int, e: int) -> Span:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def fixed_windows(text: str, start: int, end: int, size: int, overlap: int) -> List[Span]:
    """Windows over text[start:end]; each ends at the last whitespace inside the window when one exists."""
    spans: List[Span] = []
    pos = start
    while pos < end:
        stop = min(pos + size, end)
        if stop < end:
            cut = text.rfind(" ", pos + size // 2, stop)
            cut = max(cut, text.rfind("\n", pos + size // 2, stop))
            if cut > pos:
                stop = cut
        s, e = _trim(text, pos, stop)
        if e > s:
            spans.append((s, e))
        if stop >= end:
            break
        nxt = stop - overlap if overlap else stop
        pos = nxt if nxt > pos else stop
    return spans


def _segments(text: str, boundary: re.Pattern) -> List[Span]:
    cuts = [0] + [m.start() for m in boundary.finditer(text)] + [len(text)]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        s, e = _trim(text, a, b)
        if e > s:
            out.append((s, e))
    return out


_HEADING_START = re.compile(r"(?:\d{1,2}(?:\.\d{1,2})*\.?[ \t]+\S|section[ \t]+\d|article[ \t]+[ivxlc\d]+)", re.I)


def _pack(text: str, segments: List[Span], cfg: ChunkingConfig, keep_headings_apart: bool = False) -> List[Span]:
    """Greedily merge consecutive segments up to cfg.size; split oversize segments into windows.

    keep_headings_apart (clause strategy): a segment that starts with a section number/heading always starts a new
    chunk, unless the current chunk is still shorter than min_size (e.g. a bare heading line)."""
    out: List[Span] = []
    cur: List[int] = []
    for s, e in segments:
        if e - s > cfg.size:
            if cur:
                out.append((cur[0], cur[1]))
                cur = []
            out.extend(fixed_windows(text, s, e, cfg.size, cfg.overlap))
            continue
        new_clause = keep_headings_apart and bool(_HEADING_START.match(text, s))
        short = bool(cur) and cur[1] - cur[0] < cfg.min_size
        if cur and e - cur[0] <= cfg.size and (short or not new_clause):
            cur[1] = e
        else:
            if cur:
                out.append((cur[0], cur[1]))
            cur = [s, e]
    if cur:
        out.append((cur[0], cur[1]))
    return out


def chunk_spans(text: str, cfg: ChunkingConfig) -> List[Span]:
    if cfg.strategy == "fixed":
        return fixed_windows(text, 0, len(text), cfg.size, cfg.overlap)
    if cfg.strategy == "paragraph":
        return _pack(text, _segments(text, _BLANK_LINE), cfg)
    if cfg.strategy == "clause":
        segs: List[Span] = []
        for s, e in _segments(text, _BLANK_LINE):
            sub = _segments(text[s:e], _HEADING)
            segs.extend((s + a, s + b) for a, b in sub)
        return _pack(text, segs, cfg, keep_headings_apart=True)
    raise ValueError(f"unknown chunking strategy {cfg.strategy!r}")


def make_chunks(doc_id: str, text: str, cfg: ChunkingConfig) -> List[Chunk]:
    chunks = []
    for i, (s, e) in enumerate(chunk_spans(text, cfg)):
        piece = text[s:e]
        assert piece == text[s:e] and e > s, "chunk offsets must index the original text"
        chunks.append(Chunk(f"{doc_id}#{i}", doc_id, s, e, piece))
    return chunks

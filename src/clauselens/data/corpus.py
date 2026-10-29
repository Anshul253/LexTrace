"""Contract text store. Files are read exactly as the benchmark assumes: Python default text mode, UTF-8
(universal newlines, so "\\r\\n" becomes "\\n"). The text is never modified afterwards; every offset in the system
refers to these strings."""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..paths import CORPUS_DIR


class MissingDocumentError(FileNotFoundError):
    """A referenced contract is not on disk."""


def read_contract(path: Path) -> str:
    with open(path, encoding="utf-8") as f:  # default text mode, as the benchmark offsets assume
        return f.read()


def locate(corpus_dir: Path, doc_id: str) -> Optional[Path]:
    """Find a document, tolerating NFC/NFD differences in non-ASCII file names."""
    for cand in dict.fromkeys([doc_id, unicodedata.normalize("NFC", doc_id), unicodedata.normalize("NFD", doc_id)]):
        p = corpus_dir / cand
        if p.exists():
            return p
    return None


class Corpus:
    """In-memory, read-only map doc_id -> full text."""

    def __init__(self, texts: Dict[str, str]):
        if not texts:
            raise ValueError("corpus is empty")
        self._texts = dict(sorted(texts.items()))

    @classmethod
    def load(cls, doc_ids: Iterable[str], corpus_dir: Path = CORPUS_DIR) -> "Corpus":
        texts, missing = {}, []
        for d in doc_ids:
            p = locate(corpus_dir, d)
            if p is None:
                missing.append(d)
            else:
                texts[d] = read_contract(p)
        if missing:
            raise MissingDocumentError(
                f"{len(missing)} referenced contract(s) not found under {corpus_dir}, e.g. {missing[0]!r}. "
                "Run `python -m clauselens prepare` to download them.")
        return cls(texts)

    def text(self, doc_id: str) -> str:
        return self._texts[doc_id]

    def doc_ids(self) -> List[str]:
        return list(self._texts)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._texts

    def __len__(self) -> int:
        return len(self._texts)

    def total_chars(self) -> int:
        return sum(len(t) for t in self._texts.values())

    @staticmethod
    def title_of(doc_id: str) -> str:
        """Human-readable name from the file name (CUAD names encode company, date, exhibit and agreement type)."""
        stem = Path(doc_id).stem
        return stem.replace("_", " ")

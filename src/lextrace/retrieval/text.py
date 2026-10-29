"""Tokenisation shared by the lexical retrievers and the router, plus optional concept-lexicon query expansion."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ..config import TokenizerConfig
from ..paths import LEXICON_FILE

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
# keep words that carry legal meaning even though generic stopword lists drop them
_KEEP = {"not", "no", "nor", "without", "any", "all", "against", "within", "after", "before", "until", "during"}
STOPWORDS = frozenset(w for w in ENGLISH_STOP_WORDS if w not in _KEEP)


def light_stem(word: str) -> str:
    """Conservative suffix stripping: plurals, -ing, -ed. 'licenses'->'licens', 'licensed'->'licens'."""
    if len(word) <= 4:
        return word
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", "e"), ("s", "")):
        if word.endswith(suf) and not (suf == "s" and word.endswith("ss")) and len(word) - len(suf) >= 3:
            word = word[: len(word) - len(suf)] + rep
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str, cfg: TokenizerConfig) -> List[str]:
    toks = _WORD.findall(text.lower() if cfg.lowercase else text)
    toks = [t.split("'")[0] for t in toks]
    if cfg.stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    if cfg.stemming == "light":
        toks = [light_stem(t) for t in toks]
    return toks


@lru_cache(maxsize=4)
def _load_lexicon(path: str) -> Dict[str, List[str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out = {}
    for entry in data.get("concepts", []):
        for trigger in entry["triggers"]:
            out[trigger.lower()] = list(entry["expand"])
    return out


def expand_query(question: str, path: Path = LEXICON_FILE) -> str:
    """Append the expansion terms of every lexicon trigger phrase found in the question.

    The lexicon is written from general contract-law vocabulary (CUAD's public clause categories), not from
    benchmark answers; it maps how a question names a concept to how contracts usually word it."""
    lex = _load_lexicon(str(path))
    low = question.lower()
    extra: List[str] = []
    for trigger, terms in lex.items():
        if trigger in low:
            extra.extend(terms)
    return question if not extra else question + " " + " ".join(dict.fromkeys(extra))

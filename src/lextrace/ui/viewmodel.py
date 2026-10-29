"""Framework-free view models for the web app, so request handling is unit-testable without a browser.

Nothing here invents data: evaluation and experiment views read the JSON files written by the CLI and say so
when a file is missing."""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import ConfigError, list_configs, load_config
from ..data.corpus import Corpus
from ..paths import EXPERIMENTS_DIR, RESULTS_DIR, TEST_LEDGER
from ..pipeline import Engine, InvalidQueryError

CONTEXT_CHARS = 280


@dataclass
class EvidenceCard:
    rank: int
    doc_id: str
    contract: str
    start: int
    end: int
    score: float
    method: str
    text: str
    before: str
    after: str

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class SearchView:
    status: str  # answered | abstained | invalid | error
    message: str = ""
    cards: List[EvidenceCard] = field(default_factory=list)
    weak_cards: List[EvidenceCard] = field(default_factory=list)  # below-threshold candidates, shown only on request
    meta: Dict = field(default_factory=dict)


def contract_label(doc_id: str) -> str:
    return Corpus.title_of(doc_id).replace("cuad/", "")


def _card(p, corpus: Corpus) -> EvidenceCard:
    text = corpus.text(p.doc_id)
    assert text[p.start:p.end] == p.text, "displayed passage must equal document[start:end]"
    return EvidenceCard(p.rank, p.doc_id, contract_label(p.doc_id), p.start, p.end, p.score, p.method, p.text,
                        text[max(0, p.start - CONTEXT_CHARS):p.start], text[p.end:p.end + CONTEXT_CHARS])


def run_search(engine: Engine, question: str, k: int) -> SearchView:
    try:
        resp = engine.search(question, k=k)
    except InvalidQueryError as e:
        return SearchView("invalid", f"Please revise the question: {e}.")
    except Exception as e:  # noqa: BLE001 - never show a traceback in the UI
        return SearchView("error", f"The search failed ({type(e).__name__}: {e}). No answer was produced.")
    meta = {"config": resp.config_name, "method": resp.method, "latency_ms": resp.latency_ms,
            "signals": resp.signals, "contract_named": resp.contract_named,
            "routed": [(contract_label(d), s) for d, s in resp.routed_contracts],
            "gates": [(g.signal, g.threshold) for g in engine.cfg.abstention.gates],
            "failed_gate": resp.failed_gate, "n_candidates": len(resp.candidates)}
    if resp.abstained:
        return SearchView("abstained", resp.abstention_reason or "No passage met the evidence threshold.",
                          weak_cards=[_card(p, engine.corpus) for p in resp.candidates[:k]], meta=meta)
    return SearchView("answered", f"{len(resp.passages)} passage(s) met the evidence threshold.",
                      cards=[_card(p, engine.corpus) for p in resp.passages], meta=meta)


def highlight_html(card: EvidenceCard) -> str:
    """Passage in its surrounding context; only the retrieved span is highlighted. All text is escaped."""
    esc = lambda s: html.escape(s).replace("\n", "<br>")  # noqa: E731
    return (f'<span class="cl-ctx">{"&hellip;" if card.before else ""}{esc(card.before)}</span>'
            f'<mark class="cl-hit">{esc(card.text)}</mark>'
            f'<span class="cl-ctx">{esc(card.after)}{"&hellip;" if card.after else ""}</span>')


# Clause questions a rights / legal-affairs reviewer typically asks (CUAD clause categories, phrased as questions).
CLAUSE_TEMPLATES: Dict[str, str] = {
    "Sublicensing": "Does this agreement allow a party to sublicense its rights to a third party?",
    "Exclusivity": "Is there an exclusivity provision, and when does the exclusivity period end?",
    "IP ownership": "Who owns the intellectual property, including in derivative works?",
    "License grant": "What licenses are granted under this contract?",
    "Term / expiry": "What is the expiration date of this contract?",
    "Renewal": "What is the renewal term after the initial term expires?",
    "Termination": "Can a party terminate this contract for convenience?",
    "Assignment": "Is there an anti-assignment clause in this contract?",
    "Change of control": "Does a change of control of a party trigger any rights?",
    "Non-compete": "Is there a non-compete clause restricting a party from competing?",
    "Revenue share": "Does this contract include any revenue or profit-sharing arrangement?",
    "Liability cap": "Is there a cap on liability under this contract?",
    "Audit rights": "Does a party have the right to audit the other party's books or records?",
    "Governing law": "What is the governing law for this contract?",
}

_AGREEMENT_TYPE = re.compile(r"(?:EX-[\d.]+[_ -]*)?([A-Za-z][A-Za-z ,&\-]*?(?:AGREEMENT|Agreement)[A-Za-z ]*)\d*$")


def agreement_type(doc_id: str) -> str:
    """Agreement type from the CUAD file name, e.g. '..._EX-10.11_Co-Branding Agreement1.txt' -> 'Co-Branding Agreement'."""
    stem = Path(doc_id).stem
    tail = re.split(r"EX-[\d.]+", stem)[-1].strip(" _-")
    m = _AGREEMENT_TYPE.search(tail) or _AGREEMENT_TYPE.search(stem)
    text = (m.group(1) if m else tail).replace("_", " ").strip(" -")
    text = re.sub(r"\s+", " ", text).split(" - ")[-1]
    return text.title() if text.isupper() else text


def compose_question(contract_title: Optional[str], question: str) -> str:
    """Benchmark-style question naming the contract, which lets the router restrict the search to it."""
    question = question.strip()
    return f"Consider the {contract_title}; {question}" if contract_title else question


def contract_catalog(corpus: Corpus, titles: Dict[str, str], n_chunks: Optional[Dict[str, int]] = None) -> List[Dict]:
    """One row per indexed contract: display title, agreement type, size. Titles come from contract names only."""
    rows = []
    for d in corpus.doc_ids():
        rows.append({"doc_id": d, "title": titles.get(d) or contract_label(d), "type": agreement_type(d),
                     "characters": len(corpus.text(d)), "chunks": (n_chunks or {}).get(d)})
    return sorted(rows, key=lambda r: r["title"].lower())


def contract_titles(queries) -> Dict[str, str]:
    """Contract name per document, taken from the 'Consider the <name>;' clause of TRAIN questions (a contract name
    carries no answer information; test questions are never read here)."""
    from ..retrieval.router import parse_query

    out: Dict[str, str] = {}
    for q in queries:
        if q.split != "train":
            continue
        t = parse_query(q.query).title
        if t:
            for d in q.doc_ids:
                out.setdefault(d, t)
    return out


def find_in_document(text: str, needle: str, limit: int = 50) -> List[Tuple[int, int]]:
    """Case-insensitive occurrences of a phrase (for the library's in-document search)."""
    if not needle.strip():
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle.strip()), text, re.I)][:limit]


def document_html(text: str, spans: List[Tuple[int, int]], cls: str = "cl-hit") -> str:
    """Escape a document and wrap the given [start, end) spans in <mark>. Spans must not overlap."""
    out, pos = [], 0
    for s, e in sorted(spans):
        if s < pos:
            continue
        out.append(html.escape(text[pos:s]))
        out.append(f'<mark class="{cls}" id="cl-{s}">{html.escape(text[s:e])}</mark>')
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def headline_metrics(split: str = "train") -> Dict[str, Optional[Dict]]:
    """Saved summaries for the baseline and final configs (None when not yet produced)."""
    res = load_results(split)
    pick = lambda tag: res[tag]["summary"] if tag in res else None  # noqa: E731
    return {"baseline": pick("baseline"), "final": pick("final")}


def usable_configs() -> List[str]:
    """Configs whose thresholds are calibrated (an uncalibrated config cannot decide abstention)."""
    out = []
    for name in list_configs():
        try:
            if not load_config(name).needs_calibration():
                out.append(name)
        except ConfigError:
            continue
    return out


def load_results(split: str) -> Dict[str, Dict]:
    """All saved evaluation summaries for a split, keyed by tag (file stem without the split)."""
    out = {}
    if not RESULTS_DIR.exists():
        return out
    for p in sorted(RESULTS_DIR.glob(f"*_{split}.json")):
        try:
            out[p.stem[: -len(split) - 1]] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def load_experiment_log(path: Path = EXPERIMENTS_DIR / "experiment_log.jsonl") -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_run_count(path: Path = TEST_LEDGER) -> Optional[int]:
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

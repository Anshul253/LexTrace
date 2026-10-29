"""Framework-free view models for the web app, so request handling is unit-testable without a browser.

Nothing here invents data: evaluation and experiment views read the JSON files written by the CLI and say so
when a file is missing."""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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

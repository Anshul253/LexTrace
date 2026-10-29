"""Small synthetic fixtures (clearly NOT benchmark data) so unit tests need no download or embedding model."""
from __future__ import annotations

import pytest

from clauselens.config import config_from_dict
from clauselens.data.benchmark import BenchmarkQuery, GoldSpan
from clauselens.data.corpus import Corpus

DOC_A = (
    "SPONSORSHIP AGREEMENT\n\nThis Sponsorship Agreement is made between Alpha Media Ltd. and Beta Drinks Inc.\n\n"
    "1. Term. The term of this Agreement shall be two (2) years from the Effective Date.\n\n"
    "2. Governing Law. This Agreement shall be governed by the laws of the State of New York.\n\n"
    "3. Assignment. Neither party may assign this Agreement without the prior written consent of the other party.\n\n"
    "4. Exclusivity. Beta Drinks shall be the exclusive beverage sponsor of the show during the Term.\n"
)
DOC_B = (
    "DISTRIBUTION AGREEMENT\n\nThis Distribution Agreement is entered into by Gamma Films LLC and Delta Streaming Corp.\n\n"
    "1. Territory. Delta may distribute the Films in India and Sri Lanka.\n\n"
    "2. Royalties. Delta shall pay Gamma a royalty of ten percent of net receipts.\n\n"
    "3. Governing Law. This Agreement is governed by the laws of England and Wales.\n\n"
    "4. Sublicensing. Delta may not sublicense the Films to any third-party platform.\n"
)
DOC_C = (
    "SERVICE AGREEMENT\n\nThis Service Agreement is between Epsilon Studios and Zeta Consulting.\n\n"
    "1. Services. Zeta shall provide post-production consulting services.\n\n"
    "2. Fees. Epsilon shall pay Zeta a monthly fee of USD 5,000.\n\n"
    "3. Confidentiality. Each party shall keep the other's confidential information secret.\n"
)
DOCS = {"fx/alpha_beta_sponsorship.txt": DOC_A, "fx/gamma_delta_distribution.txt": DOC_B,
        "fx/epsilon_zeta_service.txt": DOC_C}


def gold_for(doc_id: str, phrase: str) -> GoldSpan:
    text = DOCS[doc_id]
    s = text.index(phrase)
    return GoldSpan(doc_id, s, s + len(phrase), phrase)


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(dict(DOCS))


def small_config(**over) -> "object":
    base = {
        "name": "fixture",
        "chunking": {"strategy": "paragraph", "size": 200, "overlap": 0, "min_size": 10},
        "routing": {"mode": "none"},
        "retrieval": {"method": "bm25", "candidates": 10},
        "abstention": {"enabled": True, "gates": [{"signal": "top_score", "threshold": 0.0}]},
        "evaluation": {"ks": [1, 2, 4], "primary_k": 1},
    }
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return config_from_dict(base)


@pytest.fixture
def cfg():
    return small_config()


def fixture_queries():
    """Toy answerable queries, half train half test by construction (split label set explicitly)."""
    A, B = "fx/alpha_beta_sponsorship.txt", "fx/gamma_delta_distribution.txt"
    return [
        BenchmarkQuery("fx-001", "Consider the Sponsorship Agreement between Alpha Media Ltd. and Beta Drinks Inc.; "
                       "What is the governing law?", "train",
                       (gold_for(A, "This Agreement shall be governed by the laws of the State of New York."),)),
        BenchmarkQuery("fx-002", "Consider the Distribution Agreement between Gamma Films LLC and Delta Streaming Corp.; "
                       "Can Delta sublicense the films to a third-party platform?", "train",
                       (gold_for(B, "Delta may not sublicense the Films to any third-party platform."),)),
        BenchmarkQuery("fx-003", "Consider the Sponsorship Agreement between Alpha Media Ltd. and Beta Drinks Inc.; "
                       "Is there an anti-assignment clause?", "train",
                       (gold_for(A, "Neither party may assign this Agreement without the prior written consent of the other party."),)),
    ]

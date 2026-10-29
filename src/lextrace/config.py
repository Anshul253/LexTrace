"""Pipeline configuration: typed, validated, YAML-backed.

A configuration is a tree of dataclasses. YAML files may `extends: <other>.yaml` and override any subset of keys.
Unknown keys are an error (a typo must never silently fall back to a default). The improvement loop creates
candidates with `apply_patch`, which deep-merges a dict of overrides and re-validates."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .paths import CONFIG_DIR


class ConfigError(ValueError):
    """Invalid or unreadable configuration."""


@dataclass
class ChunkingConfig:
    strategy: str = "fixed"  # fixed | paragraph | clause
    size: int = 500  # max characters per chunk
    overlap: int = 0  # characters shared by consecutive windows (fixed windows and long-paragraph splits)
    min_size: int = 40  # paragraphs/clauses shorter than this are merged into a neighbour

    def validate(self) -> None:
        _choice("chunking.strategy", self.strategy, ["fixed", "paragraph", "clause"])
        if self.size < 50:
            raise ConfigError("chunking.size must be >= 50")
        if not 0 <= self.overlap < self.size:
            raise ConfigError("chunking.overlap must be in [0, size)")


@dataclass
class TokenizerConfig:
    lowercase: bool = True
    stopwords: bool = False
    stemming: str = "none"  # none | light

    def validate(self) -> None:
        _choice("tokenizer.stemming", self.stemming, ["none", "light"])


@dataclass
class RoutingConfig:
    mode: str = "none"  # none | hard | soft
    top_docs: int = 1  # hard routing: search only the best-matching N contracts
    soft_weight: float = 0.5  # soft routing: final = retrieval_norm + soft_weight * route_similarity
    head_chars: int = 2000  # characters of each contract used as its "title card"
    query_part: str = "full"  # full | question: text used for passage retrieval when a title is present

    def validate(self) -> None:
        _choice("routing.mode", self.mode, ["none", "hard", "soft"])
        _choice("routing.query_part", self.query_part, ["full", "question"])
        if self.top_docs < 1:
            raise ConfigError("routing.top_docs must be >= 1")


@dataclass
class DenseConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    query_prefix: str = ""
    batch_size: int = 32


@dataclass
class HybridConfig:
    components: List[str] = field(default_factory=lambda: ["bm25", "dense"])
    fusion: str = "rrf"  # rrf | weighted
    weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    rrf_k: int = 60

    def validate(self) -> None:
        _choice("hybrid.fusion", self.fusion, ["rrf", "weighted"])
        for c in self.components:
            _choice("hybrid.components[]", c, ["bm25", "tfidf", "dense"])
        if len(self.weights) != len(self.components):
            raise ConfigError("hybrid.weights must have one weight per component")


@dataclass
class RetrievalConfig:
    method: str = "bm25"  # bm25 | tfidf | dense | hybrid
    candidates: int = 64  # ranked list length kept before post-processing
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    tfidf_ngram_max: int = 1
    tfidf_sublinear_tf: bool = True
    expansion: str = "none"  # none | lexicon (concept lexicon in configs/lexicons/)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)

    def validate(self) -> None:
        _choice("retrieval.method", self.method, ["bm25", "tfidf", "dense", "hybrid"])
        _choice("retrieval.expansion", self.expansion, ["none", "lexicon"])
        if self.candidates < 1:
            raise ConfigError("retrieval.candidates must be >= 1")
        self.tokenizer.validate()
        self.hybrid.validate()


@dataclass
class RerankConfig:
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 32  # rerank only the first N candidates
    weight: float = 1.0  # final = weight * ce_norm + (1 - weight) * first_stage_norm

    def validate(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ConfigError("rerank.weight must be in [0, 1]")


@dataclass
class PostprocessConfig:
    dedupe_overlaps: bool = True  # drop candidates overlapping a higher-ranked one (keeps metrics un-inflated)
    expand_chars: int = 0  # widen each passage by N characters on both sides (clipped to the document)


@dataclass
class Gate:
    signal: str  # see abstention.SIGNALS
    threshold: Optional[float] = None  # None = calibrate on train


@dataclass
class AbstentionConfig:
    enabled: bool = True
    gates: List[Gate] = field(default_factory=lambda: [Gate(signal="top_score")])
    max_false_abstention: float = 0.10  # calibration budget on TRAIN answerable queries

    def validate(self) -> None:
        from .abstention import SIGNALS

        for g in self.gates:
            _choice("abstention.gates[].signal", g.signal, sorted(SIGNALS))
        if not 0.0 <= self.max_false_abstention < 1.0:
            raise ConfigError("abstention.max_false_abstention must be in [0, 1)")


@dataclass
class EvaluationConfig:
    ks: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    primary_k: int = 4


@dataclass
class PipelineConfig:
    name: str = "unnamed"
    description: str = ""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    abstention: AbstentionConfig = field(default_factory=AbstentionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> "PipelineConfig":
        self.chunking.validate()
        self.routing.validate()
        self.retrieval.validate()
        self.rerank.validate()
        self.abstention.validate()
        if self.evaluation.primary_k not in self.evaluation.ks:
            raise ConfigError("evaluation.primary_k must be one of evaluation.ks")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self, include_thresholds: bool = True) -> str:
        d = self.to_dict()
        d.pop("name", None)
        d.pop("description", None)
        if not include_thresholds:
            for g in d["abstention"]["gates"]:
                g["threshold"] = None
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def needs_calibration(self) -> bool:
        return self.abstention.enabled and any(g.threshold is None for g in self.abstention.gates)

    def uses_dense(self) -> bool:
        r = self.retrieval
        return r.method == "dense" or (r.method == "hybrid" and "dense" in r.hybrid.components) or any(
            g.signal == "dense_top_cosine" for g in self.abstention.gates)


# ---------------------------------------------------------------- construction


def _choice(name: str, value: Any, allowed: List[str]) -> None:
    if value not in allowed:
        raise ConfigError(f"{name}={value!r} is not one of {allowed}")


def _build(cls, data: Any, path: str):
    if not is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'} must be a mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"unknown key(s) at {path or 'top level'}: {sorted(unknown)}")
    kwargs = {}
    hints = _field_classes(cls)
    for name, value in data.items():
        sub = hints.get(name)
        if sub is Gate:
            if not isinstance(value, list):
                raise ConfigError(f"{path}.{name} must be a list")
            kwargs[name] = [_build(Gate, v, f"{path}.{name}[{i}]") for i, v in enumerate(value)]
        elif sub is not None:
            kwargs[name] = _build(sub, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"{path or 'config'}: {e}") from e


def _field_classes(cls) -> Dict[str, Any]:
    """Map field name -> nested dataclass (annotations are strings under `from __future__ import annotations`)."""
    nested = {
        PipelineConfig: {"chunking": ChunkingConfig, "routing": RoutingConfig, "retrieval": RetrievalConfig,
                         "rerank": RerankConfig, "postprocess": PostprocessConfig, "abstention": AbstentionConfig,
                         "evaluation": EvaluationConfig},
        RetrievalConfig: {"tokenizer": TokenizerConfig, "dense": DenseConfig, "hybrid": HybridConfig},
        AbstentionConfig: {"gates": Gate},
    }
    return nested.get(cls, {})


def deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_from_dict(data: Dict[str, Any]) -> PipelineConfig:
    return _build(PipelineConfig, data, "").validate()


def apply_patch(cfg: PipelineConfig, patch: Dict[str, Any], name: Optional[str] = None) -> PipelineConfig:
    d = deep_merge(cfg.to_dict(), patch)
    if name:
        d["name"] = name
    return config_from_dict(d)


def resolve_config_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    for cand in (CONFIG_DIR / name_or_path, CONFIG_DIR / f"{name_or_path}.yaml"):
        if cand.exists():
            return cand
    raise ConfigError(f"configuration not found: {name_or_path} (looked in {CONFIG_DIR})")


def _load_raw(path: Path, seen: Optional[List[Path]] = None) -> Dict[str, Any]:
    seen = seen or []
    if path in seen:
        raise ConfigError(f"circular extends: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"cannot parse {path}: {e}") from e
    parent = data.pop("extends", None)
    if parent:
        base = _load_raw(resolve_config_path(str(path.parent / parent)) if (path.parent / parent).exists()
                         else resolve_config_path(parent), seen + [path])
        data = deep_merge(base, data)
    return data


def load_config(name_or_path: str) -> PipelineConfig:
    return config_from_dict(_load_raw(resolve_config_path(name_or_path)))


def save_config(cfg: PipelineConfig, path: Path, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    if header:
        text = "".join(f"# {line}\n" for line in header.splitlines()) + text
    path.write_text(text, encoding="utf-8")


def list_configs() -> List[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

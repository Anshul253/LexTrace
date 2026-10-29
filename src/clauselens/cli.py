"""ClauseLens command-line interface.

    python -m clauselens prepare | validate | build-index | query | evaluate | analyze | compare | improve | tables | test

Exit codes: 0 ok (passages returned) | 10 abstained (not an error) | 2 invalid input | 3 data/config problem."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer

from .config import ConfigError, load_config, save_config
from .paths import CONFIG_DIR, EXPERIMENTS_DIR, ROOT

app = typer.Typer(add_completion=False, help="ClauseLens - clause evidence retrieval over contracts.",
                  no_args_is_help=True)

EXIT_ABSTAINED, EXIT_INVALID, EXIT_DATA = 10, 2, 3


def _fail(msg: str, code: int = EXIT_DATA) -> None:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _load(config: str):
    try:
        return load_config(config)
    except ConfigError as e:
        _fail(str(e))


def _engine(cfg):
    from .data.benchmark import DatasetError
    from .data.corpus import MissingDocumentError
    from .pipeline import Engine
    from .workspace import corpus

    try:
        return Engine(cfg, corpus())
    except (DatasetError, MissingDocumentError) as e:
        _fail(str(e))
    except RuntimeError as e:  # model download / load problems
        _fail(str(e))


@app.command()
def prepare() -> None:
    """Download the benchmark file and the referenced contracts (pinned revision) and write the manifest."""
    from .data.download import DownloadError, prepare as run

    try:
        run(log=typer.echo)
    except DownloadError as e:
        _fail(str(e))


@app.command()
def validate(out: Path = typer.Option(EXPERIMENTS_DIR / "dataset_validation.json", help="where to write the report")) -> None:
    """Check selection, split, gold offsets, unanswerable lock and absence checks."""
    from .data.benchmark import DatasetError
    from .data.unanswerable import absence_report, file_sha256, load_unanswerable, split_counts
    from .data.validate import validate_offsets
    from .workspace import all_benchmark_queries, corpus

    try:
        qs, c = all_benchmark_queries(), corpus()
        rep = validate_offsets(qs, c)
        un = load_unanswerable()
    except DatasetError as e:
        _fail(str(e))
    absence = absence_report(un, c)
    rep["unanswerable"] = {"sha256": file_sha256(), "lock_verified": True, "split_counts": split_counts(un),
                           "absence_check_failures": [r for r in absence if not r["ok"]]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    typer.echo(f"queries {rep['n_queries']} (split {rep['split_counts']}), gold spans {rep['n_gold_spans']}, "
               f"documents {rep['n_documents']}, offset failures {len(rep['offset_failures'])}")
    typer.echo(f"unanswerable {rep['unanswerable']['split_counts']}, lock ok, absence-check failures "
               f"{len(rep['unanswerable']['absence_check_failures'])}")
    if rep["offset_failures"] or rep["unanswerable"]["absence_check_failures"]:
        raise typer.Exit(1)


@app.command("build-index")
def build_index(config: str = typer.Option("final", help="config name in configs/ or a YAML path")) -> None:
    """Chunk the corpus and build the configured indexes (dense embeddings are cached in artifacts/)."""
    cfg = _load(config)
    eng = _engine(cfg)
    sizes = [c.end - c.start for c in eng.chunks]
    typer.echo(f"{cfg.name}: {len(eng.corpus)} contracts, {len(eng.chunks)} chunks (mean {sum(sizes) / len(sizes):.0f} "
               f"chars), retrievers {sorted(eng.retrievers)}, built in {eng.build_seconds:.2f}s")


@app.command()
def query(question: str = typer.Argument(..., help="natural-language question"),
          config: str = typer.Option("final"), k: int = typer.Option(5, "--k", "-k", min=1, max=64),
          as_json: bool = typer.Option(False, "--json", help="print the full response as JSON"),
          show_candidates: bool = typer.Option(False, help="when abstaining, also list the (insufficient) best passages")) -> None:
    """Search the contracts for passages answering QUESTION."""
    from .pipeline import InvalidQueryError

    cfg = _load(config)
    if cfg.needs_calibration():
        _fail(f"config {cfg.name!r} has uncalibrated thresholds; run `evaluate --split train --config {config} "
              "--save-calibrated` first")
    eng = _engine(cfg)
    try:
        resp = eng.search(question, k=k)
    except InvalidQueryError as e:
        _fail(str(e), EXIT_INVALID)
    if as_json:
        typer.echo(json.dumps(resp.to_dict(), indent=2, default=float))
    else:
        typer.echo(f"config {resp.config_name} | {resp.method} | {resp.latency_ms:.0f} ms")
        if resp.contract_named:
            best = resp.routed_contracts[0] if resp.routed_contracts else ("-", 0.0)
            typer.echo(f"contract named: {resp.contract_named}\nbest match: {best[0]} (similarity {best[1]:.3f})")
        if resp.abstained:
            typer.secho(f"\nNO RESULT - {resp.abstention_reason}", fg=typer.colors.YELLOW)
            if show_candidates:
                for p in resp.candidates[:k]:
                    typer.echo(f"  (below threshold) #{p.rank} {p.doc_id} [{p.start}:{p.end}] score={p.score:.3f}")
        for p in resp.passages:
            typer.secho(f"\n#{p.rank}  {p.doc_id}  [{p.start}:{p.end}]  score={p.score:.4f}  ({p.method})", bold=True)
            typer.echo(p.text.strip()[:1200])
    raise typer.Exit(EXIT_ABSTAINED if resp.abstained else 0)


@app.command()
def evaluate(split: str = typer.Option("train", help="train (development) or test (final reporting only)"),
             config: str = typer.Option("final"), tag: Optional[str] = typer.Option(None, help="result file prefix"),
             confirm_final: bool = typer.Option(False, "--confirm-final",
                                                help="required for --split test; the run is recorded in the test ledger"),
             note: str = typer.Option("", help="ledger note for test runs"),
             save_calibrated: bool = typer.Option(False, help="write thresholds calibrated on train back to the config")) -> None:
    """Evaluate a configuration on a split and save metrics + predictions under experiments/results/."""
    from .evaluation.evaluator import SplitGuardError, evaluate as run, record_test_run, save_result
    from .workspace import benchmark_split, unanswerable_split

    cfg = _load(config)
    if split == "test" and not confirm_final:
        _fail("test evaluation is reserved for final reporting; re-run with --confirm-final", EXIT_INVALID)
    eng = _engine(cfg)
    try:
        res = run(cfg, eng, benchmark_split(split), unanswerable_split(split), split, allow_test=(split == "test"))
    except SplitGuardError as e:
        _fail(str(e), EXIT_INVALID)
    path = save_result(res, tag or cfg.name)
    s = res["summary"]
    for k in cfg.evaluation.ks:
        r = s["ranking"][str(k)]
        typer.echo(f"k={k:<3} recall {r['recall']:.3f}  precision {r['precision']:.3f}  span-hit {r['span_hit']:.3f}  "
                   f"(e2e recall {s['end_to_end'][str(k)]['recall']:.3f})")
    a = s["abstention"]
    typer.echo(f"MRR {s['mrr']:.3f} | false abstention {a['false_abstentions']}/{a['n_answerable']} | correct abstention "
               f"{a['correct_abstentions']}/{a['n_unanswerable']} | p95 {s['latency_ms']['p95']:.0f} ms | saved {path}")
    if split == "test":
        n = record_test_run(res["config"], s, note or f"final reporting: {cfg.name}")
        typer.echo(f"test ledger now holds {n} evaluation(s)")
    if save_calibrated and res["calibration"] is not None:
        target = CONFIG_DIR / f"{cfg.name}.yaml" if not Path(config).exists() else Path(config)
        save_config(res["config"], target, header="thresholds calibrated on train by `clauselens evaluate --save-calibrated`")
        typer.echo(f"calibrated thresholds written to {target}")


@app.command()
def analyze(config: str = typer.Option("final"), split: str = typer.Option("train"),
            out: Path = typer.Option(EXPERIMENTS_DIR / "error_analysis.json")) -> None:
    """Write per-failure evidence records (query, gold, top passages, diagnosis) for error analysis."""
    from .evaluation.analysis import error_records
    from .evaluation.evaluator import evaluate as run
    from .workspace import benchmark_split, unanswerable_split

    if split != "train":
        _fail("error analysis runs on train so that no tuning signal comes from test", EXIT_INVALID)
    cfg = _load(config)
    eng = _engine(cfg)
    res = run(cfg, eng, benchmark_split(split), unanswerable_split(split), split)
    rec = error_records(res, eng.corpus)
    out.write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
    typer.echo(f"{len(rec['failures'])} failure records, categories {rec['counts']} -> {out}")


@app.command()
def compare(base: str = typer.Option("final", help="config whose chunking/routing is shared by all methods"),
            out: Path = typer.Option(EXPERIMENTS_DIR / "method_comparison.json")) -> None:
    """Train-only comparison of BM25, TF-IDF, dense and hybrid retrieval under one configuration."""
    from .optimization.compare import compare_methods

    rows = compare_methods(_load(base), log=typer.echo)
    out.write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    typer.echo(f"-> {out}")


@app.command()
def improve(start: str = typer.Option("baseline"), iterations: int = typer.Option(12, min=5),
            resume: bool = typer.Option(False, help="append to the existing log instead of starting fresh")) -> None:
    """Run the train-only self-improvement loop; writes experiments/experiment_log.jsonl and configs/final.yaml."""
    from .optimization.loop import ImprovementLoop

    t0 = time.time()
    res = ImprovementLoop(_load(start), log=typer.echo).run(iterations, fresh=not resume)
    typer.echo(f"done in {time.time() - t0:.0f}s; accepted: {res['accepted']}")


@app.command()
def tables(out: Path = typer.Option(EXPERIMENTS_DIR / "report_tables.md")) -> None:
    """Render markdown tables from saved results (used to build REPORT.md)."""
    from .evaluation.tables import render_all

    out.write_text(render_all(), encoding="utf-8")
    typer.echo(f"-> {out}")


@app.command()
def test(integration: bool = typer.Option(False, help="also run tests needing the downloaded dataset"),
         extra: Optional[List[str]] = typer.Argument(None)) -> None:
    """Run the automated tests (unit tests by default)."""
    cmd = [sys.executable, "-m", "pytest", "-q"] + (["-m", "integration or not integration"] if integration else [])
    raise typer.Exit(subprocess.call(cmd + (extra or []), cwd=ROOT))


if __name__ == "__main__":  # pragma: no cover
    app()

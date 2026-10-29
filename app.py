"""LexTrace web app.   streamlit run app.py

A contract-review workspace over the indexed agreements:
  Clause Finder     pick a contract and a clause type (or type any question) -> ranked evidence or an abstention
  Contract Library  browse the indexed agreements, read one in full, see retrieved evidence highlighted in place
  Overview          corpus, benchmark and headline measured results
  Evaluation / Experiments / Configuration   saved metrics, the improvement-loop log, the active pipeline
All retrieval goes through `Engine.search`, the same entry point as the CLI; every number shown is read from data
files or result files the CLI wrote."""
from __future__ import annotations

import html
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lextrace.config import ConfigError, load_config, resolve_config_path  # noqa: E402
from lextrace.data.benchmark import DatasetError  # noqa: E402
from lextrace.data.corpus import MissingDocumentError  # noqa: E402
from lextrace.pipeline import Engine  # noqa: E402
from lextrace.ui import viewmodel as vm  # noqa: E402
from lextrace.ui.theme import CSS  # noqa: E402

st.set_page_config(page_title="LexTrace - Contract Intelligence", page_icon=":material/gavel:", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

DISCLAIMER = ("LexTrace finds candidate evidence in the indexed agreements. It is not legal advice, and an empty "
              "result does not prove that an agreement lacks a clause.")
ANY_CONTRACT = "All indexed contracts (no contract named)"


# ------------------------------------------------------------------ data & engine

@st.cache_resource(show_spinner=False)
def get_engine(name: str, mtime: float) -> Engine:  # mtime invalidates the cache when the YAML changes
    from lextrace.workspace import corpus

    return Engine(load_config(name), corpus())


@st.cache_resource(show_spinner=False)
def library():
    """(corpus, catalog rows, doc->title) or raises a dataset error."""
    from lextrace.workspace import all_benchmark_queries, corpus

    c = corpus()
    titles = vm.contract_titles(all_benchmark_queries())
    return c, vm.contract_catalog(c, titles), titles


def load_library():
    try:
        return library(), None
    except (DatasetError, MissingDocumentError) as e:
        return None, f"Dataset not available: {e}. Run `python -m lextrace prepare`."


def active_engine():
    name = st.session_state.get("config_name")
    if not name:
        return None, ("No calibrated configuration found. Run `python -m lextrace improve` (or "
                      "`python -m lextrace evaluate --split train --config baseline --save-calibrated`).")
    try:
        with st.spinner(f"Loading the '{name}' index (the first load of a dense model can take a minute)..."):
            return get_engine(name, resolve_config_path(name).stat().st_mtime), None
    except (DatasetError, MissingDocumentError) as e:
        return None, f"Dataset not available: {e}"
    except ConfigError as e:
        return None, f"Configuration problem: {e}"
    except Exception as e:  # noqa: BLE001
        return None, f"Indexing failed ({type(e).__name__}): {e}"


# ------------------------------------------------------------------ components

def eyebrow(text: str) -> None:
    st.markdown(f'<div class="cl-eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


def banner(kind: str, title: str, body: str = "") -> None:
    st.markdown(f'<div class="cl-banner {kind}" role="status"><b>{html.escape(title)}</b>{html.escape(body)}</div>',
                unsafe_allow_html=True)


def chips(items) -> None:
    st.markdown(" ".join(f'<span class="cl-chip">{html.escape(str(x))}</span>' for x in items), unsafe_allow_html=True)


def tile(col, label: str, value: str, note: str = "") -> None:
    col.markdown(f'<div class="cl-tile"><div class="cl-tile-label">{html.escape(label)}</div>'
                 f'<div class="cl-tile-value">{html.escape(value)}</div>'
                 f'<div class="cl-tile-note">{html.escape(note)}</div></div>', unsafe_allow_html=True)


def evidence_card(c: vm.EvidenceCard, title: str, max_score: float, selected: bool, weak: bool = False) -> None:
    width = 100 * (c.score / max_score) if max_score > 0 else 0
    label = "below threshold" if weak else f"#{c.rank}"
    excerpt = c.text if len(c.text) <= 900 else c.text[:900] + " ..."
    st.markdown(
        f'<article class="cl-card{" selected" if selected else ""}{" weak" if weak else ""}">'
        f'<div class="cl-card-head"><span class="cl-rank">{label}</span>'
        f'<span class="cl-type">{html.escape(vm.agreement_type(c.doc_id))}</span>'
        f'<span class="cl-contract">{html.escape(title)}</span></div>'
        f'<div class="cl-meta">chars {c.start:,}&ndash;{c.end:,} &middot; {c.length:,} chars &middot; '
        f'score {c.score:.4f} &middot; {html.escape(c.method)}</div>'
        f'<div class="cl-excerpt">{html.escape(excerpt)}</div>'
        f'<div class="cl-bar" title="score relative to the top result"><div style="width:{max(width, 0):.0f}%"></div></div>'
        f'</article>', unsafe_allow_html=True)


# ------------------------------------------------------------------ Clause Finder

def page_finder() -> None:
    eyebrow("Clause Finder")
    st.title("Find the clause that answers your question")
    st.caption(DISCLAIMER)
    lib, err = load_library()
    engine, err2 = active_engine()
    if err or err2:
        banner("err", "The workspace is unavailable", " " + (err or err2))
        return
    corpus, catalog, titles = lib

    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        options = [ANY_CONTRACT] + [r["doc_id"] for r in catalog]
        fmt = lambda d: d if d == ANY_CONTRACT else f"{titles.get(d, vm.contract_label(d))}  ·  {vm.agreement_type(d)}"  # noqa: E731
        doc = c1.selectbox("1 · Contract", options, format_func=fmt, key="finder_doc",
                           help="Naming the contract restricts the search to it (contract routing).")
        k = c2.number_input("Passages to show", min_value=1, max_value=20, value=5, step=1)
        clause = st.pills("2 · Clause type (fills in a question you can edit)", list(vm.CLAUSE_TEMPLATES),
                          key="finder_clause", selection_mode="single")
        if clause and st.session_state.get("_last_clause") != clause:
            st.session_state["finder_q"] = vm.CLAUSE_TEMPLATES[clause]
            st.session_state["_last_clause"] = clause
        q = st.text_area("3 · Question", key="finder_q", height=80, max_chars=1500,
                         placeholder="e.g. Can the licensee sublicense the content to a third-party platform?")
        title = None if doc == ANY_CONTRACT else titles.get(doc, vm.contract_label(doc))
        full = q.strip() if q.strip().lower().startswith("consider ") else vm.compose_question(title, q)
        st.caption("Sent to the engine: " + (full if full else "—"))
        run = st.button("Find evidence", type="primary", use_container_width=True, disabled=not q.strip())
    if run:
        st.session_state["view"] = vm.run_search(engine, full, int(k))
        st.session_state["selected"] = 0

    view = st.session_state.get("view")
    if view is None:
        st.info("Choose a contract and a clause type, or type any question, then press **Find evidence**. Each result "
                "shows the exact character range in the original agreement.")
        return
    if view.status in ("invalid", "error"):
        banner("err", "No search was run" if view.status == "invalid" else "Search failed", " " + view.message)
        return
    m = view.meta
    chips([f"config {m['config']}", m["method"], f"{m['latency_ms']:.0f} ms", f"{m['n_candidates']} candidates"])
    if view.status == "abstained":
        banner("abstain", "No clause met the evidence threshold", " " + view.message)
        st.caption("Treat this as 'not found by the system', not as 'the agreement has no such clause'. Read the "
                   "agreement in the Contract Library if it matters.")
        if view.weak_cards and st.toggle("Show the closest passages anyway (below threshold)", value=False):
            top = max(c.score for c in view.weak_cards)
            for c in view.weak_cards:
                evidence_card(c, titles.get(c.doc_id, c.contract), top, False, weak=True)
        signals_panel(m)
        return
    banner("ok", view.message, " Verify each passage in context before relying on it.")
    left, right = st.columns([3, 2], gap="large")
    sel = min(st.session_state.get("selected", 0), len(view.cards) - 1)
    with left:
        top = max(c.score for c in view.cards)
        for i, c in enumerate(view.cards):
            evidence_card(c, titles.get(c.doc_id, c.contract), top, i == sel)
            b1, b2 = st.columns(2)
            if b1.button(f"Inspect #{c.rank}", key=f"insp{i}", use_container_width=True):
                st.session_state["selected"] = i
                st.rerun()
            if b2.button(f"Open #{c.rank} in Contract Library", key=f"lib{i}", use_container_width=True):
                st.session_state["lib_doc"] = c.doc_id
                st.session_state["lib_spans"] = [(x.start, x.end) for x in view.cards if x.doc_id == c.doc_id]
                st.switch_page(PAGES["library"])
    with right:
        c = view.cards[sel]
        eyebrow("Evidence inspector")
        st.subheader(f"#{c.rank} · {vm.agreement_type(c.doc_id)}")
        st.markdown(f'<dl class="cl-kv"><dt>Contract</dt><dd>{html.escape(titles.get(c.doc_id, c.contract))}</dd>'
                    f'<dt>File</dt><dd>{html.escape(c.doc_id)}</dd><dt>Offsets</dt><dd>[{c.start}, {c.end})</dd>'
                    f'<dt>Length</dt><dd>{c.length} chars</dd><dt>Score</dt><dd>{c.score:.6f}</dd>'
                    f'<dt>Method</dt><dd>{html.escape(c.method)}</dd></dl>', unsafe_allow_html=True)
        st.markdown(f'<div class="cl-doc" tabindex="0" aria-label="passage in context">{vm.highlight_html(c)}</div>',
                    unsafe_allow_html=True)
        st.markdown('<p class="cl-note">Highlighted text is exactly <code>document[start:end]</code>; grey text is the '
                    'surrounding agreement.</p>', unsafe_allow_html=True)
        signals_panel(m)


def signals_panel(m: dict) -> None:
    with st.expander("Why this result? Contract match and abstention signals"):
        if m.get("contract_named"):
            st.write(f"**Contract named:** {m['contract_named']}")
            st.dataframe(pd.DataFrame(m["routed"], columns=["closest indexed contracts", "similarity"]),
                         hide_index=True, use_container_width=True)
        else:
            st.write("No contract was named, so all indexed agreements were searched.")
        rows = [{"signal": s, "threshold": t, "value": m["signals"].get(s),
                 "passed": (m["signals"].get(s, 0) >= t) if t is not None else None} for s, t in m["gates"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ Contract Library

def page_library() -> None:
    eyebrow("Contract Library")
    st.title("Indexed agreements")
    lib, err = load_library()
    if err:
        banner("err", "Library unavailable", " " + err)
        return
    corpus, catalog, titles = lib
    df = pd.DataFrame(catalog)
    types = sorted(df["type"].unique())
    f1, f2 = st.columns([2, 3])
    picked_types = f1.multiselect("Agreement type", types, placeholder="All types")
    name_filter = f2.text_input("Filter by name or party", placeholder="e.g. Sponsorship, Roanoke, Jaguars")
    view = df[df["type"].isin(picked_types)] if picked_types else df
    if name_filter.strip():
        view = view[view["title"].str.contains(name_filter.strip(), case=False, regex=False)]
    st.caption(f"{len(view)} of {len(df)} agreements · {int(df['characters'].sum()):,} characters indexed in total")
    st.dataframe(view[["title", "type", "characters"]].rename(columns={"title": "agreement", "characters": "chars"}),
                 hide_index=True, use_container_width=True, height=min(38 * (len(view) + 1), 300))
    if view.empty:
        return
    ids = list(view["doc_id"])
    current = st.session_state.get("lib_doc")
    doc = st.selectbox("Read agreement", ids, index=ids.index(current) if current in ids else 0,
                       format_func=lambda d: f"{titles.get(d, vm.contract_label(d))}  ·  {vm.agreement_type(d)}")
    if doc != current:
        st.session_state["lib_doc"], st.session_state["lib_spans"] = doc, []
    text = corpus.text(doc)
    left, right = st.columns([1, 3], gap="large")
    with left:
        eyebrow("Details")
        st.markdown(f'<dl class="cl-kv"><dt>Type</dt><dd>{html.escape(vm.agreement_type(doc))}</dd>'
                    f'<dt>File</dt><dd>{html.escape(doc)}</dd><dt>Length</dt><dd>{len(text):,} chars</dd></dl>',
                    unsafe_allow_html=True)
        needle = st.text_input("Find in this agreement", placeholder="e.g. sublicense")
        hits = vm.find_in_document(text, needle)
        evidence = st.session_state.get("lib_spans") or []
        if evidence:
            st.markdown(f'<span class="cl-chip">{len(evidence)} retrieved passage(s) highlighted in teal</span>',
                        unsafe_allow_html=True)
        if needle.strip():
            st.caption(f"{len(hits)} match(es) for '{needle.strip()}' highlighted in amber" + (" (first 50)" if len(hits) == 50 else ""))
            for s, e in hits[:10]:
                st.markdown(f'<span class="cl-meta">[{s}, {e})</span> {html.escape(text[max(0, s - 40):e + 40])}',
                            unsafe_allow_html=True)
    with right:
        spans = [(s, e, "cl-hit") for s, e in evidence] + [(s, e, "cl-find") for s, e in hits
                                                           if not any(s < b and a < e for a, b in evidence)]
        body, pos = [], 0
        for s, e, cls in sorted(spans):
            if s < pos:
                continue
            body.append(vm.document_html(text[pos:s], []))
            body.append(f'<mark class="{cls}">{html.escape(text[s:e]).replace(chr(10), "<br>")}</mark>')
            pos = e
        body.append(vm.document_html(text[pos:], []))
        st.markdown(f'<div class="cl-doc tall" tabindex="0" aria-label="full agreement text">{"".join(body)}</div>',
                    unsafe_allow_html=True)


# ------------------------------------------------------------------ Overview

def page_overview() -> None:
    eyebrow("Overview")
    st.title("Corpus and measured quality")
    lib, err = load_library()
    if err:
        banner("err", "Dataset unavailable", " " + err)
        return
    corpus, catalog, _ = lib
    from lextrace.data.unanswerable import load_unanswerable, split_counts
    from lextrace.workspace import all_benchmark_queries

    qs = all_benchmark_queries()
    un = split_counts(load_unanswerable())
    n_train = sum(q.split == "train" for q in qs)
    c = st.columns(4)
    tile(c[0], "Indexed agreements", f"{len(corpus)}", f"{corpus.total_chars():,} characters")
    tile(c[1], "Benchmark questions", f"{len(qs)}", f"LegalBench-RAG CUAD mini · {n_train} train / {len(qs) - n_train} test")
    tile(c[2], "Unanswerable questions", f"{un['total']}", f"hand-written, frozen · {un['train']} train / {un['test']} test")
    n_test = vm.test_run_count()
    tile(c[3], "Held-out test evaluations", f"{n_test or 0}", "recorded in experiments/test_runs.jsonl")

    st.subheader("Headline results (train split, from saved evaluation files)")
    hm = vm.headline_metrics("train")
    if not hm["final"] and not hm["baseline"]:
        banner("abstain", "No saved train results yet", " Run `python -m lextrace evaluate --split train`.")
    else:
        c = st.columns(4)
        for col, (label, key, fmt) in zip(c, [("Recall@4", ("ranking", "4", "recall"), "{:.3f}"),
                                              ("Precision@4", ("ranking", "4", "precision"), "{:.3f}"),
                                              ("MRR", ("mrr",), "{:.3f}"),
                                              ("Correct abstentions", ("abstention", "correct_abstentions"), "{}")]):
            def get(s):
                for part in key:
                    s = s[part]
                return s
            b = get(hm["baseline"]) if hm["baseline"] else None
            f = get(hm["final"]) if hm["final"] else None
            tile(col, label, fmt.format(f) if f is not None else "pending",
                 f"baseline {fmt.format(b)}" if b is not None else "baseline pending")
        st.caption("Test-split numbers are reported in REPORT.md and on the Evaluation page once the final evaluation has run.")

    st.subheader("Agreement types in the corpus")
    counts = Counter(r["type"] for r in catalog)
    st.bar_chart(pd.DataFrame({"agreements": counts}).sort_values("agreements", ascending=False), horizontal=True,
                 color="#0f7c80")
    st.subheader("How a question is answered")
    st.markdown(
        '<ol class="cl-steps"><li><b>Name the contract</b> - the router matches the agreement named in the question '
        'against each contract\'s title and opening text.</li><li><b>Retrieve</b> - passages of that agreement are '
        'ranked by the configured retriever (BM25, TF-IDF, dense or hybrid, optional reranker).</li>'
        '<li><b>Check the evidence</b> - calibrated gates decide whether the best passage is strong enough; if not, '
        'LexTrace abstains.</li><li><b>Cite</b> - every passage carries its file and exact character offsets.</li></ol>',
        unsafe_allow_html=True)


# ------------------------------------------------------------------ Evaluation / Experiments / Configuration

def page_evaluation() -> None:
    eyebrow("Measured quality")
    st.title("Evaluation results")
    st.caption("Read from experiments/results/*.json, written by `lextrace evaluate`. Nothing here is computed in the browser.")
    split = st.segmented_control("Split", ["train", "test"], default="train", key="eval_split") or "train"
    res = vm.load_results(split)
    if not res:
        banner("abstain", f"No saved {split} results", " Run `python -m lextrace evaluate --split train` first."
               if split == "train" else " The held-out test split is evaluated only for final reporting.")
        return
    if split == "test":
        n = vm.test_run_count()
        if n:
            st.caption(f"Test ledger: {n} recorded evaluation(s).")
    names = [t for t in res if not t.startswith(("loop_", "compare_"))] or list(res)
    chosen = st.multiselect("Configurations", list(res), default=names[:4])
    if not chosen:
        return
    first = res[chosen[0]]["summary"]
    rows = []
    for tag in chosen:
        s = res[tag]["summary"]
        for k in s["ranking"]:
            rows.append({"config": tag, "k": int(k), "recall": s["ranking"][k]["recall"],
                         "precision": s["ranking"][k]["precision"]})
    df = pd.DataFrame(rows)
    st.subheader("Character-level retrieval (ranking view)")
    st.caption(f"Macro mean over answerable {split} questions (n = {first['n_answerable']}).")
    st.dataframe(df.pivot_table(index="k", columns="config", values=["recall", "precision"]).round(3),
                 use_container_width=True)
    st.line_chart(df.pivot_table(index="k", columns="config", values="recall"), x_label="k", y_label="recall@k")
    st.subheader("Abstention")
    ab = []
    for t in chosen:
        a = res[t]["summary"]["abstention"]
        ab.append({"config": t, "false abstentions": f"{a['false_abstentions']}/{a['n_answerable']}",
                   "correct abstentions": f"{a['correct_abstentions']}/{a['n_unanswerable']}",
                   "MRR": round(res[t]["summary"]["mrr"], 3),
                   "p95 latency (ms)": round(res[t]["summary"]["latency_ms"]["p95"], 1)})
    st.dataframe(pd.DataFrame(ab), hide_index=True, use_container_width=True)


def page_experiments() -> None:
    eyebrow("Self-improvement loop")
    st.title("Experiment history")
    log = vm.load_experiment_log()
    if not log:
        banner("abstain", "No experiment log yet", " Run `python -m lextrace improve`.")
        return
    st.caption("Every iteration is evaluated on train only. Accepted changes become the parent of the next iteration.")
    skipped = [e for e in log if e["kind"] == "skipped"]
    log = [e for e in log if e["kind"] != "skipped"]
    rows = []
    for e in log:
        m = e.get("metrics_after") or e.get("metrics", {})
        rows.append({"iter": e["iteration"], "change": e.get("proposal_id", "baseline"), "decision": e["decision"],
                     "recall@4": m.get("recall@4"), "precision@4": m.get("precision@4"), "MRR": m.get("mrr"),
                     "correct abst.": m.get("correct_abstentions"), "why": e.get("explanation", "")})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True,
                 column_config={"why": st.column_config.TextColumn(width="large")})
    if skipped:
        st.caption("Skipped without evaluation (no effect on the parent): "
                   + "; ".join(f"{e['proposal_id']} - {e['explanation']}" for e in skipped))
    cands = [r["iter"] for r in rows if r["iter"]]
    if not cands:
        return
    entry = next(e for e in log if e["iteration"] == st.selectbox("Inspect iteration", cands))
    c1, c2 = st.columns(2)
    c1.markdown("**Change**")
    c1.code(yaml.safe_dump(entry.get("change", {}), sort_keys=False) or "-", language="yaml")
    c1.markdown(f"**Why it was tried:** {entry.get('reason', '-')}")
    c1.markdown(f"**Proposed by:** {entry.get('proposal_source', '-')}")
    c2.markdown("**Guardrails**")
    c2.dataframe(pd.DataFrame([{"guardrail": k, "pass": v} for k, v in entry.get("guardrails", {}).items()]),
                 hide_index=True, use_container_width=True)
    c2.markdown(f"**Decision:** {entry['decision']} - {entry.get('explanation', '')}")


def page_configuration() -> None:
    eyebrow("Pipeline")
    st.title("Configuration")
    name = st.session_state.get("config_name")
    if not name:
        banner("abstain", "No calibrated configuration", " Run the improvement loop first.")
        return
    cfg = load_config(name)
    st.write(f"**Active:** `{name}` - {cfg.description}")
    chips([f"chunks: {cfg.chunking.strategy} <= {cfg.chunking.size}", f"routing: {cfg.routing.mode}",
           f"retrieval: {cfg.retrieval.method}", f"rerank: {'on' if cfg.rerank.enabled else 'off'}",
           f"expansion: {cfg.retrieval.expansion}", f"fingerprint {cfg.fingerprint()}"])
    st.subheader("Abstention gates")
    st.dataframe(pd.DataFrame([{"signal": g.signal, "threshold": g.threshold} for g in cfg.abstention.gates]),
                 hide_index=True, use_container_width=True)
    st.caption(f"Thresholds were calibrated on train so that at most {cfg.abstention.max_false_abstention:.0%} of "
               "train answerable questions are refused.")
    st.subheader("Full configuration")
    st.code(yaml.safe_dump(cfg.to_dict(), sort_keys=False), language="yaml")
    manifest = ROOT / "data" / "dataset_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        st.subheader("Dataset provenance")
        st.write(f"{m['n_queries']} benchmark questions over {m['n_documents']} agreements from {m['source']} "
                 f"@ `{m['revision'][:10]}`; {m['checksums_verified']} file checksums verified.")


# ------------------------------------------------------------------ shell

PAGES = {
    "finder": st.Page(page_finder, title="Clause Finder", icon=":material/manage_search:", default=True),
    "library": st.Page(page_library, title="Contract Library", icon=":material/folder_open:"),
    "overview": st.Page(page_overview, title="Overview", icon=":material/dashboard:"),
    "evaluation": st.Page(page_evaluation, title="Evaluation", icon=":material/monitoring:"),
    "experiments": st.Page(page_experiments, title="Experiments", icon=":material/science:"),
    "configuration": st.Page(page_configuration, title="Configuration", icon=":material/tune:"),
}

st.logo(str(ROOT / "assets" / "lextrace_logo.svg"), size="large")
with st.sidebar:
    st.markdown('<div class="cl-sub">Contract Intelligence &middot; clause evidence retrieval</div>', unsafe_allow_html=True)
    configs = vm.usable_configs()
    if configs:
        default = configs.index("final") if "final" in configs else 0
        st.selectbox("Active configuration", configs, index=default, key="config_name",
                     help="Only configurations with calibrated thresholds are listed.")
    else:
        st.session_state["config_name"] = None
        st.caption("No calibrated configuration yet.")

nav = st.navigation({"Review": [PAGES["finder"], PAGES["library"]],
                     "Quality": [PAGES["overview"], PAGES["evaluation"], PAGES["experiments"]],
                     "System": [PAGES["configuration"]]})
with st.sidebar:
    st.markdown(f'<p class="cl-note" style="margin-top:24px">{html.escape(DISCLAIMER)}</p>', unsafe_allow_html=True)
nav.run()

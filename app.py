"""LexTrace web app.   streamlit run app.py

Views: Search (question -> ranked evidence or abstention, with a document inspector), Evaluation (saved metrics),
Experiments (improvement-loop log), Configuration (active pipeline and dataset status). All retrieval goes through
`Engine.search`, the same entry point as the CLI; all numbers shown are read from files the CLI wrote."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lextrace.config import ConfigError, load_config, resolve_config_path  # noqa: E402
from lextrace.data.benchmark import DatasetError  # noqa: E402
from lextrace.data.corpus import MissingDocumentError  # noqa: E402
from lextrace.pipeline import Engine  # noqa: E402
from lextrace.ui import viewmodel as vm  # noqa: E402
from lextrace.ui.theme import CSS  # noqa: E402

st.set_page_config(page_title="LexTrace - Contract Intelligence", page_icon=":material/plagiarism:", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

DISCLAIMER = ("LexTrace locates candidate evidence in the indexed contracts. It does not give legal advice, and an "
              "empty result does not prove that a contract lacks a clause.")


# ------------------------------------------------------------------ resources

@st.cache_resource(show_spinner=False)
def get_engine(name: str, mtime: float) -> Engine:  # mtime invalidates the cache when the YAML changes
    from lextrace.workspace import corpus

    return Engine(load_config(name), corpus())


def active_engine():
    name = st.session_state.get("config_name")
    if not name:
        return None, "No calibrated configuration found. Run `python -m lextrace improve` (or evaluate a config on train with --save-calibrated)."
    try:
        with st.spinner(f"Building the index for '{name}' (first load can take a minute for dense models)..."):
            return get_engine(name, resolve_config_path(name).stat().st_mtime), None
    except (DatasetError, MissingDocumentError) as e:
        return None, f"Dataset not available: {e}"
    except ConfigError as e:
        return None, f"Configuration problem: {e}"
    except Exception as e:  # noqa: BLE001
        return None, f"Indexing failed ({type(e).__name__}): {e}"


def banner(kind: str, title: str, body: str = "") -> None:
    st.markdown(f'<div class="cl-banner {kind}" role="status"><b>{html.escape(title)}</b>{html.escape(body)}</div>',
                unsafe_allow_html=True)


def evidence_card(c: vm.EvidenceCard, max_score: float, selected: bool, weak: bool = False) -> None:
    width = 100 * (c.score / max_score) if max_score > 0 else 0
    label = "below threshold" if weak else f"rank {c.rank}"
    excerpt = c.text if len(c.text) <= 900 else c.text[:900] + " ..."
    st.markdown(
        f'<div class="cl-card{" selected" if selected else ""}">'
        f'<div class="cl-card-head"><span class="cl-rank">{label}</span>'
        f'<span class="cl-contract">{html.escape(c.contract)}</span>'
        f'<span class="cl-meta">chars {c.start:,}-{c.end:,} &middot; {c.length:,} chars &middot; score {c.score:.4f}</span></div>'
        f'<div class="cl-excerpt">{html.escape(excerpt)}</div>'
        f'<div class="cl-bar" title="score relative to rank 1"><div style="width:{width:.0f}%"></div></div></div>',
        unsafe_allow_html=True)


# ------------------------------------------------------------------ pages

def page_search() -> None:
    st.markdown('<div class="cl-eyebrow">Clause search</div>', unsafe_allow_html=True)
    st.title("Ask a question about a contract")
    st.caption(DISCLAIMER)
    engine, err = active_engine()
    if err:
        banner("err", "The search workspace is unavailable", " " + err)
        return
    with st.expander("Example questions (from the train split and the unanswerable set)"):
        examples = example_questions()
        for i, (label, q) in enumerate(examples):
            if st.button(label, key=f"ex{i}", help=q):
                st.session_state["question"] = q
    with st.form("search", border=False):
        q = st.text_area("Question", key="question", height=96, max_chars=2000,
                         placeholder="Consider the Sponsorship Agreement between ... ; Is there an anti-assignment clause?",
                         help="Name the contract (\"Consider the <agreement> between <A> and <B>; ...\") for best results.")
        c1, c2 = st.columns([1, 3])
        k = c1.number_input("Passages to show", min_value=1, max_value=20, value=5, step=1)
        submitted = c2.form_submit_button("Search contracts", type="primary", use_container_width=True)
    if submitted:
        st.session_state["view"] = vm.run_search(engine, q, int(k))
        st.session_state["selected"] = 0
    view: vm.SearchView = st.session_state.get("view")
    if view is None:
        st.info("Enter a question and press **Search contracts**. Results show the exact character offsets of each passage.")
        return
    if view.status in ("invalid", "error"):
        banner("err", "No search was run" if view.status == "invalid" else "Search failed", " " + view.message)
        return
    m = view.meta
    st.markdown(" ".join(f'<span class="cl-chip">{html.escape(x)}</span>' for x in [
        f"config: {m['config']}", m["method"], f"{m['latency_ms']:.0f} ms", f"{m['n_candidates']} candidates"]),
        unsafe_allow_html=True)
    if view.status == "abstained":
        banner("abstain", "No passage met the evidence threshold", " " + view.message)
        if view.weak_cards and st.toggle("Show the closest (insufficient) passages anyway", value=False):
            top = max(c.score for c in view.weak_cards)
            for c in view.weak_cards:
                evidence_card(c, top, False, weak=True)
        details_panel(m)
        return
    banner("ok", view.message, " Read each passage in context before relying on it.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        top = max(c.score for c in view.cards)
        sel = st.session_state.get("selected", 0)
        for i, c in enumerate(view.cards):
            evidence_card(c, top, i == sel)
            if st.button(f"Inspect rank {c.rank} in the document", key=f"insp{i}"):
                st.session_state["selected"] = i
                st.rerun()
    with right:
        c = view.cards[min(st.session_state.get("selected", 0), len(view.cards) - 1)]
        st.markdown('<div class="cl-eyebrow">Evidence inspector</div>', unsafe_allow_html=True)
        st.subheader(f"Rank {c.rank}")
        st.markdown(f'<dl class="cl-kv"><dt>Document</dt><dd>{html.escape(c.doc_id)}</dd><dt>Offsets</dt>'
                    f'<dd>[{c.start}, {c.end})</dd><dt>Length</dt><dd>{c.length} chars</dd><dt>Score</dt>'
                    f'<dd>{c.score:.6f}</dd><dt>Method</dt><dd>{html.escape(c.method)}</dd></dl>', unsafe_allow_html=True)
        st.markdown(f'<div class="cl-doc" tabindex="0" aria-label="passage in context">{vm.highlight_html(c)}</div>',
                    unsafe_allow_html=True)
        st.markdown('<p class="cl-note">Highlighted text is exactly <code>document[start:end]</code>; grey text is '
                    'surrounding context.</p>', unsafe_allow_html=True)
        details_panel(m)


def details_panel(m: dict) -> None:
    with st.expander("Why this result? (routing and abstention signals)"):
        if m.get("contract_named"):
            st.write(f"**Contract named in the question:** {m['contract_named']}")
            st.dataframe(pd.DataFrame(m["routed"], columns=["closest indexed contracts", "similarity"]),
                         hide_index=True, use_container_width=True)
        else:
            st.write("The question names no contract, so all indexed contracts were searched.")
        rows = [{"gate signal": s, "threshold": t, "value": m["signals"].get(s),
                 "passed": (m["signals"].get(s, 0) >= t) if t is not None else None} for s, t in m["gates"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


@st.cache_data(show_spinner=False)
def example_questions():
    try:
        from lextrace.workspace import benchmark_split, unanswerable_split

        tr = benchmark_split("train")
        picks = [tr[i] for i in (0, 7, 23, 41) if i < len(tr)]
        un = unanswerable_split("train")[:2]
        return ([(f"Answerable: {q.query.split(';')[-1].strip()[:60]}", q.query) for q in picks]
                + [(f"Unanswerable: {q.query.split(';')[-1].strip()[:60]}", q.query) for q in un])
    except Exception:  # noqa: BLE001 - examples are optional
        return []


def page_evaluation() -> None:
    st.markdown('<div class="cl-eyebrow">Measured quality</div>', unsafe_allow_html=True)
    st.title("Evaluation results")
    st.caption("Read from experiments/results/*.json, written by `lextrace evaluate`. Nothing on this page is computed in the browser.")
    split = st.segmented_control("Split", ["train", "test"], default="train", key="eval_split") or "train"
    res = vm.load_results(split)
    if not res:
        banner("abstain", f"No saved {split} results", " Run `python -m lextrace evaluate --split train` first."
               if split == "train" else " The held-out test split is evaluated only for final reporting.")
        return
    if split == "test":
        n = vm.test_run_count()
        st.caption(f"Test ledger: {n} recorded evaluation(s) (experiments/test_runs.jsonl)." if n else "")
    names = [t for t in res if not t.startswith(("loop_", "compare_"))] or list(res)
    chosen = st.multiselect("Configurations", list(res), default=names[:4])
    if not chosen:
        return
    first = res[chosen[0]]["summary"]
    ks = list(first["ranking"])
    rows = []
    for tag in chosen:
        s = res[tag]["summary"]
        for k in ks:
            rows.append({"config": tag, "k": int(k), "recall": s["ranking"][k]["recall"],
                         "precision": s["ranking"][k]["precision"], "span hit": s["ranking"][k]["span_hit"],
                         "e2e recall": s["end_to_end"][k]["recall"]})
    df = pd.DataFrame(rows)
    st.subheader("Character-level retrieval (ranking view)")
    st.caption(f"Macro mean over answerable {split} queries (n = {first['n_answerable']}). "
               "End-to-end recall scores abstained answerable queries as 0.")
    st.dataframe(df.pivot_table(index="k", columns="config", values=["recall", "precision"]).round(3),
                 use_container_width=True)
    st.line_chart(df.pivot_table(index="k", columns="config", values="recall"), x_label="k", y_label="recall@k")
    st.subheader("Abstention")
    ab = [{"config": t, "false abstentions": f"{res[t]['summary']['abstention']['false_abstentions']}/"
                                             f"{res[t]['summary']['abstention']['n_answerable']}",
           "correct abstentions": f"{res[t]['summary']['abstention']['correct_abstentions']}/"
                                  f"{res[t]['summary']['abstention']['n_unanswerable']}",
           "MRR": round(res[t]["summary"]["mrr"], 3), "p95 latency (ms)": round(res[t]["summary"]["latency_ms"]["p95"], 1)}
          for t in chosen]
    st.dataframe(pd.DataFrame(ab), hide_index=True, use_container_width=True)


def page_experiments() -> None:
    st.markdown('<div class="cl-eyebrow">Self-improvement loop</div>', unsafe_allow_html=True)
    st.title("Experiment history")
    log = vm.load_experiment_log()
    if not log:
        banner("abstain", "No experiment log yet", " Run `python -m lextrace improve` to create experiments/experiment_log.jsonl.")
        return
    st.caption("Every iteration is evaluated on train only. Accepted changes become the parent of the next iteration.")
    skipped = [e for e in log if e["kind"] == "skipped"]
    log = [e for e in log if e["kind"] != "skipped"]
    rows = []
    for e in log:
        m = e.get("metrics_after") or e.get("metrics", {})
        rows.append({"iter": e["iteration"], "change": e.get("proposal_id", "baseline"),
                     "decision": e["decision"], "recall@4": m.get("recall@4"), "precision@4": m.get("precision@4"),
                     "MRR": m.get("mrr"), "correct abst.": m.get("correct_abstentions"),
                     "false abst. rate": m.get("false_abstention_rate"), "why": e.get("explanation", "")})
    df = pd.DataFrame(rows)
    st.dataframe(df.round(3), hide_index=True, use_container_width=True,
                 column_config={"why": st.column_config.TextColumn(width="large")})
    if skipped:
        st.caption("Skipped without evaluation (no effect on the parent): "
                   + "; ".join(f"{e['proposal_id']} - {e['explanation']}" for e in skipped))
    pick = st.selectbox("Inspect iteration", [r["iter"] for r in rows][1:] or [0])
    entry = next(e for e in log if e["iteration"] == pick)
    c1, c2 = st.columns(2)
    c1.markdown("**Change**")
    c1.code(yaml.safe_dump(entry.get("change", {}), sort_keys=False) or "-", language="yaml")
    c1.markdown(f"**Reason tried:** {entry.get('reason', '-')}")
    c1.markdown(f"**Proposed by:** {entry.get('proposal_source', '-')}")
    c2.markdown("**Guardrails**")
    c2.dataframe(pd.DataFrame([{"guardrail": k, "pass": v} for k, v in entry.get("guardrails", {}).items()]),
                 hide_index=True, use_container_width=True)
    c2.markdown(f"**Decision:** {entry['decision']} - {entry.get('explanation', '')}")


def page_configuration() -> None:
    st.markdown('<div class="cl-eyebrow">Pipeline</div>', unsafe_allow_html=True)
    st.title("Configuration")
    name = st.session_state.get("config_name")
    if not name:
        banner("abstain", "No calibrated configuration", " Run the improvement loop first.")
        return
    cfg = load_config(name)
    st.write(f"**Active:** `{name}` - {cfg.description}")
    st.markdown(" ".join(f'<span class="cl-chip">{html.escape(x)}</span>' for x in [
        f"chunks: {cfg.chunking.strategy} <= {cfg.chunking.size}", f"routing: {cfg.routing.mode}",
        f"retrieval: {cfg.retrieval.method}", f"rerank: {'on' if cfg.rerank.enabled else 'off'}",
        f"expansion: {cfg.retrieval.expansion}", f"fingerprint {cfg.fingerprint()}"]), unsafe_allow_html=True)
    st.subheader("Abstention gates")
    st.dataframe(pd.DataFrame([{"signal": g.signal, "threshold": g.threshold} for g in cfg.abstention.gates]),
                 hide_index=True, use_container_width=True)
    st.caption(f"Thresholds were calibrated on train so that at most {cfg.abstention.max_false_abstention:.0%} of "
               "train answerable questions are refused.")
    st.subheader("Full configuration")
    st.code(yaml.safe_dump(cfg.to_dict(), sort_keys=False), language="yaml")
    manifest = Path(__file__).resolve().parent / "data" / "dataset_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        st.subheader("Dataset")
        st.write(f"{m['n_queries']} benchmark queries over {m['n_documents']} contracts from {m['source']} "
                 f"@ `{m['revision'][:10]}`; {m['checksums_verified']} file checksums verified.")


# ------------------------------------------------------------------ shell

st.logo(str(Path(__file__).resolve().parent / "assets" / "lextrace_logo.svg"), size="large")
with st.sidebar:
    st.markdown('<div class="cl-sub">Contract Intelligence &middot; evidence retrieval</div>', unsafe_allow_html=True)
    configs = vm.usable_configs()
    if configs:
        default = configs.index("final") if "final" in configs else 0
        st.selectbox("Active configuration", configs, index=default, key="config_name",
                     help="Only configurations with calibrated thresholds are listed.")
    else:
        st.session_state["config_name"] = None
        st.caption("No calibrated configuration yet.")

nav = st.navigation([
    st.Page(page_search, title="Search", icon=":material/search:", default=True),
    st.Page(page_evaluation, title="Evaluation", icon=":material/monitoring:"),
    st.Page(page_experiments, title="Experiments", icon=":material/science:"),
    st.Page(page_configuration, title="Configuration", icon=":material/tune:"),
])
with st.sidebar:
    st.markdown(f'<p class="cl-note" style="margin-top:24px">{html.escape(DISCLAIMER)}</p>', unsafe_allow_html=True)
nav.run()

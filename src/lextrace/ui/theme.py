"""LexTrace design system: one stylesheet injected into Streamlit.

Tokens: navy ink for structure, teal for interactive/positive evidence, amber for abstention, neutral greys for
text and borders. One type scale (12/14/16/20/28), 4-px spacing grid, 8-px radius, visible 2-px focus rings."""

CSS = """
<style>
:root {
  --cl-navy-900: #0b1f33; --cl-navy-700: #17324d; --cl-navy-500: #2c4c6b;
  --cl-teal-600: #0f7c80; --cl-teal-500: #149a9e; --cl-teal-50: #e6f4f4;
  --cl-amber-600: #a15c07; --cl-amber-50: #fdf3e2;
  --cl-red-600: #b42318; --cl-red-50: #fdecea;
  --cl-ink: #1d2733; --cl-muted: #5b6878; --cl-line: #dfe4ea; --cl-surface: #ffffff; --cl-canvas: #f5f7f9;
  --cl-radius: 8px; --cl-focus: 0 0 0 2px #ffffff, 0 0 0 4px var(--cl-teal-500);
}
.stApp { background: var(--cl-canvas); color: var(--cl-ink); }
[data-testid="stSidebar"] { background: var(--cl-navy-900); }
[data-testid="stSidebar"] * { color: #dbe6f0 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: var(--cl-ink) !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: var(--cl-navy-700); border-left: 3px solid var(--cl-teal-500); }
h1, h2, h3 { color: var(--cl-navy-900); letter-spacing: -0.01em; }
a:focus-visible, button:focus-visible, textarea:focus-visible, input:focus-visible, [role="tab"]:focus-visible,
[role="radio"]:focus-visible, summary:focus-visible { outline: none !important; box-shadow: var(--cl-focus) !important; }
.stButton > button[kind="primary"] { background: var(--cl-teal-600); border-color: var(--cl-teal-600); }
.stButton > button[kind="primary"]:hover { background: var(--cl-navy-700); border-color: var(--cl-navy-700); }
.cl-brand { font-weight: 700; font-size: 20px; color: #ffffff !important; letter-spacing: 0.02em; }
.cl-brand span { color: var(--cl-teal-500) !important; }
.cl-sub { font-size: 12px; color: #9fb3c8 !important; margin-top: -4px; }
.cl-eyebrow { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--cl-muted); margin-bottom: 4px; }
.cl-card { background: var(--cl-surface); border: 1px solid var(--cl-line); border-radius: var(--cl-radius);
  padding: 16px 20px; margin: 0 0 12px 0; }
.cl-card.selected { border-color: var(--cl-teal-500); box-shadow: inset 3px 0 0 var(--cl-teal-500); }
.cl-card-head { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; margin-bottom: 8px; }
.cl-rank { background: var(--cl-navy-900); color: #fff; border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 700; }
.cl-contract { font-weight: 600; color: var(--cl-navy-700); font-size: 14px; }
.cl-meta { font-size: 12px; color: var(--cl-muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.cl-excerpt { font-size: 14px; line-height: 1.6; white-space: pre-wrap; color: var(--cl-ink); }
.cl-bar { height: 4px; background: var(--cl-line); border-radius: 2px; margin-top: 10px; }
.cl-bar > div { height: 4px; background: var(--cl-teal-500); border-radius: 2px; }
.cl-banner { border-radius: var(--cl-radius); padding: 14px 18px; margin: 8px 0 16px 0; font-size: 14px; }
.cl-banner.abstain { background: var(--cl-amber-50); border: 1px solid #f1d39b; color: #5c3a06; }
.cl-banner.ok { background: var(--cl-teal-50); border: 1px solid #b5dfe0; color: #0b4f52; }
.cl-banner.err { background: var(--cl-red-50); border: 1px solid #f5b8b2; color: #7a1a12; }
.cl-banner b { display: block; font-size: 16px; margin-bottom: 2px; }
.cl-ctx { color: var(--cl-muted); }
mark.cl-hit { background: #d6efef; color: var(--cl-ink); padding: 1px 0; border-bottom: 2px solid var(--cl-teal-500); }
.cl-doc { background: var(--cl-surface); border: 1px solid var(--cl-line); border-radius: var(--cl-radius);
  padding: 16px; font-size: 13px; line-height: 1.65; max-height: 520px; overflow-y: auto; }
.cl-kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 12px; font-size: 13px; }
.cl-kv dt { color: var(--cl-muted); } .cl-kv dd { margin: 0; font-family: ui-monospace, Consolas, monospace; overflow-wrap: anywhere; }
.cl-chip { display: inline-block; border: 1px solid var(--cl-line); background: var(--cl-surface); border-radius: 999px;
  padding: 2px 10px; font-size: 12px; margin: 0 6px 6px 0; color: var(--cl-navy-700); }
.cl-note { font-size: 12px; color: var(--cl-muted); }
@media (max-width: 640px) {
  .cl-card { padding: 12px 14px; } .cl-excerpt { font-size: 13px; } h1 { font-size: 24px !important; }
}
</style>
"""

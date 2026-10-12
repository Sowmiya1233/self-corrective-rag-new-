"""Streamlit front end:  streamlit run app.py"""
import html
import json
import math

import streamlit as st

import config
from src import metrics
from src.pipeline import SelfCorrectiveRAG
from src.retriever import Retriever

st.set_page_config(page_title="Self-Corrective RAG", page_icon="🔍", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, .stApp, [class*="css"] {font-family: 'Inter', sans-serif;}
.stApp {
  background:
    radial-gradient(1100px 560px at 8% -10%, #1d2f6b 0%, transparent 60%),
    radial-gradient(900px 520px at 100% 0%, #4b2a8a 0%, transparent 55%),
    #0b1020;
  color: #e8ecf8;
}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.08);}
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.stApp, .stApp label, [data-testid="stWidgetLabel"] p, .stMarkdown p, .stMarkdown li, .stCaption, [data-testid="stSidebar"] p {color: #e8ecf8;}
.stTextInput input, div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.07) !important; color: #fff !important;
  border: 1px solid rgba(255,255,255,0.18) !important; border-radius: 12px !important;
}
button[data-baseweb="tab"] p {color: #b9c3e6; font-weight: 600;}
button[data-baseweb="tab"][aria-selected="true"] p {color: #ffffff;}
[data-testid="stExpander"] {background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.12); border-radius: 14px;}
[data-testid="stExpander"] summary p {color: #e8ecf8;}

.hero {
  background: linear-gradient(120deg, #2a4fd8, #7a4fd0, #e0457b, #2a4fd8);
  background-size: 300% 300%; animation: shift 14s ease infinite;
  border-radius: 22px; padding: 30px 36px; margin-bottom: 18px;
  box-shadow: 0 14px 40px rgba(122,79,208,0.35);
}
@keyframes shift {0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;}}
.hero h1 {margin: 0; font-size: 2.3rem; font-weight: 800; color: #fff;}
.hero p {margin: 8px 0 0 0; font-size: 1.05rem; color: rgba(255,255,255,0.92);}
.tag {display: inline-block; margin: 14px 8px 0 0; padding: 5px 13px; border-radius: 999px; font-size: 0.78rem;
  background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.25); color: #fff;}

.glass {
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.13);
  border-radius: 20px; padding: 22px 26px; margin-bottom: 16px; backdrop-filter: blur(8px);
}
.answer {border-left: 8px solid var(--accent); box-shadow: 0 0 40px -12px var(--accent);}
.badge {display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 800; font-size: 0.82rem; color: #fff; background: var(--accent);}
.answer-text {font-size: 2.1rem; font-weight: 800; line-height: 1.2; margin: 14px 0 8px 0; color: #fff;}
.explain {font-size: 1rem; line-height: 1.55; color: #c9d2ee;}
.gold {display: inline-block; margin-top: 14px; padding: 6px 13px; border-radius: 10px; font-size: 0.86rem;
  background: rgba(59,111,212,0.18); border: 1px solid rgba(110,150,255,0.45); color: #dbe5ff;}

.gauge {text-align: center;}
.gauge .cap {font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #aab5da; margin-top: 4px;}

.mgrid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 16px;}
.mcard {background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.13); border-radius: 18px; padding: 16px 18px;}
.ml {font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #aab5da;}
.mv {font-size: 1.9rem; font-weight: 800; color: #fff; margin-top: 4px;}
.ms {font-size: 0.78rem; color: #8f9bc4; margin-top: 2px;}
.bar {height: 8px; border-radius: 99px; background: rgba(255,255,255,0.12); margin-top: 9px; overflow: hidden;}
.bar > div {height: 100%; border-radius: 99px; background: linear-gradient(90deg, #4f8cff, #38d39f);}

.pipe {display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 4px 0 8px 0;}
.pnode {padding: 10px 14px; border-radius: 14px; text-align: center; font-size: 0.82rem; font-weight: 600; color: #aab5da;
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12);}
.pnode.on {color: #fff; background: rgba(79,140,255,0.2); border-color: rgba(110,150,255,0.7); box-shadow: 0 0 22px -4px rgba(79,140,255,0.7);}
.pico {font-size: 1.15rem; margin-bottom: 2px;}
.parrow {color: #6f7bab; font-weight: 800;}
.loop {display: inline-block; margin: 4px 0 14px 0; padding: 5px 14px; border-radius: 999px; font-size: 0.82rem; font-weight: 700;
  color: #fff; background: linear-gradient(120deg, #7a4fd0, #e0457b);}

.iter {border-left: 3px solid rgba(255,255,255,0.18); margin-left: 8px; padding-left: 22px; position: relative; margin-bottom: 18px;}
.iter:before {content: ""; position: absolute; left: -9px; top: 6px; width: 15px; height: 15px; border-radius: 50%;
  background: var(--dot); box-shadow: 0 0 14px var(--dot);}
.ihead {display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px;}
.ititle {font-weight: 800; font-size: 1.05rem; color: #fff;}
.ichip {padding: 3px 11px; border-radius: 999px; font-size: 0.75rem; font-weight: 800; color: #fff; background: var(--dot);}
.ians {color: #c9d2ee; font-size: 0.92rem;}
.claim {display: flex; gap: 12px; align-items: flex-start; padding: 11px 14px; margin-bottom: 8px; border-radius: 14px;
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);}
.chip {flex: 0 0 auto; margin-top: 2px; padding: 3px 10px; border-radius: 999px; font-size: 0.68rem; font-weight: 800; letter-spacing: 0.05em; color: #fff;}
.chip.SUPPORTED {background: #2e9e5b;}
.chip.UNSUPPORTED {background: #d98e00;}
.chip.CONTRADICTED {background: #c94452;}
.ct {font-weight: 600; color: #fff;}
.cr {font-size: 0.84rem; color: #9aa6cf; margin-top: 2px;}
.claim details {margin-top: 6px; font-size: 0.82rem; color: #b9c3e6;}
.claim summary {cursor: pointer; color: #7fb0ff;}
.qs {margin: 8px 0 0 0; font-size: 0.88rem; color: #c9d2ee;}
.qs span {display: inline-block; margin: 3px 6px 3px 0; padding: 4px 11px; border-radius: 10px;
  background: rgba(122,79,208,0.22); border: 1px solid rgba(160,120,255,0.5);}

.pass {display: flex; align-items: center; gap: 12px; margin-bottom: 7px;}
.pname {flex: 0 0 260px; font-size: 0.86rem; color: #e8ecf8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.pbar {flex: 1; height: 9px; border-radius: 99px; background: rgba(255,255,255,0.1); overflow: hidden;}
.pbar > div {height: 100%; background: linear-gradient(90deg, #7a4fd0, #4f8cff);}
.pscore {flex: 0 0 44px; font-size: 0.82rem; color: #aab5da; text-align: right;}

.cmp {margin-bottom: 18px;}
.cmp-t {font-weight: 700; color: #fff; margin-bottom: 6px;}
.row {display: flex; align-items: center; gap: 10px; margin: 4px 0;}
.rl {flex: 0 0 120px; font-size: 0.8rem; color: #aab5da;}
.rb {flex: 1; height: 14px; border-radius: 99px; background: rgba(255,255,255,0.1); overflow: hidden;}
.rb > div {height: 100%; border-radius: 99px;}
.rb.base > div {background: #8892b8;}
.rb.sc > div {background: linear-gradient(90deg, #4f8cff, #38d39f);}
.rv {flex: 0 0 56px; font-size: 0.84rem; font-weight: 700; color: #fff; text-align: right;}

div.stButton > button {border-radius: 12px; font-weight: 800; padding: 0.6rem 1.8rem; border: none; color: #fff;
  background: linear-gradient(120deg, #4f8cff, #7a4fd0, #e0457b); box-shadow: 0 8px 22px -6px rgba(122,79,208,0.8);}
div.stButton > button:hover {filter: brightness(1.12); color: #fff;}
.foot {text-align: center; color: #7f8bb5; font-size: 0.8rem; margin-top: 26px;}
@media (max-width: 800px) {.mgrid {grid-template-columns: 1fr;} .pname {flex-basis: 140px;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STATUS = {
    "verified": ("✅ Verified", "#2e9e5b"),
    "low_confidence": ("⚠️ Low confidence", "#e0a21a"),
    "abstained": ("🛑 Abstained", "#e0455b"),
}


def esc(x):
    return html.escape(str(x))


@st.cache_resource
def load_rag():
    r = Retriever()
    return SelfCorrectiveRAG(r), r


def pipeline_html(res=None):
    nodes = [("❓", "Question"), ("🔎", "Retrieve"), ("✍️", "Generate"),
             ("🧩", "Split claims"), ("🛡️", "Verify"), ("🏁", "Answer")]
    on = "on" if res else ""
    parts = []
    for i, (ic, t) in enumerate(nodes):
        parts.append(f'<div class="pnode {on}"><div class="pico">{ic}</div><div>{t}</div></div>')
        if i < len(nodes) - 1:
            parts.append('<div class="parrow">→</div>')
    loop = ""
    if res and len(res["iterations"]) > 1:
        n = len(res["iterations"]) - 1
        loop = f'<div class="loop">↻ {n} self-correction round{"s" if n > 1 else ""} triggered</div>'
    return '<div class="pipe">' + "".join(parts) + "</div>" + loop


def gauge_html(value, threshold, color):
    r, c = 56, 2 * math.pi * 56
    dash = max(0.0, min(1.0, value)) * c
    ang = math.radians(threshold * 360 - 90)
    x1, y1 = 70 + (r - 10) * math.cos(ang), 70 + (r - 10) * math.sin(ang)
    x2, y2 = 70 + (r + 10) * math.cos(ang), 70 + (r + 10) * math.sin(ang)
    return (
        '<div class="glass gauge">'
        '<svg viewBox="0 0 140 140" width="170" height="170">'
        '<circle cx="70" cy="70" r="56" fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="12"/>'
        f'<circle cx="70" cy="70" r="56" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round" '
        f'stroke-dasharray="{dash:.1f} {c:.1f}" transform="rotate(-90 70 70)"/>'
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#ffffff" stroke-width="3"/>'
        f'<text x="70" y="76" text-anchor="middle" font-size="26" font-weight="800" fill="#ffffff">{value:.0%}</text>'
        "</svg>"
        '<div class="cap">Faithfulness</div>'
        f'<div class="cap">pass line (white tick): {threshold:.0%}</div>'
        "</div>"
    )


def mcard(label, value, sub="", bar=None):
    b = f'<div class="bar"><div style="width:{bar}%"></div></div>' if bar is not None else ""
    return (f'<div class="mcard"><div class="ml">{label}</div><div class="mv">{value}</div>'
            f'<div class="ms">{sub}</div>{b}</div>')


def iteration_html(it, is_last):
    dot = "#2e9e5b" if it["passed"] else "#e0455b"
    claims = ""
    for c in it["claims"]:
        ev = "".join(f"<li>{esc(e)}</li>" for e in c["evidence"])
        claims += (
            f'<div class="claim"><span class="chip {c["verdict"]}">{c["verdict"]}</span>'
            f'<div><div class="ct">{esc(c["claim"])}</div><div class="cr">{esc(c["reason"])}</div>'
            f'<details><summary>Evidence used</summary><ul>{ev}</ul></details></div></div>'
        )
    if not it["claims"]:
        claims = '<div class="cr">The model said the context was insufficient, so there were no claims to verify.</div>'
    qs = ""
    if it["queries"]:
        qs = '<div class="qs">🔎 Follow-up queries: ' + "".join(f"<span>{esc(q)}</span>" for q in it["queries"]) + "</div>"
    return (
        f'<div class="iter" style="--dot:{dot}">'
        f'<div class="ihead"><span class="ititle">Iteration {it["iteration"]}</span>'
        f'<span class="ichip">{it["faithfulness"]:.0%} supported · {"PASSED" if it["passed"] else "NOT ENOUGH"}</span>'
        f'<span class="ians">answer: <b>{esc(it["answer"])}</b></span></div>'
        f"{claims}{qs}</div>"
    )


def cmp_html(title, base, sc, fmt="{:.0%}", scale=1.0):
    def row(cls, name, v):
        w = max(0.0, min(100.0, v / scale * 100))
        return (f'<div class="row"><div class="rl">{name}</div><div class="rb {cls}"><div style="width:{w:.0f}%"></div></div>'
                f'<div class="rv">{fmt.format(v)}</div></div>')
    return f'<div class="cmp"><div class="cmp-t">{title}</div>{row("base", "Plain RAG", base)}{row("sc", "Self-Corrective", sc)}</div>'


def load_eval():
    p = config.RESULTS_DIR / "eval_summary.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------- page
st.markdown(
    '<div class="hero"><h1>🔍 Self-Corrective RAG</h1>'
    "<p>Adaptive evidence verification for hallucination-resistant answer generation</p>"
    '<span class="tag">Claim-level verification</span><span class="tag">Self-correction loop</span>'
    '<span class="tag">Abstains when unsure</span><span class="tag">HotpotQA</span></div>',
    unsafe_allow_html=True,
)

try:
    rag, retriever = load_rag()
except FileNotFoundError:
    st.error("Index not found. In the terminal run:  python build_index.py")
    st.stop()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Initial passages (top-k)", 3, 10, config.TOP_K)
    max_iters = st.slider("Max correction rounds", 0, 4, config.MAX_ITERATIONS)
    base_thr = st.slider("Base faithfulness threshold", 0.5, 1.0, config.BASE_THRESHOLD, 0.05)
    abstain = st.slider("Abstain below", 0.0, 0.9, config.ABSTAIN_BELOW, 0.05)
    st.divider()
    st.caption(f"🤖 Model: {config.GROQ_MODEL}")
    st.caption(f"📚 Corpus: {len(retriever.docs):,} paragraphs")

tab_demo, tab_eval, tab_how = st.tabs(["🚀 Live demo", "📊 Evaluation", "🧭 How it works"])

# ---------------------------------------------------------------- live demo
with tab_demo:
    examples = [q["question"] for q in retriever.questions[:20]]
    choice = st.selectbox("💡 Pick a HotpotQA example (or type your own below)", [""] + examples)
    question = st.text_input("❓ Your question", value=choice, placeholder="Type a question here...")
    ask = st.button("Ask", type="primary")

    if ask and question.strip():
        with st.spinner("Retrieving → generating → verifying → correcting..."):
            st.session_state["res"] = rag.run(
                question.strip(), top_k=top_k, max_iterations=max_iters,
                base_threshold=base_thr, abstain_below=abstain,
            )
            st.session_state["question"] = question.strip()

    res = st.session_state.get("res")
    st.markdown(pipeline_html(res), unsafe_allow_html=True)

    if not res:
        st.markdown('<div class="glass">👆 Pick an example or type a question, then press <b>Ask</b>.</div>',
                    unsafe_allow_html=True)
    else:
        label, accent = STATUS[res["status"]]
        gold = next((q["answer"] for q in retriever.questions if q["question"] == st.session_state["question"]), None)
        gold_html = ""
        if gold:
            ok = res["status"] != "abstained" and metrics.f1_score(res["final_answer"], gold) >= 0.5
            gold_html = f'<div class="gold">{"✔️" if ok else "ℹ️"} Gold answer: <b>{esc(gold)}</b></div>'
        expl = f'<div class="explain">{esc(res["final_explanation"])}</div>' if res["final_explanation"] else ""

        c1, c2 = st.columns([2.2, 1])
        with c1:
            st.markdown(
                f'<div class="glass answer" style="--accent:{accent}"><span class="badge">{label}</span>'
                f'<div class="answer-text">{esc(res["final_answer"])}</div>{expl}{gold_html}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(gauge_html(res["faithfulness"], res["threshold"], accent), unsafe_allow_html=True)

        st.markdown(
            '<div class="mgrid">'
            + mcard("Adaptive threshold", f'{res["threshold"]:.0%}', "support needed to pass", int(res["threshold"] * 100))
            + mcard("Retrieval confidence", f'{res["retrieval_confidence"]:.2f}',
                    f'checks {res["evidence_depth"]} evidence sentences per claim', int(min(1.0, res["retrieval_confidence"]) * 100))
            + mcard("LLM calls / time", f'{res["llm_calls"]} / {res["latency"]:.1f}s', f'{len(res["iterations"])} iteration(s)')
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### 🧪 Self-correction timeline")
        st.markdown("".join(iteration_html(it, it is res["iterations"][-1]) for it in res["iterations"]),
                    unsafe_allow_html=True)

        with st.expander("📄 Retrieved passages (similarity scores)"):
            for it in res["iterations"]:
                st.markdown(f"**Iteration {it['iteration']}** — {it['num_passages']} passages in context")
                rows = "".join(
                    f'<div class="pass"><div class="pname">{esc(p["title"])}</div>'
                    f'<div class="pbar"><div style="width:{max(0, min(1, p["score"])) * 100:.0f}%"></div></div>'
                    f'<div class="pscore">{p["score"]:.2f}</div></div>'
                    for p in it["passages"]
                )
                st.markdown(rows, unsafe_allow_html=True)

# ---------------------------------------------------------------- evaluation
with tab_eval:
    ev = load_eval()
    if not ev:
        st.info("No evaluation results yet. Run:  python run_eval.py --n 60")
    else:
        b, s = ev.get("baseline", {}), ev.get("self_corrective", {})
        st.markdown(f"### Plain RAG vs Self-Corrective RAG — {ev.get('n_questions', '?')} HotpotQA questions")
        left, right = st.columns(2)
        with left:
            st.markdown(
                '<div class="glass">'
                + cmp_html("Exact match (all questions)", b.get("EM_all", 0), s.get("EM_all", 0))
                + cmp_html("F1 score (all questions)", b.get("F1_all", 0), s.get("F1_all", 0))
                + cmp_html("Supported claims (answered only)", b.get("avg_faithfulness_on_answered", 0),
                           s.get("avg_faithfulness_on_answered", 0))
                + "</div>",
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                '<div class="glass">'
                + cmp_html("Refused to answer", b.get("abstain_rate", 0), s.get("abstain_rate", 0))
                + cmp_html("Unfaithful answers", b.get("unfaithful_answer_rate", 0), s.get("unfaithful_answer_rate", 0))
                + cmp_html("Exact match (answered only)", b.get("EM_on_answered", 0), s.get("EM_on_answered", 0))
                + "</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="mgrid">'
            + mcard("Fully verified answers", f'{s.get("verified_rate", 0):.0%}', "self-corrective system")
            + mcard("LLM calls per question", f'{s.get("avg_llm_calls", 0):.2f}', "plain RAG is about 3")
            + mcard("Latency per question", f'{s.get("avg_latency_s", 0):.1f}s', "self-corrective system")
            + "</div>",
            unsafe_allow_html=True,
        )
        if ev.get("stopped_early"):
            st.caption("Note: this run stopped early because of the Groq daily token limit.")

# ---------------------------------------------------------------- how it works
with tab_how:
    st.markdown(pipeline_html(None), unsafe_allow_html=True)
    st.markdown(
        '<div class="glass"><b>1. Retrieve</b> – hybrid search (local embeddings + title-match boost) finds the best paragraphs.<br>'
        "<b>2. Generate</b> – the LLM answers using only that context.<br>"
        "<b>3. Split</b> – the explanation becomes small factual claims.<br>"
        "<b>4. Verify</b> – each claim is checked against its best evidence sentences.<br>"
        "<b>5. Correct</b> – if too few claims are supported, follow-up queries fetch more evidence and the answer is rewritten.<br>"
        "<b>6. Decide</b> – return a verified answer, a low-confidence answer, or abstain.<br><br>"
        "The pass threshold and the evidence depth adapt to how good the retrieval was.</div>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="foot">Self-Corrective RAG · Adaptive Evidence Verification</div>', unsafe_allow_html=True)
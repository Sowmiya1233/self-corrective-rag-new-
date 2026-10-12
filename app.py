"""Streamlit front end:  streamlit run app.py"""
import pandas as pd
import streamlit as st

import config
from src.pipeline import SelfCorrectiveRAG
from src.retriever import Retriever

st.set_page_config(page_title="Self-Corrective RAG", page_icon="🔍", layout="wide")


@st.cache_resource
def load_rag():
    r = Retriever()
    return SelfCorrectiveRAG(r), r


st.title("🔍 Self-Corrective RAG")
st.caption("Adaptive evidence verification for hallucination-resistant answer generation")

try:
    rag, retriever = load_rag()
except FileNotFoundError:
    st.error("Index not found. In the terminal run:  python build_index.py")
    st.stop()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Initial passages (top-k)", 3, 10, config.TOP_K)
    max_iters = st.slider("Max correction rounds", 0, 4, config.MAX_ITERATIONS)
    base_thr = st.slider("Base faithfulness threshold", 0.5, 1.0, config.BASE_THRESHOLD, 0.05)
    abstain = st.slider("Abstain below", 0.0, 0.9, config.ABSTAIN_BELOW, 0.05)
    st.caption(f"Model: {config.GROQ_MODEL}\n\nCorpus: {len(retriever.docs)} paragraphs")

examples = [q["question"] for q in retriever.questions[:20]]
choice = st.selectbox("Pick a HotpotQA question (or type your own below)", [""] + examples)
question = st.text_input("Question", value=choice)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving, generating, verifying, correcting..."):
        res = rag.run(question.strip(), top_k=top_k, max_iterations=max_iters,
                      base_threshold=base_thr, abstain_below=abstain)

    badge = {"verified": "✅ Verified", "low_confidence": "⚠️ Low confidence", "abstained": "🛑 Abstained"}[res["status"]]
    st.subheader(f"{badge}")
    st.markdown(f"### {res['final_answer']}")
    if res["final_explanation"]:
        st.write(res["final_explanation"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faithfulness", f"{res['faithfulness']:.0%}")
    c2.metric("Adaptive threshold", f"{res['threshold']:.0%}")
    c3.metric("Retrieval confidence", f"{res['retrieval_confidence']:.2f}")
    c4.metric("LLM calls / time", f"{res['llm_calls']} / {res['latency']:.1f}s")

    gold = next((q["answer"] for q in retriever.questions if q["question"] == question.strip()), None)
    if gold:
        st.info(f"Gold answer: **{gold}**")

    st.divider()
    st.subheader("Self-correction trace")
    for it in res["iterations"]:
        label = f"Iteration {it['iteration']} — answer: {it['answer']} — faithfulness {it['faithfulness']:.0%} " \
                f"{'✅' if it['passed'] else '❌'}"
        with st.expander(label, expanded=(it is res["iterations"][-1])):
            if it["claims"]:
                df = pd.DataFrame([{"Verdict": c["verdict"], "Claim": c["claim"], "Reason": c["reason"]}
                                   for c in it["claims"]])
                st.dataframe(df, use_container_width=True, hide_index=True)
                with st.popover("Evidence used"):
                    for c in it["claims"]:
                        st.markdown(f"**{c['claim']}**")
                        for e in c["evidence"]:
                            st.caption(f"• {e}")
            else:
                st.write("Model abstained: context was insufficient.")
            if it["queries"]:
                st.write("Follow-up queries issued:", it["queries"])
            st.caption(f"{it['num_passages']} passages in context: " +
                       ", ".join(f"{p['title']} ({p['score']:.2f})" for p in it["passages"]))

import streamlit as st
import os

# Bridge Streamlit secrets -> environment variables so the pipeline's
# os.getenv(...) calls find them. Must run BEFORE importing the pipeline.
try:
    for key in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass  

st.set_page_config(page_title="FinIntel AI", layout="centered")

@st.cache_resource(show_spinner="Loading models and index (first load ~1 min)...")
def load_pipeline():
    from src.agents.graph import run
    from src.features.qa_chain import QAChain
    clarifier = QAChain()
    return run, clarifier


run, clarifier = load_pipeline()


st.title(" FinIntel AI")
st.caption(
    "Financial Q&A over TCS, Infosys, and Wipro annual reports, investor "
    "presentations, and earnings calls (FY24–FY26). "
    "Answers are grounded strictly in the source filings."
)

with st.expander("What can I ask?  (this is a scoped demo)"):
    st.markdown(
        "- **Scope:** TCS, Infosys, Wipro only, FY24–FY26 only.\n"
        "- **Try:** *What was TCS revenue in FY26?* · "
        "*Compare TCS and Infosys net profit in FY26.* · "
        "*What was Infosys operating margin in Q2 FY26?*\n"
        "- The system will **ask you to clarify** if you omit the company or year.\n"
        "- It **refuses** questions outside its data instead of guessing."
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def _dedupe_sources(metadata):
    seen, out = set(), []
    for m in metadata or []:
        key = (m.get("source"), m.get("page"))
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


prompt = st.chat_input("Ask about TCS, Infosys, or Wipro financials...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        resolution = clarifier.resolve(prompt)

        if "query" not in resolution:
            msg = resolution.get("question", "Could you rephrase that?")
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            resolved_query = resolution["query"]
            companies = resolution.get("companies", [])

            with st.spinner("Searching filings and verifying..."):
                start = time.perf_counter()
                result = run(resolved_query, companies=companies)
                latency_ms = (time.perf_counter() - start) * 1000

            answer = result.get("answer", "")
            st.markdown(answer)

            detected = result.get("companies") or []
            if detected:
                clarifier.memory["companies"] = list(detected)

            verification = result.get("verification") or {}
            grounded = verification.get("grounded")
            confidence = verification.get("confidence")

            cols = st.columns(3)
            if confidence is not None:
                cols[0].metric("Confidence", f"{confidence*100:.0f}%")
            cols[1].metric("Grounded", "Yes" if grounded else "No")
            cols[2].metric("Latency", f"{latency_ms/1000:.1f}s")

            sources = _dedupe_sources(result.get("metadata"))
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for s in sources:
                        st.markdown(f"- `{s.get('source')}` — page {s.get('page')}")

            st.session_state.messages.append({"role": "assistant", "content": answer})


st.divider()
st.caption(
    "Demo project — retrieval-augmented generation with a verification agent and "
    "a 52-question evaluation harness (87% accuracy). Scoped to 3 companies / 3 years."
)
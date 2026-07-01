import time

from dotenv import load_dotenv
from groq import Groq, GroqError
from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.evidence_agent import run_evidence_agent
from src.agents.consistency_agent import run_consistency_agent, should_run_consistency
from src.agents.verification_agent import run_verification_agent, check_grounding
from src.features.qa_chain import QAChain

load_dotenv()

_client = Groq()
_qa = QAChain.__new__(QAChain)
_qa.client = _client


def _route_after_evidence(state: AgentState) -> str:
    if should_run_consistency(state):
        return "consistency_agent"
    return "generate_answer"


def run_generate_answer(state: AgentState) -> AgentState:
    query = state["query"]
    evidence = state.get("evidence", {})
    context = evidence.get("context", "")

    consistency = evidence.get("consistency")
    if consistency:
        consistency_prefix = "\n\n".join(
            f"[Cross-source check | {r['company']} | {r['metric']} | {r['period']}]\n"
            f"Verdict: {r['verdict']}\n"
            f"Summary: {r['summary']}"
            for r in consistency
        )
        context = consistency_prefix + "\n\n---\n\n" + context

    prompt = _qa._build_prompt(query, context)

    answer = None
    for attempt in range(3):
        try:
            response = _client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            answer = response.choices[0].message.content
            break
        except GroqError as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                answer = (
                    f"Sorry, I couldn't reach the AI service right now "
                    f"({type(e).__name__}). Please try again in a moment."
                )

    return {
        **state,
        "draft_answer": answer,
    }


def run_finalize(state: AgentState) -> AgentState:
    draft_answer = state.get("draft_answer", "")
    evidence = state.get("evidence", {})
    verification = state.get("verification", {})

    metadata = evidence.get("metadata", [])
    consistency = evidence.get("consistency")

    seen = set()
    citations = []
    for meta in metadata:
        source = meta.get("source", "unknown")
        page = meta.get("page")
        key = f"{source}_{page}"
        if key not in seen:
            seen.add(key)
            citations.append(f"{source} (page {page})" if page else source)

    citation_block = (
        "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)
        if citations else ""
    )

    consistency_block = ""
    if consistency:
        verdicts = [
            f"  [{r['company']} | {r['metric']} | {r['period']}]: {r['verdict']}"
            for r in consistency
        ]
        consistency_block = "\n\nConsistency Check:\n" + "\n".join(verdicts)

    confidence = verification.get("confidence")
    confidence_block = (
        f"\n\n[Confidence: {round(confidence * 100)}%]"
        if confidence is not None else ""
    )

    unverified = verification.get("unverified_claims", [])
    unverified_block = ""
    if unverified:
        unverified_block = "\n\n[Unverified claims:]\n" + "\n".join(
            f"  - {c}" for c in unverified
        )

    reason = verification.get("reason", "")
    if reason and not verification.get("grounded"):
        unverified_block += f"\n  Note: {reason}"

    return {
        **state,
        "final_answer": (
            draft_answer
            + citation_block
            + consistency_block
            + confidence_block
            + unverified_block
        ),
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("evidence_agent", run_evidence_agent)
    graph.add_node("consistency_agent", run_consistency_agent)
    graph.add_node("generate_answer", run_generate_answer)
    graph.add_node("verification_agent", run_verification_agent)
    graph.add_node("finalize", run_finalize)

    graph.set_entry_point("evidence_agent")

    graph.add_conditional_edges(
        "evidence_agent",
        _route_after_evidence,
        {
            "consistency_agent": "consistency_agent",
            "generate_answer": "generate_answer",
        },
    )

    graph.add_edge("consistency_agent", "generate_answer")
    graph.add_edge("generate_answer", "verification_agent")

    graph.add_conditional_edges(
        "verification_agent",
        check_grounding,
        {
            "grounded": "finalize",
            "retry": "evidence_agent",
        },
    )

    graph.add_edge("finalize", END)

    return graph.compile()


pipeline = build_graph()


def run(query: str, companies: list = None) -> dict:
    initial_state: AgentState = {
        "query": query,
        "evidence": None,
        "draft_answer": None,
        "verification": None,
        "final_answer": None,
        "retry_count": 0,
        "companies": companies or [],
        "metrics": [],
        "topics": [],
        "years": [],
        "quarters": [],
        "is_comparison": False,
        "needs_cross_source": False,
    }

    result = pipeline.invoke(initial_state)

    return {
        "ok": True,
        "query": result["query"],
        "answer": result.get("final_answer", ""),
        "verification": result.get("verification", {}),
        "subqueries": result.get("evidence", {}).get("subqueries", []),
        "citations": result.get("evidence", {}).get("citations", []),
        "consistency": result.get("evidence", {}).get("consistency"),
    }
import time
import os
from dotenv import load_dotenv
from src.llm.client import client
from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.evidence_agent import run_evidence_agent
from src.agents.verification_agent import run_verification_agent, check_grounding
from src.agents.prompt import _build_prompt
from src.llm.usage import record_usage

load_dotenv()

_client = client

def run_generate_answer(state: AgentState) -> AgentState:
    query = state["query"]
    evidence = state.get("evidence") or {}
    context = evidence.get("context", "")

    prompt = _build_prompt(query, context)

    answer = None
    for attempt in range(3):
        try:
            response = _client.responses.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                input=prompt,
            )
            u = getattr(response, "usage", None)
            if u is not None:
                record_usage(getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))

            
            answer = response.output_text
            break
        except Exception as e:
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
    draft_answer = state.get("draft_answer") or ""
    evidence = state.get("evidence") or {}
    verification = state.get("verification") or {}

    metadata = evidence.get("metadata", [])

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

    try:
        confidence = float(verification.get("confidence"))
    except (TypeError, ValueError):
        confidence = None

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
            + confidence_block
            + unverified_block
        ),
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("evidence_agent", run_evidence_agent)
    graph.add_node("generate_answer", run_generate_answer)
    graph.add_node("verification_agent", run_verification_agent)
    graph.add_node("finalize", run_finalize)

    graph.set_entry_point("evidence_agent")

    graph.add_edge("evidence_agent", "generate_answer")
    graph.add_edge("generate_answer", "verification_agent")

    graph.add_conditional_edges(
        "verification_agent",
        check_grounding,
        {
            "grounded": "finalize",
            "give_up": "finalize",
            "retry": "evidence_agent",
        },
    )

    graph.add_edge("finalize", END)

    return graph.compile()


pipeline = build_graph()


def run(query: str, companies: list = None):
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
    }

    result = pipeline.invoke(initial_state)

    return {
        "ok": True,
        "metadata": (result.get("evidence") or {}).get("metadata", []),
        "query": result["query"],
        "answer": result.get("final_answer", ""),
        "verification": result.get("verification", {}),
        "subqueries": (result.get("evidence") or {}).get("subqueries", []),
        "companies": result.get("companies", []),
        "retry_count": result.get("retry_count", 0),
    }
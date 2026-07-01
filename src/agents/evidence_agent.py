from src.agents.state import AgentState
from src.features.rag import RAGPipeline
from src.retrieval.query_decomposer import QueryDecomposer

_rag = RAGPipeline()
_decomposer = QueryDecomposer()


def run_evidence_agent(state: AgentState) -> AgentState:
    query = state["query"]
    retry_count = state.get("retry_count", 0)

    companies = state.get("companies") or []
    metrics = state.get("metrics") or []
    topics = state.get("topics") or []
    years = state.get("years") or []
    quarters = state.get("quarters") or []
    is_comparison = state.get("is_comparison") or False

    if not companies and not metrics:
        decomposed = _decomposer.decompose(query)
        companies = decomposed.get("companies", [])
        metrics = decomposed.get("metrics", [])
        topics = decomposed.get("topics", [])
        years = decomposed.get("years", [])
        quarters = decomposed.get("quarters", [])
        is_comparison = decomposed.get("is_comparison", False)

    result = _rag.get_context(query, memory_companies=companies or None)

    evidence = {
        "documents": result["documents"],
        "metadata": result["metadata"],
        "citations": [],
        "subqueries": result["subqueries"],
        "context": result["context"],
        "consistency": None,
    }

    return {
        **state,
        "evidence": evidence,
        "companies": companies,
        "metrics": metrics,
        "topics": topics,
        "years": years,
        "quarters": quarters,
        "is_comparison": is_comparison,
        "retry_count": retry_count + 1,
    }
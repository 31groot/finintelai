from typing import Optional, TypedDict


class Evidence(TypedDict):
    documents: list[str]
    metadata: list[dict]
    citations: list[str]
    context: str
    subqueries: list[str]
    consistency: Optional[list]


class Verification(TypedDict):
    grounded: bool
    confidence: float
    unverified_claims: list[str]
    reason: Optional[str]


class AgentState(TypedDict):
    query: str
    evidence: Optional[Evidence]
    draft_answer: Optional[str]
    verification: Optional[Verification]
    final_answer: Optional[str]
    retry_count: int
    companies: list
    metrics: list
    topics: list
    years: list
    quarters: list
    is_comparison: bool
    needs_cross_source: bool
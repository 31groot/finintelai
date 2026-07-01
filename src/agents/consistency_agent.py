import json
import re
import time

from dotenv import load_dotenv
from groq import Groq, GroqError

from src.agents.state import AgentState

load_dotenv()

_client = Groq()

DOC_TYPES = ["investor_presentation", "earnings_call", "annual_report"]

DIFF_PROMPT = """
You are a financial analyst comparing what different source types say about the same metric.

You will be given three excerpts about the same company, metric, and time period — each from a different document type:
1. Investor Presentation
2. Earnings Call Transcript
3. Annual Report

Your job:
- Read all three excerpts carefully.
- Identify any differences in the numbers, percentages, or statements about this metric.
- Classify the result as one of:
  - "expected_evolution": Sources cover different periods or contexts. Differences are explained by natural progression (e.g. guidance in Q1, actuals in Q4).
  - "actual_inconsistency": Same period described but numbers or claims directly contradict each other with no clear business explanation.
  - "data_discrepancy": Same metric and period appear in multiple sources with different values.
  - "insufficient_data": One or more sources have no relevant content for this metric and period.

Return ONLY valid JSON in this exact format, nothing else:

{{
    "verdict": "expected_evolution" or "actual_inconsistency" or "data_discrepancy" or "insufficient_data",
    "differences": ["difference 1", "difference 2"],
    "summary": "One paragraph explaining what each source says and why the verdict was chosen."
}}

Company: {company}
Metric: {metric}
Period: {period}

Investor Presentation:
{presentation_context}

Earnings Call Transcript:
{earnings_context}

Annual Report:
{annual_context}
"""


def _retrieve_for_doc_type(
    retriever,
    reranker,
    company: str,
    metric: str,
    period: str,
    doc_type: str,
) -> str:
    query = f"{company} {metric} {period}"

    filters = {
        "companies": [company],
        "doc_types": [doc_type],
    }

    results = retriever.search(query=query, n_results=12, filters=filters)

    docs = results.get("documents", [])
    metadata = results.get("metadata", [])

    if not docs:
        return ""

    top_results = reranker.rerank(
        query=query,
        documents=docs,
        metadata=metadata,
        top_k=min(4, len(docs)),
    )

    return "\n\n".join(doc for doc, _meta, _score in top_results)


def _parse_diff_response(raw: str) -> dict:
    raw = raw.strip()
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {
        "verdict": "insufficient_data",
        "differences": ["Could not parse classifier response."],
        "summary": "Consistency check could not be completed due to a parsing error.",
    }


def _call_llm_with_retry(prompt: str, max_attempts: int = 3) -> dict:
    for attempt in range(max_attempts):
        try:
            response = _client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return _parse_diff_response(response.choices[0].message.content)
        except GroqError as e:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "verdict": "insufficient_data",
                    "differences": [],
                    "summary": f"Consistency check failed after {max_attempts} attempts ({type(e).__name__}).",
                }


def run_consistency_agent(state: AgentState) -> AgentState:
    from src.pipeline import get_retriever, get_reranker

    companies = state.get("companies") or []
    metrics = state.get("metrics") or []
    years = state.get("years") or []
    quarters = state.get("quarters") or []
    existing_evidence = state.get("evidence") or {}

    period_parts = years + quarters
    period = " ".join(period_parts) if period_parts else "latest"

    retriever = get_retriever()
    reranker = get_reranker()

    consistency_results = []

    for company in companies:
        for metric in metrics:
            source_contexts = {}

            for doc_type in DOC_TYPES:
                context = _retrieve_for_doc_type(
                    retriever=retriever,
                    reranker=reranker,
                    company=company,
                    metric=metric,
                    period=period,
                    doc_type=doc_type,
                )
                source_contexts[doc_type] = context if context else "No relevant content found."

            has_content = any(
                v != "No relevant content found."
                for v in source_contexts.values()
            )

            if not has_content:
                consistency_results.append({
                    "verdict": "insufficient_data",
                    "differences": [],
                    "summary": "No content found across any source type for this metric and period.",
                    "company": company,
                    "metric": metric,
                    "period": period,
                })
                continue

            prompt = DIFF_PROMPT.format(
                company=company,
                metric=metric,
                period=period,
                presentation_context=source_contexts["investor_presentation"],
                earnings_context=source_contexts["earnings_call"],
                annual_context=source_contexts["annual_report"],
            )

            result = _call_llm_with_retry(prompt)

            consistency_results.append({
                **result,
                "company": company,
                "metric": metric,
                "period": period,
            })

            time.sleep(1)

    updated_evidence = {
        **existing_evidence,
        "consistency": consistency_results,
    }

    return {
        **state,
        "evidence": updated_evidence,
    }


def should_run_consistency(state: AgentState) -> bool:
    metrics = state.get("metrics") or []
    quarters = state.get("quarters") or []
    years = state.get("years") or []
    companies = state.get("companies") or []

    return bool(metrics) and bool(quarters or years) and bool(companies)
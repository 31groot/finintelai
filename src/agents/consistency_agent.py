import json
import re

from dotenv import load_dotenv
from groq import Groq, GroqError

from src.agents.state import AgentState
from src.features.rag import RAGPipeline

load_dotenv()

_client = Groq()
_rag = RAGPipeline()

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


def _retrieve_for_doc_type(company: str, metric: str, period: str, doc_type: str) -> str:
    query = f"{company} {metric} {period} {doc_type}"

    result = _rag.get_context(
        query,
        memory_companies=[company],
    )

    docs = result.get("documents", [])
    if not docs:
        return ""

    filtered = []
    for doc, meta in zip(docs, result.get("metadata", [])):
        if str(meta.get("doc_type", "")).lower() == doc_type:
            filtered.append(doc)

    return "\n\n".join(filtered[:4]) if filtered else "\n\n".join(docs[:3])


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


def run_consistency_agent(state: AgentState) -> AgentState:
    companies = state.get("companies") or []
    metrics = state.get("metrics") or []
    years = state.get("years") or []
    quarters = state.get("quarters") or []
    existing_evidence = state.get("evidence") or {}

    period_parts = years + quarters
    period = " ".join(period_parts) if period_parts else "latest"

    consistency_results = []

    for company in companies:
        for metric in metrics:
            source_contexts = {}

            for doc_type in DOC_TYPES:
                context = _retrieve_for_doc_type(
                    company=company,
                    metric=metric,
                    period=period,
                    doc_type=doc_type,
                )
                source_contexts[doc_type] = context if context else "No relevant content found."

            prompt = DIFF_PROMPT.format(
                company=company,
                metric=metric,
                period=period,
                presentation_context=source_contexts["investor_presentation"],
                earnings_context=source_contexts["earnings_call"],
                annual_context=source_contexts["annual_report"],
            )

            try:
                response = _client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                raw = response.choices[0].message.content
                result = _parse_diff_response(raw)

            except GroqError:
                result = {
                    "verdict": "insufficient_data",
                    "differences": [],
                    "summary": "Consistency check could not be completed — LLM call failed.",
                }

            result["company"] = company
            result["metric"] = metric
            result["period"] = period
            consistency_results.append(result)

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
import json
import re
import time

from dotenv import load_dotenv
from groq import Groq, GroqError

from src.agents.state import AgentState

load_dotenv()

MAX_RETRIES = 2

_client = Groq()

VERIFICATION_PROMPT = """
You are a strict financial fact-checker.

You will be given:
1. A draft answer produced by a financial analyst.
2. The retrieved context that the analyst used.

Your job:
- Go through every number, percentage, metric value, company name, fiscal year, and quarter in the draft answer.
- For each claim, find the exact sentence or table cell in the retrieved context that supports it.
- If every claim has explicit support in the retrieved context, set grounded to true.
- If any claim cannot be traced to the retrieved context, set grounded to false and list that claim in unverified_claims.

Rules:
- Do not use outside knowledge to verify claims.
- A claim is only grounded if the exact value appears in the retrieved context.
- Approximate or inferred values do not count as grounded.
- "Not available in the retrieved context" is always a grounded answer — do not flag it.

Return your response as JSON in this exact format, nothing else:

{{
    "grounded": true or false,
    "confidence": 0.0 to 1.0,
    "unverified_claims": ["claim 1", "claim 2"],
    "reason": "one sentence explaining the verdict"
}}

Draft Answer:
{draft_answer}

Retrieved Context:
{context}
"""


def _parse_verification_response(raw: str) -> dict:
    raw = raw.strip()
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {
        "grounded": False,
        "confidence": 0.0,
        "unverified_claims": ["Verification response could not be parsed."],
        "reason": "Parser failed to extract JSON from LLM response.",
    }


def run_verification_agent(state: AgentState) -> AgentState:
    draft_answer = state.get("draft_answer", "")
    evidence = state.get("evidence") or {}
    context = evidence.get("context", "")

    if not draft_answer or not context:
        return {
            **state,
            "verification": {
                "grounded": False,
                "confidence": 0.0,
                "unverified_claims": ["Missing draft answer or context."],
                "reason": "Verification skipped — input incomplete.",
            },
        }

    prompt = VERIFICATION_PROMPT.format(
        draft_answer=draft_answer,
        context=context,
    )

    for attempt in range(3):
        try:
            response = _client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            verification = _parse_verification_response(
                response.choices[0].message.content
            )
            break
        except GroqError as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                verification = {
                    "grounded": False,
                    "confidence": 0.0,
                    "unverified_claims": [],
                    "reason": f"Verification unavailable — LLM call failed ({type(e).__name__}).",
                }

    return {
        **state,
        "verification": verification,
    }


def check_grounding(state: AgentState) -> str:
    verification = state.get("verification") or {}
    retry_count = state.get("retry_count", 0)

    if verification.get("grounded"):
        return "grounded"

    if retry_count >= MAX_RETRIES:
        return "give_up"

    return "retry"
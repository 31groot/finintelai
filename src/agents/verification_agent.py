import json
import re
import time
import os
from dotenv import load_dotenv
from src.llm.client import client

from src.agents.state import AgentState

load_dotenv()

MAX_RETRIES = 2

_client = client

VERIFICATION_PROMPT = """

You are a strict financial fact-checker.
 
You will be given:
1. The user's original question.
2. A draft answer produced by a financial analyst.
3. The retrieved context that the analyst used.
 
A claim is only grounded if it passes BOTH checks:
 
PART A - Existence: the exact value must appear in the retrieved context.
PART B - Relevance: the value must actually answer THIS question - the same
company, the same fiscal year or quarter, and the same metric the question
asks about.
 
A number that appears in the context but describes a different period, a
different company, or a different metric than the question asks about is NOT
grounded, even though it exists in the text. For example, if the question
asks for a full-year figure and the answer gives a quarterly figure that
happens to appear in the context, that is NOT grounded - it is the wrong
figure for the question.
 
Rules:
- Do not use outside knowledge to verify claims.
- A claim is grounded only if the exact value appears in the context AND it
  matches the company, period, and metric the question asks for.
- Approximate or inferred values do not count as grounded.
- If the answer gives a value for the wrong period, company, or metric, set
  grounded to false and explain the mismatch in the reason.
- "Not available in the retrieved context" is always a grounded answer — do not flag it.
 
Return your response as JSON in this exact format, nothing else:
 
{{
    "grounded": true or false,
    "confidence": 0.0 to 1.0,
    "unverified_claims": ["claim 1", "claim 2"],
    "reason": "one sentence explaining the verdict"
}}
 
Question:
{query}
 
Draft Answer:
{draft_answer}
 
Retrieved Context:
{context}
"""
 

def _parse_verification_response(raw: str):
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

def run_verification_agent(state: AgentState):
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
        query=state.get("query", ""),
        draft_answer=draft_answer,
        context=context,
    )

    verification = None
    for attempt in range(3):
        try:
            response = _client.responses.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                input=prompt,
            )
            verification = _parse_verification_response(response.output_text)
            break
        except Exception as e:
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

def check_grounding(state: AgentState):
    verification = state.get("verification") or {}
    retry_count = state.get("retry_count", 0)

    if verification.get("grounded"):
        return "grounded"

    if retry_count >= MAX_RETRIES:
        return "give_up"

    return "retry"
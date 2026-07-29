import logging
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.graph import run
from src.features.qa_chain import QAChain
from src.llm.usage import reset_usage, get_usage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("finintel.api")

app = FastAPI(
    title="FinIntel AI",
    description="Financial RAG over TCS / Infosys / Wipro filings (FY24–FY26).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_clarifier = QAChain()

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What was TCS revenue in FY26?"])
    companies: list[str] | None = Field(default=None, examples=[["tcs"]])


class Source(BaseModel):
    source: str
    page: int | str | None = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    cost_usd: float


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[Source]
    grounded: bool | None = None
    confidence: float | None = None
    subqueries: list[str] = []
    companies: list[str] = []
    retry_count: int | None = None
    latency_ms: float
    usage: Usage


class ClarifyResponse(BaseModel):
    """Returned when the pipeline needs the user to specify company or year."""
    request_id: str
    clarification: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str



@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check for load balancers / uptime monitors."""
    return HealthResponse(status="ok", version=app.version)


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={200: {"model": ClarifyResponse}},
)
def query(req: QueryRequest):
   
    request_id = str(uuid.uuid4())[:8]
    reset_usage()             
    start = time.perf_counter()

    if req.companies:
        _clarifier.memory["companies"] = list(req.companies)

    resolution = _clarifier.resolve(req.question)

    if "query" not in resolution:
        latency_ms = (time.perf_counter() - start) * 1000
        message = resolution.get("question", "Could you rephrase that?")
        logger.info(
            "req=%s type=clarify latency_ms=%.0f q=%r",
            request_id, latency_ms, req.question,
        )
        return ClarifyResponse(
            request_id=request_id,
            clarification=message,
            latency_ms=round(latency_ms, 1),
        )

    resolved_query = resolution["query"]
    companies = resolution.get("companies", [])

    result = run(resolved_query, companies=companies)

    latency_ms = (time.perf_counter() - start) * 1000
    usage = get_usage()
    verification = result.get("verification") or {}

    detected = result.get("companies") or []
    if detected:
        _clarifier.memory["companies"] = list(detected)

    sources = [
        Source(source=m.get("source", "unknown"), page=m.get("page"))
        for m in (result.get("metadata") or [])
    ]

    logger.info(
        "req=%s type=answer latency_ms=%.0f tokens=%d cost_usd=%.5f "
        "retries=%s grounded=%s conf=%s q=%r",
        request_id, latency_ms, usage.total_tokens, usage.cost_usd,
        result.get("retry_count"), verification.get("grounded"),
        verification.get("confidence"), resolved_query,
    )

    return QueryResponse(
        request_id=request_id,
        answer=result.get("answer", ""),
        sources=sources,
        grounded=verification.get("grounded"),
        confidence=verification.get("confidence"),
        subqueries=result.get("subqueries", []),
        companies=companies,
        retry_count=result.get("retry_count"),
        latency_ms=round(latency_ms, 1),
        usage=Usage(**usage.as_dict()),
    )
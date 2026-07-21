import re
from dotenv import load_dotenv
from groq import Groq, GroqError
from src.features.rag import RAGPipeline
from src.retrieval.query_decomposer import company_aliases

load_dotenv()

class QAChain:

        comparison_words = [
            "compare",
            "comparison",
            "rank",
            "ranking",
            "highest",
            "lowest",
            "best",
            "worst",
            "better",
            "versus",
            "vs",
        ]

        description_words = [
            "what does",
            "what does company do",
            "business",
            "overview",
        ]

        temporal_words = [
            "growth"
            "by year",
            "year wise",
            "year-wise",
            "over the years",
            "across years",
            "last 2 years",
            "last 3 years",
            "last 5 years",
            "past 2 years",
            "past 3 years",
            "trend",
            "historical",
            "history",
            "over time",
            "yearly",
            "fy26",
            "fy25",
            "fy24",
            "fiscal 2026",
            "fiscal 2025",
            "fiscal 2024",
        ]

    def __init__(self):
        self.rag = RAGPipeline()
        self.client = Groq()
        self.memory = {
            "companies": [],
            "pending_year_clarification": None,
            "pending_company_clarification": None,
        }
        self.valid_years = self._load_available_fiscal_years()

    def _load_available_fiscal_years(self):
        years = set()

        for meta in self.rag.retriever.metadata:
            meta = meta or {}
            fy = meta.get("fiscal_year")
            if fy:
                years.add(str(fy).upper())

        return years

    def _sort_fiscal_years_desc(self, years):
        def year_key(fy):
            match = re.search(r"(\d{2,4})", str(fy))
            if not match:
                return -1
            value = match.group(1)
            if len(value) == 4:
                value = value[-2:]
            return int(value)

        return sorted(years, key=year_key, reverse=True)

    def _extract_fiscal_year(self, query):
        query_lower = query.lower()

        match = re.search(r"\bfy[\s_-]?(\d{2,4})\b", query_lower)
        if match:
            year = match.group(1)
            if len(year) == 2:
                return f"FY{year}"
            return f"FY{year[-2:]}"

        match = re.search(r"\bfiscal[\s_-]?(\d{4})\b", query_lower)
        if match:
            year = match.group(1)
            return f"FY{year[-2:]}"

        return None

    def _needs_company_clarification(self, query):
        query_lower = query.lower()

        companies = self.rag.decomposer._find_companies(query_lower)
        metrics = self.rag.decomposer._find_metrics(query_lower)

        has_memory = bool(self.memory["companies"])

        is_description = any(word in query_lower for word in description_words)
        is_comparison = any(word in query_lower for word in comparison_words)
        is_temporal = any(word in query_lower for word in temporal_words)

        if companies:
            return False

        if has_memory:
            return False

        if metrics or is_description or is_comparison or is_temporal:
            return True

        return False

    def _needs_year_clarification(self, query):
        query_lower = query.lower()

        companies = self.rag.decomposer._find_companies(query_lower)
        metrics = self.rag.decomposer._find_metrics(query_lower)

        explicit_year = self._extract_fiscal_year(query)
        is_comparison = any(word in query_lower for word in comparison_words)
        is_temporal = any(word in query_lower for word in temporal_words)

        if explicit_year or is_temporal:
            return False

        if metrics and companies:
            return True

        if is_comparison and len(companies) >= 2 and metrics:
            return True

        return False

    def _update_memory(self, query):
        current_companies = self.rag.decomposer._find_companies(query.lower())
        if current_companies:
            self.memory["companies"] = current_companies

    def _build_prompt(self, query, context):
        return f"""
You are a financial analyst. Answer ONLY using the provided context.

Rules:

- Do not use outside knowledge.
- Do not invent facts, numbers, years, currencies, units, or reporting periods.
- If the requested information is not explicitly present in the retrieved context, write exactly:
  Not available in the retrieved context.
- Never guess or infer missing values.
- Do not calculate, derive, normalize, or convert values unless the user explicitly requests it and the retrieved context provides all required information.
- When both standalone and consolidated financial figures are present and the user does not specify which one they want, prefer consolidated figures.
- Ignore note disclosures, subsidiary schedules, related-party disclosures, accounting policies, and appendices unless the user explicitly asks about them.
- For change/delta questions (e.g. "how did revenue change from FY24 to FY26"), 
  if both period values are present in the context, report both values and 
  calculate the change. This is explicitly permitted.
Context Usage:

- Use all relevant retrieved chunks before answering.
- Do not stop after finding the first matching value.
- If the requested information appears across multiple retrieved chunks, combine the information into a single answer.
- If duplicate information appears, use the clearest and most complete version.

Single-company questions:

- Extract the requested value directly from the most relevant context.
- Prefer the chunk containing the requested company, fiscal year, and metric together.
- Do not replace missing values using information from another fiscal year or company.

Year-wise / Trend questions:

- Search all retrieved chunks and extract every fiscal year or reporting period explicitly available. Do not stop after finding the first year.
- Include only years that appear in the context.
- Do not invent missing years.
- After listing the values, briefly summarize the observed trend.

Comparison questions:

- Extract values for every requested company.
- Compare only if:
  - the metric is identical,
  - the reporting period is identical,
  - the units and currencies are identical.
- If units or currencies differ, write exactly:
  Ranking not possible because the reported units/currencies differ across companies.
- If one or more companies are missing values, report the available values and state which companies are missing.
- Search all retrieved chunks before concluding that a company's value is unavailable.
- If multiple chunks contain the requested metric, use the clearest and most complete value.
- Do not stop after finding the first company's value.

Business overview questions:

- Summarize only information explicitly stated in the retrieved context.
- Do not add external company knowledge.

Presentation and Earnings Call questions:

- Prioritize management commentary, guidance, outlook, strategy, AI initiatives, deal wins, pipeline, bookings, hiring, macro commentary, client demand, pricing, and operational highlights when available in the retrieved context.

Output Format

1) Single-company KPI

Company: <company>

Metric: <metric>

Value: <value or "Not available in the retrieved context">

Reporting period: <period or "Not available in the retrieved context">

Unit/Currency: <unit or "Not available in the retrieved context">

Explanation:
<1-2 concise sentences>

--------------------------------------------------

2) Year-wise / Trend

| Year / Period | Metric | Value | Unit/Currency |
|---------------|--------|-------|---------------|

Explanation:
<brief trend summary>

--------------------------------------------------

3) Comparison

| Company | Metric | Value | Unit/Currency |
|---------|--------|-------|---------------|

Unit Validation:
<same unit/currency or different>

Ranking:
<ranking or the exact required sentence>

Explanation:
<brief comparison summary>

--------------------------------------------------

4) Business Overview

Company: <company>

Business Overview:

- <point 1>

- <point 2>

- <point 3>

# After line 241 (the </brief trend summary> line), add:

--------------------------------------------------

5) Change / Delta question (e.g. "how did X change from FY24 to FY26")

Metric: <metric>
Company: <company>

| Period     | Value          | Unit/Currency |
|------------|----------------|---------------|
| <earlier>  | <earlier value>| <unit>        |
| <later>    | <later value>  | <unit>        |

Change: <absolute change> (<percentage change if calculable from the two values above>)

Explanation:
<1-2 sentences on direction and magnitude of change>

Context:
{context}

Question:
{query}

Answer:
"""

    def _generate_answer(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content, None
        except GroqError as e:
            return None, (
                "Sorry, I couldn't reach the AI service right now "
                f"({type(e).__name__}). Please try again in a moment."
            )

    def _format_citations(self, sources, max_sources=15):
        if not sources:
            return ""

        citations = "\n\nSources:\n"
        seen = set()
        count = 0

        for item in sources:
            source = item.get("source", "unknown_source")
            page = item.get("page", "N/A")
            key = (source, page)

            if key not in seen:
                citations += f"- {source} (page {page})\n"
                seen.add(key)
                count += 1

            if count >= max_sources:
                break

        return citations

    def _handle_pending_company_clarification(self, query):
        pending = self.memory.get("pending_company_clarification")
        if not pending:
            return None

        query_clean = query.strip().lower()
        clarification_patterns = []
        for aliases in company_aliases.values():
            for alias in aliases:
                clarification_patterns.append(
                    rf"^(for\s+)?{re.escape(alias)}$"
                )

        is_company_only_reply = any(
            re.fullmatch(pattern, query_clean) for pattern in clarification_patterns
        )

        if not is_company_only_reply:
            return None

        companies = self.rag.decomposer._find_companies(query_clean)

        if not companies:
            return {
                "error": "Please choose a valid company: TCS, Infosys, or Wipro.",
            }

        company_text = " and ".join(companies)
        original_query = pending["original_query"]
        resolved_query = f"{original_query} for {company_text}"

        self.memory["companies"] = companies
        self.memory["pending_company_clarification"] = None
        return {"resolved_query": resolved_query}

    def _handle_pending_year_clarification(self, query):
        pending = self.memory.get("pending_year_clarification")
        if not pending:
            return None

        year = self._extract_fiscal_year(query)

        if not year:
            valid_text = ", ".join(self._sort_fiscal_years_desc(pending["valid_years"]))
            return {
                "error": f"Please choose a valid fiscal year: {valid_text}.",
            }

        if year not in pending["valid_years"]:
            valid_text = ", ".join(self._sort_fiscal_years_desc(pending["valid_years"]))
            return {
                "error": f"I currently have data for {valid_text}. Please choose one of those.",
            }

        original_query = pending["original_query"]
        resolved_query = f"{original_query} in {year}"

        self._update_memory(original_query)
        self.memory["pending_year_clarification"] = None
        return {"resolved_query": resolved_query}

    def ask_with_trace(self, query):
        query = query.strip()

        if not query:
            return { "error": "Please enter a question."}

        pending_company_resolution = self._handle_pending_company_clarification(query)
        if pending_company_resolution:
            if "resolved_query" in pending_company_resolution:
                query = pending_company_resolution["resolved_query"]
            else:
                return pending_company_resolution

        pending_year_resolution = self._handle_pending_year_clarification(query)
        if pending_year_resolution:
            if "resolved_query" in pending_year_resolution:
                query = pending_year_resolution["resolved_query"]
            else:
                return pending_year_resolution

        if self._needs_company_clarification(query):
            self.memory["pending_company_clarification"] = {"original_query": query}

            return {
                "error": (
                    "Which company do you want this for - TCS, Infosys, or Wipro?"
                ),
            }

        if self._needs_year_clarification(query):
            companies = self.rag.decomposer._find_companies(query.lower())
            valid_years = self.valid_years or {"FY24", "FY25", "FY26"}
            valid_text = ", ".join(self._sort_fiscal_years_desc(valid_years))

            self.memory["pending_year_clarification"] = {
                "original_query": query,
                "companies": companies,
                "valid_years": valid_years,
            }

            return {
                "error": f"Which fiscal year should I compare — {valid_text}?",
            }

        self._update_memory(query)

        result = self.rag.get_context(query, memory_companies=self.memory["companies"])

        context = result["context"]
        documents = result.get("documents", [])
        metadata = result.get("metadata", [])
        subqueries = result.get("subqueries", [])

        prompt = self._build_prompt(query, context)
        answer, error = self._generate_answer(prompt)

        if error:
            return {"error": error}

        citations = self._format_citations(metadata)

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "documents": documents,
            "contexts": documents,
            "metadata": metadata,
            "subqueries": subqueries,
            "memory_companies": list(self.memory["companies"]),
        }

    def ask(self, query):
        trace = self.ask_with_trace(query)

        if "error" in trace:
            return trace["error"]

        return trace["answer"] + trace["citations"]

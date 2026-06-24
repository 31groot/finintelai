import re
from dotenv import load_dotenv
from groq import Groq, GroqError
from src.features.rag import RAGPipeline

load_dotenv()


class QAChain:
    def __init__(self):
        self.rag = RAGPipeline()
        self.client = Groq()
        self.memory = {
            "companies": [],
            "pending_year_clarification": None,
            "pending_company_clarification": None,
        }

        self.comparison_words = [
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

        self.description_words = [
            "what does",
            "business",
            "company",
            "do",
            "overview",
            "services",
            "operations",
        ]

        self.temporal_words = [
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
            "past 5 years",
            "trend",
            "historical",
            "history",
            "over time",
            "yearly",
            "fy26",
            "fy25",
            "fy24",
            "fy23",
            "fiscal 2026",
            "fiscal 2025",
            "fiscal 2024",
            "fiscal 2023",
        ]

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

        is_description = any(word in query_lower for word in self.description_words)
        is_comparison = any(word in query_lower for word in self.comparison_words)
        is_temporal = any(word in query_lower for word in self.temporal_words)

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

        is_comparison = any(word in query_lower for word in self.comparison_words)
        explicit_year = self._extract_fiscal_year(query)

        if is_comparison and len(companies) >= 2 and metrics and not explicit_year:
            return True

        return False

    def _update_memory(self, query):
        current_companies = self.rag.decomposer._find_companies(query.lower())
        if current_companies:
            self.memory["companies"] = current_companies

    def _build_prompt(self, query, context):
        return f"""
You are a financial analyst. Answer ONLY from the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent numbers, years, units, currencies, or reporting periods.
- If a requested value is missing or not explicitly stated for the requested company/metric/year, write exactly:
  Not available in the retrieved context.
- Do not calculate, derive, normalize, or convert values unless the user explicitly asks and the context provides the required basis.
- For year-wise questions, include only years explicitly present in the context.
- For comparison questions, compare or rank only if values are for the same metric, same fiscal year, and same unit/currency.
- If units/currencies differ, write exactly:
  Ranking not possible because the reported units/currencies differ across companies.
- Keep the answer concise and factual.
- When a question asks for revenue, profit, or other financial metrics 
  without specifying standalone or consolidated, prefer consolidated 
  figures if both are available in the context.


Answer format:

1) Single-company KPI
Company: <company>
Metric: <metric>
Value: <value>
Reporting period: <period or "Not available in the retrieved context">
Unit/Currency: <unit or "Not available in the retrieved context">

Explanation:
<1-2 sentence explanation>

2) Year-wise / trend question
| Year / Period | Metric | Value | Unit/Currency |
|---------------|--------|-------|---------------|

Explanation:
<brief trend summary>

3) Comparison question
| Company | Metric | Value | Unit/Currency |
|---------|--------|-------|---------------|

Unit Validation:
<whether units/currencies match>

Ranking:
<ranking if directly comparable, otherwise the exact sentence above>

Explanation:
<brief explanation>

4) Business / company overview
Company: <company>

Business Overview:
- <bullet 1>
- <bullet 2>
- <bullet 3>

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

        clarification_patterns = [
            r"^(for\s+)?tcs$",
            r"^(for\s+)?infosys$",
            r"^(for\s+)?wipro$",
            r"^(for\s+)?tata consultancy services$",
            r"^(for\s+)?tata consultancy$",
        ]

        is_company_only_reply = any(
            re.fullmatch(pattern, query_clean) for pattern in clarification_patterns
        )

        if not is_company_only_reply:
            return None

        companies = self.rag.decomposer._find_companies(query_clean)

        if not companies:
            return {
                "ok": False,
                "needs_clarification": True,
                "clarification_type": "company",
                "error": "Please choose a valid company: TCS, Infosys, or Wipro."
            }

        selected_company = companies[0]
        original_query = pending["original_query"]
        resolved_query = f"{original_query} for {selected_company}"

        self.memory["companies"] = [selected_company]
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
                "ok": False,
                "needs_clarification": True,
                "clarification_type": "fiscal_year",
                "error": f"Please choose a valid fiscal year: {valid_text}."
            }

        if year not in pending["valid_years"]:
            valid_text = ", ".join(self._sort_fiscal_years_desc(pending["valid_years"]))
            return {
                "ok": False,
                "needs_clarification": True,
                "clarification_type": "fiscal_year",
                "error": f"I currently have data for {valid_text}. Please choose one of those."
            }

        original_query = pending["original_query"]
        resolved_query = f"{original_query} in {year}"

        self._update_memory(original_query)
        self.memory["pending_year_clarification"] = None
        return {"resolved_query": resolved_query}

    def ask_with_trace(self, query):
        query = query.strip()

        if not query:
            return {
                "ok": False,
                "error": "Please enter a question."
            }

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
            self.memory["pending_company_clarification"] = {
                "original_query": query
            }

            return {
                "ok": False,
                "needs_clarification": True,
                "clarification_type": "company",
                "error": (
                    "Which company do you want this for — "
                    "TCS, Infosys, or Wipro?"
                )
            }

        if self._needs_year_clarification(query):
            companies = self.rag.decomposer._find_companies(query.lower())
            valid_years = self.valid_years or {"FY23", "FY24", "FY25", "FY26"}
            valid_text = ", ".join(self._sort_fiscal_years_desc(valid_years))

            self.memory["pending_year_clarification"] = {
                "original_query": query,
                "companies": companies,
                "valid_years": valid_years
            }

            return {
                "ok": False,
                "needs_clarification": True,
                "clarification_type": "fiscal_year",
                "error": f"Which fiscal year should I compare — {valid_text}?"
            }

        self._update_memory(query)

        result = self.rag.get_context(
            query,
            memory_companies=self.memory["companies"]
        )

        context = result["context"]
        documents = result.get("documents", [])
        metadata = result.get("metadata", [])
        subqueries = result.get("subqueries", [])

        prompt = self._build_prompt(query, context)
        answer, error = self._generate_answer(prompt)

        if error:
            return {
                "ok": False,
                "error": error
            }

        citations = self._format_citations(metadata)

        return {
            "ok": True,
            "query": query,
            "answer": answer,
            "citations": citations,
            "documents": documents,
            "contexts": documents,
            "metadata": metadata,
            "subqueries": subqueries,
            "memory_companies": list(self.memory["companies"])
        }

    def ask(self, query):
        trace = self.ask_with_trace(query)

        if not trace["ok"]:
            return trace["error"]

        return trace["answer"] + trace["citations"]
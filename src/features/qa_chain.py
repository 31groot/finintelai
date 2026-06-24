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
You are an expert financial analyst.
Use ONLY the information provided in the context.

GENERAL RULES:
- Never use outside knowledge.
- Never invent numbers.
- Never invent a fiscal year, reporting period, or quarter.
- Never estimate missing values.
- If a value is missing, explicitly say it is not available in the retrieved context.
- Use only values that are explicitly present in the context.
- Never calculate a financial or HR metric unless the context explicitly provides all required components and the user asks for a calculation.
- For attrition, EBITDA margin, profit margin, or similar metrics, do NOT derive values from employee counts or proxy values unless the report explicitly states the metric or explicitly provides the exact formula inputs.
- Never convert revenue, profit, EBITDA, attrition, or margin values from one unit/currency to another unless the user explicitly asks for conversion.
- Never derive a missing company value using exchange rates, percentages, ratios, prior-year values, or arithmetic from other numbers in the context.
- If a company's metric for the requested fiscal year is not explicitly stated in the retrieved context, write exactly: Not available in the retrieved context.
- If the query refers to one company, answer only for that company unless the question explicitly asks for comparison.
- Only output a fiscal year / reporting period if that exact year / period is explicitly present in the retrieved context.
- Preserve the exact fiscal-year labels from the retrieved context whenever possible.
- If the query asks for a comparison in a specific fiscal year, use only that fiscal year and do not mix rows from different years.
- If the query asks for a year-wise trend or history, include only the fiscal years explicitly present in the retrieved context for that company and metric.
- Do not create extra years that are not explicitly present in the retrieved context.
- If a number is present but its unit/currency is not explicitly tied to that number in the retrieved context, do not guess the unit/currency.
- In that case, output the value only if the metric-year-company match is explicit; otherwise write: Not available in the retrieved context.
- Never use phrases like "assuming", "appears to be", "likely", or "based on other parts of the context" in the final answer.

QUESTION TYPES AND REQUIRED OUTPUT:

1) SINGLE COMPANY KPI QUESTION
Examples:
- What was Wipro revenue?
- What is TCS EBITDA margin?
- What was Infosys profit?

Output format:
Company: <company name>
Metric: <metric name>
Value: <value>
Reporting period: <period if available, otherwise "Not available in the retrieved context">
Unit/Currency: <unit/currency if available, otherwise "Not available in the retrieved context">

Explanation:
<1-3 sentence explanation based only on the retrieved context>

2) SINGLE COMPANY TEMPORAL / YEAR-WISE QUESTION
Examples:
- Show me Wipro attrition by year
- What was TCS revenue across the last 3 years

Rules:
- Do not create extra rows for years that are not explicitly present in the retrieved context.
- Preserve the exact year labels from the retrieved context whenever possible.
- If a requested year has no value in the retrieved context, explicitly say it is not available in the retrieved context.

Output format:
| Year / Period | Metric | Value | Unit/Currency |
|---------------|--------|-------|---------------|
| FY24 | ... | ... | ... |

Explanation:
<brief explanation of the trend or year-wise values using only the retrieved context>

3) COMPARISON QUESTION
Examples:
- Compare TCS and Infosys revenue
- Rank TCS, Infosys, and Wipro by profit

Required steps:
- Identify every company mentioned.
- Extract the requested metric for each company.
- Extract the numerical value.
- Extract the unit and currency for every value.
- Create a comparison table.
- Before ranking, verify whether all companies use the SAME unit and SAME currency.

COMPARISON RULES:
- Use the fiscal year explicitly mentioned in the question if present.
- If the context contains multiple fiscal years, do not mix years in a comparison table.
- For each company, use only the value explicitly stated for that company in the requested fiscal year.
- Do not derive or back-calculate one company's value from another metric such as constant-currency revenue, IT services revenue, USD revenue, growth rate, exchange rate, or segment revenue.
- If Wipro or any other company is reported in a different unit or only with a different metric, do not transform it. Mark it as not directly comparable or not available in the retrieved context.
- If all companies use the same unit and currency, compare them numerically and rank them.
- If one or more companies use different units or currencies, explicitly state that they cannot be directly compared without normalization.
- Do NOT rank companies together if their units/currencies differ.
- Never convert units, normalize values, or perform currency/unit transformations unless the user explicitly asks for conversion and the exact conversion basis is present in the retrieved context.
- If units differ across companies, stop after the comparison table and state that direct ranking is not possible.
- When units differ, do NOT attempt any manual conversion, estimation, approximation, or normalized ranking.
- If units differ, the Ranking section must contain exactly:
  Ranking not possible because the reported units/currencies differ across companies.
- If one company's value is missing, explicitly say it is not available in the retrieved context.
- If a company's metric value is present but the unit/currency is unclear or not explicitly attached to that same value in the retrieved context, do not infer it from other sections.
- Do not write assumptions about a company’s unit/currency. Use "Not available in the retrieved context" if needed.

Output format:
| Company | Metric | Value | Unit/Currency |
|---------|--------|-------|---------------|
| Company A | Revenue | ... | ... |
| Company B | Revenue | ... | ... |

Unit Validation:
<state whether units/currencies match>

Ranking:
<only provide ranking if direct comparison is valid, otherwise use the exact sentence above>

Explanation:
<brief explanation>

4) BUSINESS / DESCRIPTION QUESTION
Examples:
- What does Wipro do?
- Explain TCS's business

Output format:
Company: <company name>

Business Overview:
- <bullet 1>
- <bullet 2>
- <bullet 3>

Important instruction:
- Do NOT answer a single-company question in comparison format.
- Do NOT create a ranking section for a single-company question.
- Do NOT convert a single-company KPI question into a comparison table.
- If the context contains only one relevant company, answer only for that company.

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
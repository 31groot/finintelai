import re
from dotenv import load_dotenv
from src.pipeline import get_pipeline
from src.retrieval.query_decomposer import company_aliases
load_dotenv()


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
    "what does company",
    "business",
    "overview",
]

temporal_words = [
    "growth",
    "by year",
    "year wise",
    "year-wise",
    "over the year",
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

superlative_words = [
    "highest", 
    "lowest", 
    "best", 
    "worst", 
    "most", 
    "least",
    "which company",
    "top", 
    "leader", 
    "leading",
]

class QAChain:

    def __init__(self):
        self.rag = get_pipeline()        
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
        is_superlative = any(w in query_lower for w in superlative_words)

        if is_superlative:
            return False

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

    def _handle_pending_company_clarification(self, query):
        pending = self.memory.get("pending_company_clarification")
        if not pending:
            return None

        query_clean = query.strip().lower()
        clarification_patterns = []
        for aliases in company_aliases.values():
            for alias in aliases:
                clarification_patterns.append(rf"^(for\s+)?{re.escape(alias)}$")

        is_company_only_reply = any(
            re.fullmatch(pattern, query_clean) for pattern in clarification_patterns
        )

        if not is_company_only_reply:
            return None

        companies = self.rag.decomposer._find_companies(query_clean)

        if not companies:
            return {
                "question": "Please choose a valid company: TCS, Infosys, or Wipro.",
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
            valid_text = ", ".join(
                self._sort_fiscal_years_desc(pending["valid_years"])
            )
            return {
                "question": f"Please choose a valid fiscal year: {valid_text}.",
            }

        if year not in pending["valid_years"]:
            valid_text = ", ".join(
                self._sort_fiscal_years_desc(pending["valid_years"])
            )
            return {
                "question": f"I currently have data for {valid_text}. Please choose one of those.",
            }

        original_query = pending["original_query"]
        resolved_query = f"{original_query} in {year}"

        self._update_memory(original_query)
        self.memory["pending_year_clarification"] = None
        return {"resolved_query": resolved_query}

    def resolve(self, query):
        query = query.strip()

        if not query:
            return {"question": "Please enter a question."}

        pending_company_resolution = self._handle_pending_company_clarification(
            query
        )
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
                "question": (
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
                "question": f"Which fiscal year should I compare - {valid_text}?",
            }

        self._update_memory(query)

        return {"query": query, "companies": list(self.memory["companies"])}
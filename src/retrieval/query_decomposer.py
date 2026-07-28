import re

company_aliases = {
    "tcs": [
        "tcs",
        "tata consultancy services",
        "tata consultancy",
        "tata_consultancy_services",
    ],
    "infosys": [
        "infosys",
    ],
    "wipro": [
        "wipro",
    ],
}

company_retrieval_map = {
    "tcs": "Tata Consultancy Services TCS",
    "infosys": "Infosys",
    "wipro": "Wipro",
}

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
    "overview",
    "business overview",
    "what is the business of",
    "describe the business",
    "company profile",
]

metric_keywords = [
    "revenue",
    "growth",
    "profit",
    "headcount",
    "ebitda",
    "margin",
    "attrition",
    "employees",
    "employee",
    "bookings",
    "tcv",
    "guidance",
    "deal",
    "deals",
    "cash flow",
    "operating income",
    "sales",
    "americas", 
    "europe", 
    "united kingdom", 
    "segment"
]

metric_retrieval_map = {
    "headcount": "headcount total employees closing headcount workforce number of employees employee count permanent employees",
    "employees": "headcount total employees closing headcount workforce number of employees permanent employees",
    "tcv": "total contract value TCV order book bookings deal value new deals",
    "bookings": "total bookings order book TCV bookings for the quarter deal bookings",
    "cash flow": "free cash flow operating cash flow cash generated from operations",
    "revenue": "consolidated revenue from operations total income turnover sales revenue consolidated statement of profit and loss",
    "sales": "consolidated revenue from operations total income turnover sales revenue consolidated statement of profit and loss",
    "growth": (
        "revenue growth "
        "YoY growth "
        "QoQ growth "
        "year-on-year growth "
        "quarter-on-quarter growth "
        "constant currency growth "
        "CC growth"
    ),
    "standalone": "standalone statement of profit and loss standalone revenue from operations standalone financial statements",
    "operating income": "operating income operating profit results from operating activities EBIT segment operating income IT services operating income",
    "profit": "consolidated profit after tax consolidated PAT profit attributable to equity holders consolidated net profit net income consolidated statement of profit and loss attributable to shareholders",
    "ebitda": "consolidated ebitda earnings before interest tax depreciation amortization consolidated operating profit",
    "attrition": "attrition voluntary attrition employee turnover workforce attrition annual attrition rate",
    "margin": "consolidated margin consolidated operating margin EBITDA margin EBIT margin profit margin consolidated net profit margin IT services operating margin",
    "guidance": "revenue guidance growth guidance FY guidance constant currency guidance management guidance outlook",
    "deal": "deal wins total contract value TCV large deals mega deals order bookings new contracts won",
    "deals": "deal wins total contract value TCV large deals mega deals order bookings new contracts won",
    "americas": "Americas geographic segment revenue United States Canada geographic breakdown",
    "europe": "Europe geographic segment revenue United Kingdom Continental Europe geographic breakdown",
    "segment": "segment revenue geographic segment vertical segment revenue breakdown",
}

NARRATIVE_TOPICS = {
    "guidance": [
        "guidance",
        "provide",
        "forecast",
        "outlook",
        "projections",
        "expected growth",
    ],
    "outlook": [
        "outlook",
        "future outlook",
        "growth trajectory",
        "demand environment",
        "macro environment",
    ],
    "deal wins": [
        "deal wins",
        "deals",
        "tcv",
        "large deals",
        "contract",
        "contracts",
        "order book",
        "wins",
    ],
}

topic_retrieval_map = {
    "ai": "AI services artificial intelligence AI revenue annualized AI generative AI GenAI",
    "guidance": "management guidance growth forecast outlook business projections management commentary forward looking expectations",
    "outlook": "business outlook demand environment growth trajectory macroeconomic commentary executive outlook",
    "deal wins": "deal wins total contract value TCV large deals mega deals order bookings new contracts won",
}

quarter_patterns = {
    "Q1": [r"\bq1\b", r"\bq\s*1\b", r"\bquarter\s*1\b", r"\bfirst quarter\b"],
    "Q2": [r"\bq2\b", r"\bq\s*2\b", r"\bquarter\s*2\b", r"\bsecond quarter\b"],
    "Q3": [r"\bq3\b", r"\bq\s*3\b", r"\bquarter\s*3\b", r"\bthird quarter\b"],
    "Q4": [r"\bq4\b", r"\bq\s*4\b", r"\bquarter\s*4\b", r"\bfourth quarter\b"],
}

temporal_words = [
    "by year",
    "year wise",
    "year-wise",
    "over the years",
    "across years",
    "last 2 years",
    "last 3 years",
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

available_fiscal_years = [
    "FY26",
    "FY25",
    "FY24",
]

FINANCIAL_METRICS = {"revenue", "profit", "ebitda", "margin", "growth", "sales"}


def _fill_year_range(years):

    if len(years) < 2:
        return years

    def fy_to_int(fy):
        m = re.search(r"(\d{2,4})$", str(fy))
        if not m:
            return -1
        v = m.group(1)
        return int(v[-2:])

    year_ints = [fy_to_int(y) for y in years]
    min_yr = min(year_ints)
    max_yr = max(year_ints)

    filled = []
    for fy in available_fiscal_years:
        n = fy_to_int(fy)
        if min_yr <= n <= max_yr:
            filled.append(fy)

    return filled if filled else years


class QueryDecomposer:
    def decompose(self, query, memory_companies=None):
        query_lower = query.lower()

        companies = self._find_companies(query_lower)
        if not companies and memory_companies:
            companies = list(memory_companies)

        metrics = self._find_metrics(query_lower)
        topics = self._find_topics(query_lower)

        is_comparison = (
            any(word in query_lower for word in comparison_words)
            or len(companies) > 1
        )

        is_description = (
            any(word in query_lower for word in description_words)
            and not metrics
        )

        explicit_years = self._extract_explicit_fiscal_years(query_lower)

        is_range_query = bool(
            re.search(r"\bfy\d{2,4}\b.*\bto\b.*\bfy\d{2,4}\b", query_lower)
            or re.search(r"\bfrom\b.*\bfy\d{2,4}\b.*\bto\b.*\bfy\d{2,4}\b", query_lower)
            or "change" in query_lower
            or "trend" in query_lower
            or "over" in query_lower
        )
        if is_range_query and len(explicit_years) >= 2:
            explicit_years = _fill_year_range(explicit_years)

        requested_years = self._get_requested_temporal_years(
            query_lower, explicit_years
        )
        requested_quarters = self._extract_quarters(query_lower)

        needs_consolidated = bool(set(metrics) & FINANCIAL_METRICS)

        if "standalone" in query_lower:
            statement_type_filter = "standalone"
        elif needs_consolidated:
            statement_type_filter = "consolidated"
        else:
            statement_type_filter = None

        time_suffixes = []
        if requested_years and requested_quarters:
            for y in requested_years:
                for q in requested_quarters:
                    time_suffixes.append(f"{y} {q}")
        elif requested_years:
            time_suffixes = list(requested_years)
        elif requested_quarters:
            time_suffixes = list(requested_quarters)
        else:
            time_suffixes = [""]

        subqueries = []

        if companies and is_comparison and metrics:
            merged_metrics = list(metrics)
            if "revenue" in merged_metrics and "growth" in merged_metrics:
                merged_metrics.remove("revenue")
                
            for company in companies:
                company_text = company_retrieval_map.get(company, company)
                for metric in merged_metrics:
                    metric_text = metric_retrieval_map.get(metric, metric)
                    if "standalone" in query_lower:
                        metric_text = metric_retrieval_map["standalone"] + " " + metric_text
                    if "constant currency" in query_lower or "constant-currency" in query_lower:
                        metric_text = "constant currency growth CC revenue growth reported growth " + metric_text
                    for time in time_suffixes:
                        subqueries.append(
                            f"{company_text} {metric_text} {time}".strip()
                        )

        elif companies and is_comparison:
            for company in companies:
                company_text = company_retrieval_map.get(company, company)
                metric_text = metric_retrieval_map["revenue"]
                for time in time_suffixes:
                    subqueries.append(
                        f"{company_text} {metric_text} {time}".strip()
                    )
            statement_type_filter = "consolidated"

        elif len(companies) > 1 and is_description:
            for company in companies:
                company_text = company_retrieval_map.get(company, company)
                subqueries.append(
                    f"{company_text} business overview services operations company profile"
                )

        elif len(companies) == 1:
            company = companies[0]
            company_text = company_retrieval_map.get(company, company)

            if metrics:
                merged_metrics = list(metrics)
                if "revenue" in merged_metrics and "growth" in merged_metrics:
                    merged_metrics.remove("revenue")

                for metric in merged_metrics:
                    metric_text = metric_retrieval_map.get(metric, metric)
                    if "standalone" in query_lower:
                        metric_text = metric_retrieval_map["standalone"] + " " + metric_text
                    if "constant currency" in query_lower or "constant-currency" in query_lower:
                        metric_text = "constant currency growth CC revenue growth reported growth " + metric_text
                    for time in time_suffixes:
                        subqueries.append(
                            f"{company_text} {metric_text} {time}".strip()
                        )

            elif topics:
                for topic in topics:
                    topic_text = topic_retrieval_map.get(topic, topic)
                    for time in time_suffixes:
                        subqueries.append(
                            f"{company_text} {topic_text} {time}".strip()
                        )

            elif is_description:
                subqueries.append(
                    f"{company_text} business overview services operations company profile"
                )

            else:
                cleaned = re.sub(
                    r"^(what|how|why|who|when|did|does|which|were|tell me|can you|please)\s+",
                    "",
                    query,
                    flags=re.IGNORECASE,
                ).strip()
                subqueries.append(f"{company_text} {cleaned}")

        if not subqueries:
            subqueries = [query.strip()]

        return {
            "subqueries": subqueries,
            "companies": companies,
            "metrics": metrics,
            "topics": topics,
            "years": requested_years,
            "quarters": requested_quarters,
            "statement_type": statement_type_filter,
            "is_comparison": is_comparison,
        }

    def _find_companies(self, query_lower):
        companies = []
        for canonical_company, aliases in company_aliases.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", query_lower):
                    if canonical_company not in companies:
                        companies.append(canonical_company)
                    break
        return companies

    def _find_metrics(self, query_lower):
        return [metric for metric in metric_keywords if metric in query_lower]

    def _find_topics(self, query_lower):
        matched_topics = []
        for topic, patterns in NARRATIVE_TOPICS.items():
            if any(word in query_lower for word in patterns):
                matched_topics.append(topic)
        return matched_topics

    def _extract_quarters(self, query_lower):
        found_quarters = []
        for quarter, patterns in quarter_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                found_quarters.append(quarter)
        return found_quarters

    def _extract_explicit_fiscal_years(self, query_lower):
        years = []
        fy_matches = re.findall(r"\bfy[\s_-]?(\d{2,4})\b", query_lower)
        for year in fy_matches:
            if len(year) == 2:
                years.append(f"FY{year}")
            else:
                years.append(f"FY{year[-2:]}")

        fiscal_matches = re.findall(r"\bfiscal[\s_-]?(\d{4})\b", query_lower)
        for year in fiscal_matches:
            years.append(f"FY{year[-2:]}")

        normalized = []
        for year in years:
            year = year.upper()
            if year not in normalized:
                normalized.append(year)
        return normalized

    def _get_requested_temporal_years(self, query_lower, explicit_years):
        if explicit_years:
            return explicit_years

        if "last 2 years" in query_lower or "past 2 years" in query_lower:
            return available_fiscal_years[:2]

        if "last 3 years" in query_lower or "past 3 years" in query_lower:
            return available_fiscal_years[:3]

        is_temporal = any(word in query_lower for word in temporal_words)
        if is_temporal:
            return available_fiscal_years

        return []
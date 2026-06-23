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
    "business",
    "company",
    "do",
    "overview",
    "services",
    "operations",
]

metric_keywords = [
    "attrition",
    "profit",
    "ebitda",
    "margin",
    "revenue",
]

metric_retrieval_map = {
    "revenue": "revenue revenue from operations total income turnover sales consolidated revenue standalone revenue IT services revenue",
    "profit": "profit net profit PAT profit after tax earnings profit attributable to equity holders net income profit for the year",
    "ebitda": "ebitda earnings before interest tax depreciation amortization operating profit",
    "attrition": "attrition voluntary attrition employee turnover workforce attrition annual attrition rate",
    "margin": "margin operating margin EBITDA margin EBIT margin profit margin net profit margin gross margin operating income margin",
}

temporal_words = [
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

# newest -> oldest
available_fiscal_years = ["FY26", "FY25", "FY24", "FY23"]


class QueryDecomposer:
    def decompose(self, query, memory_companies=None):
        query_lower = query.lower()

        companies = self._find_companies(query_lower)

        if not companies and memory_companies:
            companies = memory_companies

        metrics = self._find_metrics(query_lower)

        is_comparison = any(word in query_lower for word in comparison_words)
        is_description = any(word in query_lower for word in description_words)

        explicit_years = self._extract_explicit_fiscal_years(query_lower)
        requested_years = self._get_requested_temporal_years(
            query_lower,
            explicit_years
        )

        # Comparison + metric
        if companies and is_comparison and metrics:
            subqueries = []

            for company in companies:
                company_text = company_retrieval_map.get(company, company)

                for metric in metrics:
                    metric_text = metric_retrieval_map.get(metric, metric)

                    if requested_years:
                        for fiscal_year in requested_years:
                            subqueries.append(
                                f"{company_text} {metric_text} {fiscal_year}"
                            )
                    else:
                        subqueries.append(
                            f"{company_text} {metric_text}"
                        )

            return subqueries

        # Comparison without explicit metric -> default to revenue
        if companies and is_comparison:
            subqueries = []

            for company in companies:
                company_text = company_retrieval_map.get(company, company)
                metric_text = metric_retrieval_map["revenue"]

                if requested_years:
                    for fiscal_year in requested_years:
                        subqueries.append(
                            f"{company_text} {metric_text} {fiscal_year}"
                        )
                else:
                    subqueries.append(
                        f"{company_text} {metric_text}"
                    )

            return subqueries

        # Multi-company business description
        if len(companies) > 1 and is_description:
            subqueries = []

            for company in companies:
                company_text = company_retrieval_map.get(company, company)
                subqueries.append(
                    f"{company_text} business overview services operations company profile"
                )

            return subqueries

        # Single-company temporal metric query
        if len(companies) == 1 and metrics and requested_years:
            company = companies[0]
            company_text = company_retrieval_map.get(company, company)

            subqueries = []

            for metric in metrics:
                metric_text = metric_retrieval_map.get(metric, metric)

                for fiscal_year in requested_years:
                    subqueries.append(
                        f"{company_text} {metric_text} {fiscal_year}"
                    )

            return subqueries

        # Multi-company temporal metric query
        if len(companies) > 1 and metrics and requested_years:
            subqueries = []

            for company in companies:
                company_text = company_retrieval_map.get(company, company)

                for metric in metrics:
                    metric_text = metric_retrieval_map.get(metric, metric)

                    for fiscal_year in requested_years:
                        subqueries.append(
                            f"{company_text} {metric_text} {fiscal_year}"
                        )

            return subqueries

        # Single-company metric query
        if len(companies) == 1 and metrics:
            company = companies[0]
            company_text = company_retrieval_map.get(company, company)

            subqueries = []

            for metric in metrics:
                metric_text = metric_retrieval_map.get(metric, metric)
                subqueries.append(f"{company_text} {metric_text}")

            return subqueries

        # Single-company business description
        if len(companies) == 1 and is_description:
            company = companies[0]
            company_text = company_retrieval_map.get(company, company)

            return [
                f"{company_text} business overview services operations company profile"
            ]

        return [query]

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

        seen = set()
        normalized = []

        for year in years:
            year = year.upper()
            if year not in seen:
                seen.add(year)
                normalized.append(year)

        return normalized

    def _get_requested_temporal_years(self, query_lower, explicit_years):
        if explicit_years:
            return explicit_years

        if "last 2 years" in query_lower or "past 2 years" in query_lower:
            return available_fiscal_years[:2]

        if "last 3 years" in query_lower or "past 3 years" in query_lower:
            return available_fiscal_years[:3]

        if "last 5 years" in query_lower or "past 5 years" in query_lower:
            return available_fiscal_years[:5]

        is_temporal = any(word in query_lower for word in temporal_words)
        if is_temporal:
            return available_fiscal_years

        return []

    def _find_companies(self, query_lower):
        companies = []

        for canonical_company, aliases in company_aliases.items():
            if canonical_company in companies:
                continue

            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", query_lower):
                    companies.append(canonical_company)
                    break

        return companies

    def _find_metrics(self, query_lower):
        return [
            metric
            for metric in metric_keywords
            if metric in query_lower
        ]
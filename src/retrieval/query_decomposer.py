import re


company_aliases = {
    "mazagon dock shipbuilders": "mazdock",
    "mazagon dockship": "mazdock",
    "mazagondockship": "mazdock",
    "mazagon dock": "mazdock",
    "mazagondock": "mazdock",
    "mazagon": "mazdock",
    "mazdock": "mazdock",
    "tcs": "tcs",
    "infosys": "infosys",
    "wipro": "wipro",
}

company_retrieval_map = {
    "tcs": "Tata Consultancy Services TCS",
    "infosys": "Infosys",
    "wipro": "Wipro",
    "mazdock": "Mazagon Dock Shipbuilders Mazagon Dock MDL",
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
    "attrition": "attrition voluntary attrition employee turnover workforce attrition excluding DOP annual attrition rate",
    "margin": "margin operating margin EBITDA margin EBIT margin profit margin net profit margin gross margin operating income margin net income to turnover",
}
temporal_words = [
    "by year",
    "year wise",
    "year-wise",
    "over the years",
    "across years",
    "last 3 years",
    "last 5 years",
    "past 3 years",
    "past 5 years",
    "trend",
    "historical",
    "history",
    "over time",
    "yearly",
    "fy24",
    "fy23",
    "fy22",
    "fy21",
    "fy20",
    "fiscal 2024",
    "fiscal 2023",
    "fiscal 2022",
    "fiscal 2021",
    "fiscal 2020",
]


class QueryDecomposer:

    def decompose(self, query, memory_companies=None):
        query_lower = query.lower()

        companies = self._find_companies(query_lower)

        if not companies and memory_companies:
            companies = memory_companies

        metrics = self._find_metrics(query_lower)

        is_comparison = any(
            word in query_lower
            for word in comparison_words
        )

        is_description = any(
            word in query_lower
            for word in description_words
        )

        is_temporal = any(
            word in query_lower
            for word in temporal_words
        )

        if companies and is_comparison and metrics:
            subqueries = []

            for company in companies:
                company_text = company_retrieval_map.get(
                    company,
                    company
                )

                for metric in metrics:
                    metric_text = metric_retrieval_map.get(
                        metric,
                        metric
                    )

                    subqueries.append(
                        f"{company_text} {metric_text}"
                    )

            return subqueries

        if companies and is_comparison:
            subqueries = []

            for company in companies:
                company_text = company_retrieval_map.get(
                    company,
                    company
                )

                metric_text = metric_retrieval_map["revenue"]

                subqueries.append(
                    f"{company_text} {metric_text}"
                )

            return subqueries

        if len(companies) > 1 and is_description:
            subqueries = []

            for company in companies:
                company_text = company_retrieval_map.get(
                    company,
                    company
                )

                subqueries.append(
                    f"{company_text} business overview services operations company profile"
                )

            return subqueries

        if len(companies) == 1 and metrics and is_temporal:
            company = companies[0]
            company_text = company_retrieval_map.get(
                company,
                company
            )

            subqueries = []

            for metric in metrics:
                metric_text = metric_retrieval_map.get(
                    metric,
                    metric
                )

                subqueries.append(
                    f"{company_text} {metric_text} year wise yearly historical trend fiscal 2024 fiscal 2023 fiscal 2022 fiscal 2021 fiscal 2020"
                )

            return subqueries

        if len(companies) > 1 and metrics and is_temporal:
            subqueries = []

            for company in companies:
                company_text = company_retrieval_map.get(
                    company,
                    company
                )

                for metric in metrics:
                    metric_text = metric_retrieval_map.get(
                        metric,
                        metric
                    )

                    subqueries.append(
                        f"{company_text} {metric_text} year wise yearly historical trend fiscal 2024 fiscal 2023 fiscal 2022 fiscal 2021 fiscal 2020"
                    )

            return subqueries

        if len(companies) == 1 and metrics:
            company = companies[0]
            company_text = company_retrieval_map.get(
                company,
                company
            )

            subqueries = []

            for metric in metrics:
                metric_text = metric_retrieval_map.get(
                    metric,
                    metric
                )

                subqueries.append(
                    f"{company_text} {metric_text}"
                )

            return subqueries

        if len(companies) == 1 and is_description:
            company = companies[0]
            company_text = company_retrieval_map.get(
                company,
                company
            )

            return [
                f"{company_text} business overview services operations company profile"
            ]

        return [query]

    def _find_companies(self, query_lower):
        companies = []

        for alias, company in company_aliases.items():
            if company in companies:
                continue

            if re.search(
                rf"\b{re.escape(alias)}\b",
                query_lower
            ):
                companies.append(company)

        return companies

    def _find_metrics(self, query_lower):
        return [
            metric
            for metric in metric_keywords
            if metric in query_lower
        ]
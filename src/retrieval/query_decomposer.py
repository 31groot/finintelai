import re

class QueryDecomposer:

    def decompose(self, query):

        query_lower = query.lower()

        comparison_words = [
            "compare",
            "rank",
            "highest",
            "lowest",
            "better"
        ]

        if not any(word in query_lower for word in comparison_words):
            return [query]

        companies = []

        for company in [
            "tcs",
            "infosys",
            "wipro",
            "mazdock"
        ]:
            if company in query_lower:
                companies.append(company)

        metric = "revenue"

        if "attrition" in query_lower:
            metric = "attrition"

        elif "profit" in query_lower:
            metric = "profit"

        elif "ebitda" in query_lower:
            metric = "ebitda"

        if companies:
            return [
                f"{company} {metric}"
                for company in companies
            ]

        return [query]
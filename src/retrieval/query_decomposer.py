import re
class QueryDecomposer:
    def decompose(self, query):
        query_lower = query.lower()
        companies = []
        company_map = {
            "tcs": "tcs",
            "infosys": "infosys",
            "wipro": "wipro",
            "mazdock": "mazdock",
            "mazagon": "mazdock",
            "mazagon dock": "mazdock",
            "mazagon dock shipbuilders": "mazdock"
        }

        company_map = {
          "tcs": "tcs",
          "infosys": "infosys",
          "wipro": "wipro",
          "mazdock": "mazdock",
          "mazagon": "mazdock",
          "mazagon dock": "mazdock",
          "mazagon dock shipbuilders": "mazdock",
          "mazagondock": "mazdock",
          "mazagondockship": "mazdock",
          "mazagon dockship": "mazdock"

        }

        for alias, company in company_map.items():
            if alias in query_lower and company not in companies:
                companies.append(company)

        comparison_words = [
            "compare",
            "rank",
            "highest",
            "lowest",
            "better"
        ]

        if any(word in query_lower for word in comparison_words):
            metric = "revenue"
            if "attrition" in query_lower:
                metric = "attrition"
            elif "profit" in query_lower:
                metric = "profit"
            elif "ebitda" in query_lower:
                metric = "ebitda"
            elif "margin" in query_lower:
                metric = "margin"

            if companies:
                return [
                    f"{company} {metric}"
                    for company in companies
                ]

        description_words = [
            "what does",
            "business",
            "company",
            "do",
            "overview"
        ]

        if (
            len(companies) > 1 and
            any(
                word in query_lower
                for word in description_words
            )
        ):
            return [
                f"what does {company} do"
                for company in companies
            ]

        return [query]
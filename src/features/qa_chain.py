from dotenv import load_dotenv
from groq import Groq, GroqError
from src.features.rag import RAGPipeline

load_dotenv()


class QAChain:

    def __init__(self):
        self.rag = RAGPipeline()
        self.client = Groq()
        self.memory = {"companies": []}

    def _needs_company_clarification(self, query):
        query_lower = query.lower()

        companies = self.rag.decomposer._find_companies(query_lower)
        metrics = self.rag.decomposer._find_metrics(query_lower)

        has_memory = bool(self.memory["companies"])

        description_words = [
            "what does",
            "business",
            "company",
            "do",
            "overview",
            "services",
            "operations",
        ]

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

        is_description = any(
            word in query_lower
            for word in description_words
        )

        is_comparison = any(
            word in query_lower
            for word in comparison_words
        )

        is_temporal = any(
            word in query_lower
            for word in temporal_words
        )

        if companies:
            return False

        if has_memory:
            return False

        if metrics or is_description or is_comparison or is_temporal:
            return True

        return False

    def ask(self, query):
        if self._needs_company_clarification(query):
            return (
                "Which company do you want this for — "
                "TCS, Infosys, Wipro, or Mazagon Dock?"
            )

        current_companies = self.rag.decomposer._find_companies(
            query.lower()
        )

        if current_companies:
            self.memory["companies"] = current_companies

        result = self.rag.get_context(
            query,
            memory_companies=self.memory["companies"]
        )

        context = result["context"]
        sources = result["metadata"]

        prompt = f"""
You are an expert financial analyst.
Use ONLY the information provided in the context.

IMPORTANT RULES:
- Never use outside knowledge.
- Never invent numbers.
- Never estimate missing values.
- If a value is missing, explicitly state that it is not available in the context.
- When multiple companies appear, extract information for ALL companies before answering.
- Never calculate a financial or HR metric unless the context explicitly provides all required components and the question asks for a calculation.
- For attrition, EBITDA margin, profit margin, or similar metrics, do NOT derive the value from employee counts or other proxy values unless the report explicitly states the metric or explicitly provides the exact formula inputs for that metric.
- If the requested metric is not explicitly available in the context, say it is not available in the retrieved context. Do not infer it.

FOR COMPARISON QUESTIONS:
1. Identify every company mentioned.
2. Extract the requested metric for each company.
3. Extract the numerical value.
4. Extract the unit and currency for every value.
5. Create a comparison table.
6. Before ranking, verify that all companies use the SAME unit and SAME currency.

COMPARISON VALIDATION:
- If all companies use the same unit and currency:
  - Compare the values numerically.
  - Rank companies from highest to lowest.
- If one or more companies use a different unit or currency:
  - Explicitly state which companies cannot be directly compared.
  - Explain the unit/currency mismatch.
  - Do NOT rank those companies together.
  - Only rank companies that share the same unit and currency.

Examples:

Correct:
TCS: ₹240,893 crore
Infosys: ₹153,670 crore
Ranking:
1. TCS
2. Infosys

Correct:
TCS: ₹240,893 crore
Infosys: ₹153,670 crore
Wipro: ₹897,943 million
Wipro cannot be directly compared with TCS and Infosys because the reported unit differs (million vs crore). Therefore a reliable ranking cannot be produced without unit normalization.

FOR KPI QUESTIONS:
1. Extract the KPI value.
2. State the company.
3. State the reporting period if available.
4. Include the reported unit.
5. Provide a concise explanation.

OUTPUT FORMAT FOR COMPARISONS:
| Company | Metric | Unit/Currency |
|---------|--------|---------------|
| Company A | Value | Unit |
| Company B | Value | Unit |

Unit Validation:
...

Ranking:
...

Explanation:
...

Context:
{context}

Question:
{query}

Answer:
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
        except GroqError as e:
            return (
                "Sorry, I couldn't reach the AI service right now "
                f"({type(e).__name__}). Please try again in a moment."
            )

        answer = response.choices[0].message.content

        citations = "\n\nSources:\n"
        seen = set()
        count = 0

        for item in sources:
            source = item.get("source")
            page = item.get("page")
            key = (source, page)

            if key not in seen:
                citations += f"- {source} (page {page})\n"
                seen.add(key)
                count += 1

            if count >= 15:
                break

        return answer + citations
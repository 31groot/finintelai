from dotenv import load_dotenv
from groq import Groq
from src.features.rag import RAGPipeline

load_dotenv()


class QAChain:
    def __init__(self):
        self.rag = RAGPipeline()
        self.client = Groq()

    def ask(self, query):
        result = self.rag.get_context(query)
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

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
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

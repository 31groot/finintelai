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
3. Create a comparison table.
4. Rank companies from highest to lowest whenever numerical values are available.
5. Explain the ranking briefly.

FOR KPI QUESTIONS:

1. Extract the KPI value.
2. State the company.
3. State the reporting period if available.
4. Provide a concise explanation.

OUTPUT FORMAT FOR COMPARISONS:

| Company | Metric |
|----------|----------|
| Company A | Value |
| Company B | Value |

Ranking:
1. ...
2. ...
3. ...

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
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

        # ---------------------------------
        # Source Attribution
        # ---------------------------------

        citations = "\n\nSources:\n"

        seen = set()
        count = 0

        for item in sources:

            source = item.get("source")
            page = item.get("page")

            key = (source, page)

            if key not in seen:

                citations += (
                    f"- {source} "
                    f"(page {page})\n"
                )

                seen.add(key)

                count += 1

                # Limit displayed sources
                if count >= 15:
                    break

        return answer + citations
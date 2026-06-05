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

Use ONLY the information present in the provided context.

Rules:

1. Never use outside knowledge.

2. If information is missing from the context, explicitly say so.

3. If the question asks for:
   - comparison
   - ranking
   - highest
   - lowest
   - best
   - worst
   - compare
   - rank

   then:

   a. Extract all relevant company values from the context.

   b. Combine information across multiple chunks.

   c. Compare the values numerically.

   d. Return a ranked list whenever possible.

4. Do NOT say "only one company is mentioned"
   unless there is truly only one company
   in the provided context.

5. If multiple companies appear in the context,
   identify each company and its corresponding value
   before answering.

6. For revenue questions:

   - Extract revenue figures for every company found.

   - Rank companies from highest revenue to lowest revenue.

7. Think step-by-step:

   - Identify companies.
   - Extract values.
   - Compare values.
   - Then answer.

8. Show calculations and comparisons when relevant.

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

        # =========================
        # SOURCE ATTRIBUTION
        # =========================

        citations = "\n\nSources:\n"

        seen = set()

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

        return answer + citations
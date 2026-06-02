from dotenv import load_dotenv
import os
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
    

        prompt = f"""
You are a financial analyst.

Use ONLY the provided context.

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
        return answer
        
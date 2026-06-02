from src.features.rag import RAGPipeline

rag = RAGPipeline()

query = "What was Infosys attrition in FY24?"

context = rag.get_context(query)

print(context[:3000])
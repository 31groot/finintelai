from src.features.rag import RAGPipeline
_rag = None

def get_pipeline():
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
    return _rag

def get_retriever():
    return get_pipeline().retriever

def get_reranker():
    return get_pipeline().reranker
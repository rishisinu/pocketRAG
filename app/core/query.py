from app.core.ingestion import _embedding_model, _embedding_fn
from app.core.indexing import lib
from langchain_core.documents import Document



def query_handler(query: str):
    embed_model = _embedding_model

    chunk_list: list[tuple[Document, float]] = lib.similarity_search_with_score(query, k=4)

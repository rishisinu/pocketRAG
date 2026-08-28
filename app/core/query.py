from app.core.ingestion import _embedding_model, _embedding_fn
from app.core.indexing import lib
from langchain_core.documents import Document
from app.core.models import Chunk
from app.core.bm25 import bm25_search
from sentence_transformers import CrossEncoder
#Implementing reranker model here aswell.
def query_handler(query: str):
    embed_model = _embedding_model

    faiss_list: list[tuple[Document, float]] = lib.similarity_search_with_score(query, k=10)
    bm25_list: list[tuple[Chunk, float]] = bm25_search(query, k=10)

    final_list: set[Chunk] = set()
    # both same len
    for i in range(len(faiss_list)):
        doc = faiss_list[i][0]
        faiss_chunk = Chunk(
            doc_id=doc.metadata["doc_id"],
            chunk_id=doc.metadata["chunk_id"],
            text=doc.page_content,
            source=doc.metadata["source"],
            page=doc.metadata["page"],
            chunk_index=doc.metadata["chunk_index"],
        )
        if faiss_chunk not in final_list:
            final_list.add(faiss_chunk)
        if bm25_list[i][0] not in final_list:
            final_list.add(bm25_list[i][0])

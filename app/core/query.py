import app.core.indexing as indexing
from langchain_core.documents import Document
from app.core.models import Chunk
from app.core.bm25 import bm25_search
from sentence_transformers import CrossEncoder
import numpy as np

model_name = 'cross-encoder/ms-marco-MiniLM-L6-v2'
model = None
#Implementing reranker model here aswell.
def query_handler(query: str):


    faiss_list: list[tuple[Document, float]] = indexing.lib.similarity_search_with_score(query, k=10)
    bm25_list: list[tuple[Chunk, float]] = bm25_search(query, k=10)

    final_list: dict[str, Chunk] = {}
    for doc, _ in faiss_list:
        if doc.metadata["chunk_id"] not in final_list:
            final_list[doc.metadata["chunk_id"]] = Chunk(
                doc_id=doc.metadata["doc_id"],
                chunk_id=doc.metadata["chunk_id"],
                text=doc.page_content,
                source=doc.metadata["source"],
                page=doc.metadata["page"],
                chunk_index=doc.metadata["chunk_index"],
            )

    for chunk, _ in bm25_list:
        if chunk.chunk_id not in final_list:
            final_list[chunk.chunk_id] = chunk

    chunk_ids = list(final_list.keys())
    predict_list = [[query, final_list[cid].text] for cid in chunk_ids]

    instantiate_model(model_name)
    assert model is not None
    scores = model.predict(predict_list)
    scores = 1 / (1 + np.exp(-scores))

    ranked_chunks: list[tuple[Chunk, float]] = sorted(
        (
            (final_list[cid], float(score))
            for cid, score in zip(chunk_ids, scores)
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )

    return ranked_chunks


def instantiate_model(model_name: str):
    global model

    if model is None:
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

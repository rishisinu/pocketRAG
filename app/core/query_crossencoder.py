import app.core.indexing as indexing
from langchain_core.documents import Document
from app.core.models import Chunk
from app.core.bm25 import bm25_search
from sentence_transformers import CrossEncoder

model_name = 'cross-encoder/ms-marco-MiniLM-L6-v2'
model = None
#Implementing reranker model here aswell.
amt_of_chunks = 20  # candidates pulled from EACH of faiss and bm25 before reranking
def query_handler(query: str):


    faiss_list: list[tuple[Document, float]] = indexing.lib.similarity_search_with_score(query, k=amt_of_chunks)
    bm25_list: list[tuple[Chunk, float]] = bm25_search(query, k=amt_of_chunks)

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
    # Keep the RAW LOGITS. The sigmoid is applied only where a score is shown
    # (the prompt and the citation cards), never in the pipeline itself.
    #
    # Why: this reranker is trained with a ranking loss, so only the ORDER of
    # its scores is meaningful, and the absolute scale drifts with question
    # wording. Sigmoid then saturates everything past about +/-6, which
    # collapses the entire low end into 0.0000 - measured, an irrelevant chunk
    # (logit -11.4) and a relevant chunk the model merely worded differently
    # (logit -6.5) both render as 0.0000. The guard's abstain gate reads the
    # SHAPE of this distribution, so it needs the logits intact.
    scores = model.predict(predict_list)

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
        model = CrossEncoder(f'{model_name}')

from app.core.models import Chunk, IngestResult
import numpy as np
from numpy.typing import NDArray
import torch
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
import faiss


lib_name = "my_faiss_index"
lib = None

def add_to_index(
    all_chunks: list[Chunk],
    embeddings: list[torch.Tensor] | NDArray[Any] | torch.Tensor, model
) -> IngestResult:
    if len(all_chunks) == 0:
        return IngestResult(doc_id = "", filename = "", num_chunks = 0, status = "error", error = "Empty All_chunks")

    global lib
    metadata = []
    text = []
    ids = []
    doc_id = all_chunks[0].doc_id
    ch_count = 0
    filename = all_chunks[0].source
    for c,emb in zip(all_chunks, embeddings):
        metadata.append({"doc_id": c.doc_id, "chunk_id": c.chunk_id, "source": c.source, "page": c.page, "chunk_index": c.chunk_index})
        text.append((c.text, emb.tolist()))
        ids.append(c.chunk_id)
        ch_count+=1

    assert lib is not None

    try:
        lib.add_embeddings(
            text_embeddings=text,
            metadatas = metadata,
            ids = ids
        )
        return IngestResult(doc_id = doc_id, filename = filename, num_chunks = ch_count, status = "success")
    except Exception as e:
        return IngestResult(doc_id = doc_id, filename = filename, num_chunks = ch_count, status = "error", error = f"{e}")

def instantiate_index(model):
    global lib
    if lib is None:
        ind = faiss.IndexFlatL2(384)
        lib = FAISS(
            embedding_function = model,
            index = ind,
            docstore = InMemoryDocstore(),
            index_to_docstore_id={}
        )

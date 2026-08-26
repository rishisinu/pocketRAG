from app.core.models import Chunk, IngestResult
import numpy as np
from numpy.typing import NDArray
import torch
from typing import Any
from langchain_community.vectorstores import FAISS
import os
import faiss


lib_name = "my_faiss_index"
lib = None

def add_to_index(
    all_chunks: list[Chunk],
    embeddings: list[torch.Tensor] | NDArray[Any] | torch.Tensor, model
) -> IngestResult:
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


    lib = FAISS.from_embeddings(text, model, metadatas=metadata, ids=ids)

    try:
        index_path = os.path.join("my_faiss_index", "index.faiss")
        pkl_path = os.path.join("my_faiss_index", "index.pkl")

        if os.path.exists(index_path) and os.path.exists(pkl_path):
            print("Vector store successfully saved.")
        return IngestResult(doc_id = doc_id, filename = filename, num_chunks = ch_count, status = "success")
    except Exception as e:
        return IngestResult(doc_id = doc_id, filename = filename, num_chunks = ch_count, status = "error", error = f"{e}")

def instantiate_index():
    global lib
    if lib is None:
        lib = faiss.IndexFlatL2(384)

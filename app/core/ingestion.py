from pathlib import Path
from uuid import uuid4
import math
from docx import Document
import docx
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.models import Chunk, IngestResult
from app.core.indexing import add_to_index, instantiate_index #need to implement this
from app.core.bm25 import add_to_bm25




SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx",".md"}
chunk_size = 512
chunk_overlap = 102
model_name = "all-MiniLM-L6-v2"
_embedding_model: SentenceTransformer | None = None
_embedding_fn: HuggingFaceEmbeddings | None = None

def load_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model

def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = HuggingFaceEmbeddings(model_name=model_name)
    return _embedding_fn



def process_and_store_data(file_path: str) -> IngestResult:
    path = Path(file_path)
    doc_id = str(uuid4())

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return IngestResult(
            doc_id=doc_id,
            filename=path.name,
            num_chunks=0,
            status="error",
            error=f"Unsupported data type chud: {path.suffix}"
        )

    pages = load_document(path) #list str of pages, now embedding needs to happen
    all_chunks: list[Chunk] = []
    for i in range(len(pages)):

        page = pages[i]
        step = chunk_size - chunk_overlap
        num_chunks = ((len(page) + step - 1) // step)

        for j in range(num_chunks):
            if j == 0:
                starting_ind = 0
                ending_ind = chunk_size
            else:
                starting_ind = j*chunk_size - (j*chunk_overlap)
                ending_ind = starting_ind + chunk_size
            chunk_id = str(uuid4())
            all_chunks.append(Chunk(doc_id=doc_id, chunk_id = chunk_id, text = page[starting_ind:ending_ind], source = path.name, page = i, chunk_index=j))

    model = load_model()
    embeddings = model.encode([c.text for c in all_chunks])
    instantiate_index(get_embedding_model()) #lowk could just instantite in the indexing pipeline itself
    ing_res = add_to_index(all_chunks, embeddings, get_embedding_model())
    add_to_bm25(all_chunks)

    return ing_res




def load_document(file_path:Path) -> list[str]:
    suffix = file_path.suffix
    if suffix == ".pdf":
        return _load_pdf(file_path)
    elif suffix == ".docx":
        return _load_docx(file_path)
    elif suffix == ".txt":
        return _load_txt(file_path)
    elif suffix == ".md":
        return _load_md(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")

def _load_pdf(file_path:Path) -> list[str]:
    pages = []
    with fitz.open(file_path) as doc:
        for page in doc:
            pages.append(fitz.utils.get_text(page))
    return pages

def _load_docx(file_path:Path) -> list[str]:
    docs = docx.Document(str(file_path))
    pages = [p.text for p in docs.paragraphs if p.text.strip()]
    return ["\n".join(pages)]
def _load_txt(file_path:Path) -> list[str]:
    pages = []
    with open(file_path, "r", encoding='utf-8') as text:
        pages.append(str(text.read()))
    return pages

def _load_md(file_path: Path) -> list[str]:
    pages = []
    with open(file_path, "r", encoding='utf-8') as text:
        pages.append(str(text.read()))
    return pages

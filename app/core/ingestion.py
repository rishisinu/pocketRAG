from pathlib import Path
from uuid import uuid4

from docx import Document
import docx
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.models import Chunk, IngestResult
from app.core.indexing import add_to_index #need to implement this



SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx",".md"}
chunck_size = 512
chunk_overlap = 102
model_name = "#all-MiniLM-L6-v2"
_embedding_model: SentenceTransformer | None = None

def load_model():
    if _embedding_model is None:



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
    chunks = []
    for page in pages:


        for i in range(len(page)):




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
    return list[str]

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

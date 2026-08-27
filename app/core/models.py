from typing import Literal

from pydantic import BaseModel


class Chunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    source: str
    page: int | None = None
    chunk_index: int


class IngestResult(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    status: Literal["success", "error"]
    error: str | None = None


class QueryRequest(BaseModel):
    text: str

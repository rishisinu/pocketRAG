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
    top_k: int = 5  # how many reranked chunks actually get fed to the LLM


class Citation(BaseModel):
    marker: int  # the [n] the LLM is told to cite with
    doc_id: str
    chunk_id: str
    source: str
    page: int | None = None
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]

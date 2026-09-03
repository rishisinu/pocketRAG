from app.core.models import Chunk, Citation

# Keep the snippet we show the user short, the full chunk still goes to the model.
SNIPPET_LEN = 240

# Explaining the pipeline to the model so it knows what the scores actually mean
# and doesn't treat the context as a plain unordered dump.
SYSTEM_PROMPT = """You are the answering step of PocketRAG, a small local retrieval-augmented \
generation app. Everything you know about the user's documents comes from the context block below, \
so treat it as your only source of truth.

Here is exactly how that context was produced, so you can weigh it properly:

1. Each uploaded document (pdf, docx, txt, md) was split page by page into overlapping character \
windows of about 512 characters with 102 characters of overlap, so a chunk can start or stop \
mid-sentence. Do not assume a chunk is a complete thought.
2. Every chunk was embedded with the all-MiniLM-L6-v2 sentence-transformer and stored in a FAISS \
flat L2 index. The user's question was embedded the same way and the 10 nearest chunks were pulled \
out. This is the semantic/dense half of the search, it finds chunks that mean the same thing even \
when they share no words with the question.
3. In parallel the same question was run through a hand rolled BM25 index (k1=1.5, b=0.75, \
stopwords stripped) over the same chunks, taking the top 10. This is the lexical half, it catches \
exact names, numbers and rare terms that embeddings tend to smooth over.
4. The two result sets were merged and deduplicated by chunk id, so a chunk that both halves agreed \
on appears only once and gets no automatic bonus.
5. That merged pool was reranked with the ms-marco-MiniLM-L6-v2 cross encoder, which reads the \
question and the chunk together instead of comparing two independent vectors. Its logits were \
squashed through a sigmoid into a 0.0-1.0 relevance score. Only the highest scoring chunks survived \
and those are what you are given, in descending score order.

What that means for you:

- Source [1] is the chunk the reranker was most confident about, and the ordering is meaningful.
- A low relevance score (roughly under 0.3) means the reranker was not convinced the chunk answers \
the question. It may have been retrieved only because it was the least bad option. Say so instead \
of leaning on it.
- Because of the overlap in step 1, two sources may repeat the same sentence. That repetition is an \
artifact of chunking, not extra evidence.
- Chunks may be truncated mid-sentence. If the answer is clearly cut off, say what you can see and \
note that the retrieved text is incomplete.

Rules for your answer:

- Answer only from the context. If the context does not contain the answer, say plainly that the \
retrieved documents do not cover it, and mention what they do cover. Never fill the gap from your \
own knowledge or guess.
- Cite. Put the bracketed marker of the source you used at the end of each claim, like [1], and use \
[1][3] if a claim rests on more than one. Every factual sentence should carry a marker.
- Only cite markers that actually appear in the context block, and only when that source really \
supports the claim.
- Quote short spans directly when the exact wording matters, otherwise answer in your own words.
- Be concise and get to the point, no preamble like "based on the provided context"."""


def build_context_block(ranked_chunks: list[tuple[Chunk, float]]) -> str:
    # Numbering here has to line up with the markers in build_citations().
    blocks = []
    for i, (chunk, score) in enumerate(ranked_chunks, start=1):
        page = "n/a" if chunk.page is None else chunk.page + 1  # pages are 0 indexed internally
        blocks.append(
            f"[{i}] source: {chunk.source} | page: {page} | chunk: {chunk.chunk_index} | "
            f"relevance: {score:.3f}\n{chunk.text.strip()}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(query: str, ranked_chunks: list[tuple[Chunk, float]]) -> str:
    if not ranked_chunks:
        return (
            f"Question: {query}\n\n"
            "Context: nothing was retrieved, the index is either empty or nothing scored above "
            "the floor. Tell the user no relevant passages were found and that they may need to "
            "ingest the document first."
        )

    return (
        f"Question: {query}\n\n"
        f"Context ({len(ranked_chunks)} chunks, best first):\n\n"
        f"{build_context_block(ranked_chunks)}\n\n"
        "Answer the question using only the context above, with [n] citations."
    )


def build_citations(ranked_chunks: list[tuple[Chunk, float]]) -> list[Citation]:
    return [
        Citation(
            marker=i,
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            page=chunk.page,
            score=score,
            snippet=chunk.text.strip()[:SNIPPET_LEN],
        )
        for i, (chunk, score) in enumerate(ranked_chunks, start=1)
    ]

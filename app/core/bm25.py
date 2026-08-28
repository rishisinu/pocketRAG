import math
import re
from collections import Counter

from app.core.models import Chunk

# BM25 tuning constants (standard defaults)
K1 = 1.5
B = 0.75

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Purely grammatical/functional words with no semantic contribution.
# Deliberately excludes negations (not, no, never, none...) and quantifiers
# (all, some, few, most...) since those can flip or narrow meaning.
STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but", "if", "then", "else",
    "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "once",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "mine", "yours", "ours", "theirs",
    "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves",
    "what", "which", "who", "whom",
    "as", "until", "while", "so", "than", "too", "very",
    "s", "t", "just", "don", "now",
}

# Inverted index: term -> {chunk_id: term_frequency_in_chunk}
inverted_index: dict[str, dict[str, int]] = {}
doc_len: dict[str, int] = {}
chunk_store: dict[str, Chunk] = {}
n = 0
total_len = 0


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def add_to_bm25(chunks: list[Chunk]) -> None:
    global n, total_len

    for chunk in chunks:
        tokens = tokenize(chunk.text)
        tf = Counter(tokens) # Makes freq map

        doc_len[chunk.chunk_id] = len(tokens)
        chunk_store[chunk.chunk_id] = chunk

        for term, freq in tf.items(): # walk thru the freq map and initialze our freq map(potential optimization is just constructing the freq map ourselves so we can do this in one pass)
            inverted_index.setdefault(term, {})[chunk.chunk_id] = freq

        n += 1
        total_len += len(tokens)


def bm25_search(query: str, k: int) -> list[tuple[Chunk, float]]:
    if n == 0:
        return []

    avgdl = total_len / n
    query_terms = set(tokenize(query))
    # Some math i found off docs lol
    scores: dict[str, float] = {}
    for term in query_terms:
        postings = inverted_index.get(term)
        if not postings:
            continue

        df = len(postings)
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

        for chunk_id, freq in postings.items():
            dl = doc_len[chunk_id]
            denom = freq + K1 * (1 - B + B * dl / avgdl)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (freq * (K1 + 1)) / denom

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
    return [(chunk_store[chunk_id], score) for chunk_id, score in ranked]

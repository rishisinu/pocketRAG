"""Hallucination guard.



  L0  abstain gate     - is retrieval good enough to bother asking the LLM?
  L1  marker validation- are the [n] citations real, and does every claim carry one?
  L2  NLI entailment   - does the cited chunk actually support the sentence?
"""

import math
import re
from collections.abc import Sequence

from app.core.models import Chunk, Guard, SentenceCheck, Verdict

from sentence_transformers import CrossEncoder

ABSTAIN_CLIFF = 0.4  # largest logit drop between adjacent ranked chunks
ABSTAIN_TOP1 = -10.5  # logit floor the reranker pins to when nothing matches
MODEL_NAME = 'cross-encoder/nli-distilroberta-base'

model = None

# Sentences shorter than this are almost always connective tissue, not claims.
MIN_CLAIM_WORDS = 5

MARKER_RE = re.compile(r"\[(\d+)\]")

# Split on sentence punctuation when the next thing starts a new sentence.
SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'\[])')

# The model is told to put its citation at the END of a claim, so "revenue grew.
# [2]" is a legal shape. The split above cuts before that marker and strands it
# on the NEXT sentence, which both loses the citation on the claim that earned
# it and makes the following sentence look cited when it isn't. This matches a
# fragment that opens with markers so we can hand them back.
LEADING_MARKERS_RE = re.compile(r"^((?:\s*\[\d+\])+)\s*")

# Hedges and meta-commentary, not factual claims. An uncited sentence matching
# one of these should not be flagged.
NON_CLAIM_RE = re.compile(
    r"^\s*(?:"
    r"the retrieved (?:documents?|passages?|text)|"
    r"(?:the )?(?:documents?|context|sources?) (?:do(?:es)? not|don't|doesn't)|"
    r"i (?:could not|couldn't|cannot|can't) find|"
    r"nothing in the|"
    r"here(?:'s| is) what"
    r")",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_END_RE.split(text.strip()) if s.strip()]

    merged: list[str] = []
    for part in parts:
        lead = LEADING_MARKERS_RE.match(part)
        if lead and merged:
            # trailing citation of the previous claim, give it back
            merged[-1] = f"{merged[-1]} {lead.group(1).strip()}"
            part = part[lead.end():].strip()
            if not part:
                continue
        merged.append(part)
    return merged


def extract_markers(sentence: str) -> list[int]:
    # dedup but keep the order the model wrote them in
    seen: dict[int, None] = {}
    for m in MARKER_RE.findall(sentence):
        seen.setdefault(int(m), None)
    return list(seen)


def is_claim(sentence: str) -> bool:
    """Whether this sentence is asserting something that needs evidence."""
    if NON_CLAIM_RE.match(sentence):
        return False
    if extract_markers(sentence):
        # the model cited it, so it is asserting something. Short claims are
        # still claims: "Revenue grew 14% [1]" is three words and fully factual.
        return True
    # Uncited. Only worth flagging if it is long enough to be carrying a claim
    # rather than being connective tissue ("Sure.", "Here you go.").
    return len(sentence.split()) >= MIN_CLAIM_WORDS


def sigmoid(logit: float) -> float:
    """Logit -> 0..1, for DISPLAY only. See the note above on why the pipeline
    itself must not carry sigmoid scores."""
    return 1.0 / (1.0 + math.exp(-logit))


def find_peak(scores: Sequence[float]) -> tuple[int, float]:
    """Find the cliff in a descending list of reranker logits.

    One pass, no allocation. Returns (how many chunks sit above the cliff,
    how big the cliff is). The count is only meaningful when the cliff is real
    - on a flat distribution the largest drop is noise, and this will happily
    report a "peak" of 7 chunks out of 8. Always gate on the cliff first via
    should_abstain().

    Precondition: scores is sorted descending, which query_handler guarantees.
    """
    n = len(scores)
    if n == 0:
        return 0, 0.0
    if n == 1:
        return 1, 0.0

    best_index = 0
    best_drop = float("-inf")
    for i in range(n - 1):
        drop = scores[i] - scores[i + 1]
        if drop > best_drop:
            best_drop = drop
            best_index = i
    return best_index + 1, best_drop


def should_abstain(scores: Sequence[float], cliff: float) -> bool:
    """True when retrieval found nothing worth sending to the LLM."""
    if not scores:
        return True
    return cliff < ABSTAIN_CLIFF and scores[0] < ABSTAIN_TOP1


def select_by_peak(
    ranked_chunks: list[tuple[Chunk, float]], top_k: int
) -> tuple[list[tuple[Chunk, float]], bool]:

    if not ranked_chunks:
        return [], True

    scores = [score for _, score in ranked_chunks]
    peak_size, cliff = find_peak(scores)

    if should_abstain(scores, cliff):
        return ranked_chunks[:top_k], True

    # top_k is the user's ceiling, the peak is usually well under it
    return ranked_chunks[: min(peak_size, top_k)], False


def roll_up(sentences: list[SentenceCheck]) -> Verdict:
    """Answer level verdict from the per sentence results.

    Pure aggregation, no model. L1 calls this over its own results, and L2
    calls it again after it has downgraded some sentences, so "partial" means
    the same thing in both layers.
    """
    # sentences marked "ok" are hedges and connective tissue. They assert
    # nothing, so they are evidence neither for nor against the answer.
    claims = [s for s in sentences if s.reason != "ok"]
    if not claims:
        # an answer made entirely of hedges ("the retrieved documents do not
        # cover this") is correct behaviour, not an ungrounded answer
        return "grounded"

    failed = [s for s in claims if not s.supported]
    if not failed:
        return "grounded"
    if len(failed) == len(claims):
        return "ungrounded"
    return "partial"


# ---------------------------------------------------------------------------
# L2: NLI entailment.
#
# Premise is the CHUNK, hypothesis is the model's SENTENCE. The order is not
# cosmetic - measured on the same two texts, the correct order scored
# entailment 0.9878 and the reversed order 0.0103, with no error either way.
#
# The model emits three classes and they are not redundant: contradiction is
# not 1 - entailment, because neutral absorbs the difference. "Your documents
# refute this" (contra 0.95) and "your documents don't mention it" (contra 0.02)
# can both sit at entailment ~0.01.
NLI_MODEL_NAME = "cross-encoder/nli-distilroberta-base"
ENTAIL_PASS = 0.5  # entailment at or above this counts as supported
CONTRA_HARD = 0.85  # contradiction above this is treated as a real conflict

# Measured blind spot: this model cannot reason about numbers. A TRUE sentence
# ("about a month" vs a chunk saying "within 30 days") scored contradiction
# 0.9449, and a FALSE one (comparing 90% against 70%) scored only 0.0460. When
# both texts carry digits and those digits differ, the contradiction score is
# unreliable in either direction, so it is not allowed to fail a sentence.
DIGITS_RE = re.compile(r"\d+")

_nli_model = None
_nli_index: tuple[int, int] | None = None


def instantiate_nli():
    """Lazy load. Kept out of the lifespan on purpose - this box is memory
    tight, and a query that abstains at L0 never needs these weights."""
    global _nli_model, _nli_index
    if _nli_model is None:
        _nli_model = CrossEncoder(NLI_MODEL_NAME, max_length=512)
        raw_labels = getattr(_nli_model.config, "id2label", None) or {}
        labels = {int(i): str(v).lower() for i, v in raw_labels.items()}
        order = [labels[i] for i in sorted(labels)]
        if "entailment" not in order or "contradiction" not in order:
            raise RuntimeError(f"{NLI_MODEL_NAME} has no NLI labels: {order}")
        # read the order rather than hardcoding: a model swap can reorder these
        # and a wrong index silently scores the opposite class
        _nli_index = (order.index("entailment"), order.index("contradiction"))
    return _nli_model, _nli_index


def softmax3(row: Sequence[float]) -> list[float]:
    top = max(row)
    exps = [math.exp(v - top) for v in row]
    total = sum(exps)
    return [e / total for e in exps]


def numeric_mismatch(premise: str, hypothesis: str) -> bool:
    """True when both texts carry digits and they disagree, which is exactly
    where this model's contradiction score stops being trustworthy."""
    a = set(DIGITS_RE.findall(premise))
    b = set(DIGITS_RE.findall(hypothesis))
    return bool(a and b and a != b)


def hypothesis_for(text: str) -> str:
    """Strip citation markers before scoring.

    The markers are pipeline syntax, not part of the claim, and the NLI model
    reads them as noise. Measured: the same sentence scored entailment 0.9552
    clean and 0.4093 with a trailing "[1]" left in, which is the difference
    between passing and being flagged.
    """
    stripped = MARKER_RE.sub("", text)
    # collapse the double spaces and the " ." the markers leave behind
    return re.sub(r"\s+([.!?,;:])", r"\1", re.sub(r"\s+", " ", stripped)).strip()


def check_entailment(
    sentences: list[SentenceCheck], selected: list[tuple[Chunk, float]]
) -> list[SentenceCheck]:
    """L2. Fills in `score` and can downgrade a sentence L1 passed.

    Sentences are handled by the reason L1 gave them:
      ok                        skipped, it asserts nothing
      unchecked                 checked against the chunks it cited
      bad_marker / no_citation  checked against ALL of `selected`, which
                                separates "true claim, wrong citation" from
                                "invented claim". Strict mode keeps them
                                unsupported either way, but the score says which.
    """
    if not selected:
        return sentences

    source_count = len(selected)
    pairs: list[list[str]] = []
    owner: list[tuple[int, int]] = []  # (sentence index, marker)

    for i, sentence in enumerate(sentences):
        if sentence.reason == "ok":
            continue
        if sentence.reason == "unchecked":
            marks = [m for m in sentence.markers if 1 <= m <= source_count]
        else:
            marks = list(range(1, source_count + 1))
        for mark in marks:
            pairs.append([selected[mark - 1][0].text, hypothesis_for(sentence.text)])
            owner.append((i, mark))

    if not pairs:
        return sentences

    model, index = instantiate_nli()
    assert index is not None
    entail_i, contra_i = index
    # one call for the whole answer, per sentence calls turn ~50ms into ~1s
    raw = model.predict(pairs, show_progress_bar=False)
    probs = [softmax3([float(v) for v in row]) for row in raw]

    for i, sentence in enumerate(sentences):
        rows = [k for k, (si, _) in enumerate(owner) if si == i]
        if not rows:
            continue

        # max entailment and max contradiction are taken INDEPENDENTLY. They
        # answer different questions and need not come from the same chunk: a
        # claim can be entailed by source 1 at 0.99 while source 2 refutes it
        # at 0.98, and reading contra off the best entailment row hides that.
        best_entail = max(probs[k][entail_i] for k in rows)
        conflicting = [
            owner[k][1]
            for k in rows
            if probs[k][contra_i] > CONTRA_HARD
            and not numeric_mismatch(pairs[k][0], pairs[k][1])
        ]

        sentence.score = best_entail

        if best_entail >= ENTAIL_PASS:
            # some source backs the claim. Per the current policy entailment
            # wins outright, so any source that disagrees is dropped from the
            # citation rather than failing the sentence.
            if conflicting:
                sentence.markers = [m for m in sentence.markers if m not in conflicting]
        elif sentence.reason == "unchecked":
            # nothing supports it. This also absorbs the hard contradiction
            # case, because the reason enum has no value for "refuted" yet.
            sentence.supported = False
            sentence.reason = "low_entailment"

    return sentences


def check_answer(answer: str, selected: list[tuple[Chunk, float]]) -> Guard:
    """Verify an answer against the chunks it was generated from.

    L1 and L2 are not wired in yet, every sentence comes back unchecked. The
    plumbing is here so the response shape is stable before the logic lands.
    """

    sentences = []

    for sentence in split_sentences(answer):

        markers = extract_markers(sentence)
        inv = False
        hasMarker = False
        for marker in markers:
            if marker > len(selected) or marker <= 0:
                inv = True

            hasMarker = True
        if inv:
            sentences.append(
                SentenceCheck(
                    text = sentence,
                    markers = markers,
                    supported = False,
                    score = None,
                    reason = "bad_marker"
                )
            )
        elif is_claim(sentence):
            if hasMarker:
                sentences.append(
                    SentenceCheck(
                        text = sentence,
                        markers = markers,
                        supported = True,
                        score = None,
                        reason = "unchecked"
                    )
                )
            else:
                sentences.append(
                    SentenceCheck(
                        text = sentence,
                        markers = [],
                        supported = False,
                        score = None,
                        reason = "no_citation"
                    )
                )
        else:
            sentences.append(
                SentenceCheck(
                    text = sentence,
                    markers = [],
                    supported = True,
                    score = None,
                    reason = "ok"
                )
            )

    sentences = check_entailment(sentences, selected)

    return Guard(verdict=roll_up(sentences), sentences=sentences)


def instantiate_model():
    global model
    if model is None:
        model = CrossEncoder(MODEL_NAME)

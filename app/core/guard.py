"""Hallucination guard.

Checks the model's answer against the chunks it was actually given, in layers,
cheapest first:

  L0  abstain gate     - is retrieval good enough to bother asking the LLM?
  L1  marker validation- are the [n] citations real, and does every claim carry one?
  L2  NLI entailment   - does the cited chunk actually support the sentence?

Nothing here rewrites the answer. The guard reports, the frontend renders the
doubt, and the user gets to see both the claim and our confidence in it.
"""

import math
import re
from collections.abc import Sequence

from app.core.models import Chunk, Guard, SentenceCheck, Verdict

# ---------------------------------------------------------------------------
# L0 thresholds, in LOGIT space. Never threshold the sigmoid here.
#
# The cross encoder is trained with a ranking loss, which is invariant to
# monotonic transforms of the score: only the ORDER is optimised, so the
# absolute scale drifts with question wording. Measured on one passage,
# "Is remote work permitted?" scored logit +9.8 and "Can I work from home?"
# scored -6.5, same passage, same meaning. A fixed floor therefore filters by
# phrasing, not by relevance.
#
# What IS stable is the shape of the distribution. An answerable query produces
# a cliff - a few chunks up top, a sharp drop, then the rest. An unanswerable
# one pins every chunk to the model's floor (~-11) together, so there is no
# cliff and the largest "drop" is just noise. Measured cliff sizes:
#   answerable    0.76 .. 7.83   (n=5, incl. a paraphrased query and a pool
#                                 where every retrieved chunk was on topic)
#   unanswerable  0.02 .. 0.10   (n=2)
#
# Both conditions are required to abstain. Cliff alone would falsely abstain
# whenever the retrieved pool is uniformly relevant; top1 alone was measured
# rejecting 3 of 6 answerable queries, because a correct-but-paraphrased query
# sits at logit -10.3 while a genuinely unanswerable one sits at -11.0.
#
# TODO: both come from a small synthetic set. Tune against the real corpus by
# logging (cliff, top1) for ~20 questions with known answers before trusting
# the abstain path.
ABSTAIN_CLIFF = 0.4  # largest logit drop between adjacent ranked chunks
ABSTAIN_TOP1 = -10.5  # logit floor the reranker pins to when nothing matches

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

    return Guard(verdict=roll_up(sentences), sentences=sentences)

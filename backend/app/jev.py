from typing import Any

from .schemas import JEVAction, JEVDecision

# Memory is considered sufficient to answer from alone when the best stored
# chunk matches the query at or above this cosine similarity.
REUSE_SIMILARITY_THRESHOLD = 0.55
# Below these evaluation dimensions the workflow takes a targeted corrective
# action instead of a generic refinement.
RELEVANCE_THRESHOLD = 0.5
COVERAGE_THRESHOLD = 0.5


class JEVDecisionEngine:
    """Decision boundary for the JEV implementation.

    Full action-set policy: initial decisions map observable memory state
    (size + best query similarity) to REUSE_EXISTING / REUSE_OR_ACQUIRE /
    ACQUIRE, with INSUFFICIENT_OR_ERROR as the safe fallback. Final decisions
    inspect evaluation dimensions to choose between APPROVE, RETRIEVE_MORE
    (evidence does not address the query), REGENERATE (answer is weakly
    grounded in its evidence), REFINE_RETRIEVAL, and
    STOP_INSUFFICIENT_EVIDENCE.
    """

    implementation = "jev-full-policy-v1"

    def decide_initial(
        self, *, memory_count: int, query: str, memory_match_score: float | None = None
    ) -> JEVDecision:
        try:
            if memory_count is None or memory_count < 0 or not query or not str(query).strip():
                return JEVDecision(
                    action=JEVAction.INSUFFICIENT_OR_ERROR,
                    explanation="JEV decision inputs were invalid; the decision is marked insufficient rather than guessed.",
                    confidence=None,
                    implementation=self.implementation,
                )
            if memory_count == 0:
                return JEVDecision(
                    action=JEVAction.ACQUIRE,
                    explanation="No indexed scientific memory is available for this query, so external literature acquisition is required.",
                    confidence=None,
                    implementation=self.implementation,
                )
            if memory_match_score is not None and memory_match_score >= REUSE_SIMILARITY_THRESHOLD:
                return JEVDecision(
                    action=JEVAction.REUSE_EXISTING,
                    explanation=(
                        f"Existing memory already covers this query (best chunk similarity "
                        f"{memory_match_score:.2f} >= {REUSE_SIMILARITY_THRESHOLD}); reuse stored evidence without new acquisition."
                    ),
                    confidence=None,
                    implementation=self.implementation,
                )
            return JEVDecision(
                action=JEVAction.REUSE_OR_ACQUIRE,
                explanation="Existing scientific memory is available; reuse it and supplement only if evaluation finds gaps.",
                confidence=None,
                implementation=self.implementation,
            )
        except Exception as exc:  # safe fallback: never guess on a broken decision input
            return JEVDecision(
                action=JEVAction.INSUFFICIENT_OR_ERROR,
                explanation=f"Initial JEV decision failed ({exc}); the query is marked insufficient rather than guessed.",
                confidence=None,
                implementation=self.implementation,
            )

    def decide_after_evaluation(
        self,
        *,
        papers_found: int,
        evaluation_sufficient: bool,
        evaluation: Any = None,
        acquisition_skipped: bool = False,
    ) -> JEVDecision:
        if evaluation_sufficient:
            return JEVDecision(
                action=JEVAction.APPROVE,
                explanation="The workflow has sufficient evidence according to the supplied evaluation state.",
                confidence=None,
                implementation=self.implementation,
            )
        if papers_found == 0:
            if acquisition_skipped:
                return JEVDecision(
                    action=JEVAction.RETRIEVE_MORE,
                    explanation="Memory-only reuse was insufficient for this query; acquire external literature now.",
                    confidence=None,
                    implementation=self.implementation,
                )
            return JEVDecision(
                action=JEVAction.STOP_INSUFFICIENT_EVIDENCE,
                explanation="No usable papers were returned; stop transparently rather than fabricate an answer.",
                confidence=None,
                implementation=self.implementation,
            )
        relevance = getattr(evaluation, "relevance", None)
        coverage = getattr(evaluation, "evidence_coverage", None)
        if coverage is not None and coverage < COVERAGE_THRESHOLD:
            return JEVDecision(
                action=JEVAction.REGENERATE,
                explanation=(
                    f"Evidence exists but the answer is weakly grounded in it "
                    f"(evidence coverage {coverage:.2f} < {COVERAGE_THRESHOLD}); regenerate the answer from the same evidence."
                ),
                confidence=None,
                implementation=self.implementation,
            )
        if relevance is not None and relevance < RELEVANCE_THRESHOLD:
            return JEVDecision(
                action=JEVAction.RETRIEVE_MORE,
                explanation=(
                    f"Retrieved literature does not address the query well "
                    f"(relevance {relevance:.2f} < {RELEVANCE_THRESHOLD}); acquire more relevant literature."
                ),
                confidence=None,
                implementation=self.implementation,
            )
        return JEVDecision(
            action=JEVAction.REFINE,
            explanation="Evidence was found but remains insufficient; a bounded refinement cycle is recommended.",
            confidence=None,
            implementation=self.implementation,
        )

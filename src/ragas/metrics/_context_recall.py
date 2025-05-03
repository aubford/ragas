from __future__ import annotations

import logging
import math
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel
from scipy.spatial.distance import cosine

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._string import DistanceMeasure, NonLLMStringSimilarity
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    ensembler,
    MetricWithEmbeddings,
)
from ragas.prompt import PydanticPrompt
from ragas.run_config import RunConfig
from ragas.utils import deprecated

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


logger = logging.getLogger(__name__)


class QCA(BaseModel):
    question: str
    context: str
    answer: str


class ContextRecallClassification(BaseModel):
    statement: str
    reason: str
    attributed: int


class ContextRecallClassifications(BaseModel):
    classifications: t.List[ContextRecallClassification]


class ContextRecallClassificationPrompt(
    PydanticPrompt[QCA, ContextRecallClassifications]
):
    name: str = "context_recall_classification"
    instruction: str = (
        "Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only 'Yes' (1) or 'No' (0) as a binary classification. Output json with reason."
    )
    input_model = QCA
    output_model = ContextRecallClassifications
    examples = [
        (
            QCA(
                question="What can you tell me about albert Albert Einstein?",
                context="Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",
                answer="Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895",
            ),
            ContextRecallClassifications(
                classifications=[
                    ContextRecallClassification(
                        statement="Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
                        reason="The date of birth of Einstein is mentioned clearly in the context.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="He received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
                        reason="The exact sentence is present in the given context.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="He published 4 papers in 1905.",
                        reason="There is no mention about papers he wrote in the given context.",
                        attributed=0,
                    ),
                    ContextRecallClassification(
                        statement="Einstein moved to Switzerland in 1895.",
                        reason="There is no supporting evidence for this in the given context.",
                        attributed=0,
                    ),
                ]
            ),
        ),
    ]


@dataclass
class LLMContextRecall(MetricWithLLM, SingleTurnMetric):
    """
    Estimates context recall by estimating TP and FN using annotated answer and
    retrieved context.

    Attributes
    ----------
    name : str
    """

    name: str = "context_recall"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "retrieved_contexts",
                "reference",
            }
        }
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS
    context_recall_prompt: PydanticPrompt = field(
        default_factory=ContextRecallClassificationPrompt
    )
    max_retries: int = 1

    @staticmethod
    def _compute_score(responses: t.List[ContextRecallClassification]) -> float:
        response = [1 if item.attributed else 0 for item in responses]
        denom = len(response)
        numerator = sum(response)
        score = numerator / denom if denom > 0 else np.nan

        if np.isnan(score):
            logger.warning("The LLM did not return a valid classification.")

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "set LLM before use"

        # run classification
        classifications_list: t.List[ContextRecallClassifications] = (
            await self.context_recall_prompt.generate_multiple(
                data=QCA(
                    question=row["user_input"],
                    context="\n".join(row["retrieved_contexts"]),
                    answer=row["reference"],
                ),
                llm=self.llm,
                callbacks=callbacks,
            )
        )
        classification_dicts = []
        for classification in classifications_list:
            classification_dicts.append(
                [clasif.model_dump() for clasif in classification.classifications]
            )

        ensembled_clasif = ensembler.from_discrete(classification_dicts, "attributed")

        return self._compute_score(
            [ContextRecallClassification(**clasif) for clasif in ensembled_clasif]
        )


@dataclass
class ContextRecall(LLMContextRecall):
    name: str = "context_recall"

    @deprecated(since="0.2", removal="0.3", alternative="LLMContextRecall")
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    @deprecated(since="0.2", removal="0.3", alternative="LLMContextRecall")
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


@dataclass
class NonLLMContextRecall(SingleTurnMetric):
    name: str = "non_llm_context_recall"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_contexts",
                "reference_contexts",
            }
        }
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS
    _distance_measure: SingleTurnMetric = field(
        default_factory=lambda: NonLLMStringSimilarity()
    )

    def init(self, run_config: RunConfig) -> None: ...

    @property
    def distance_measure(self) -> SingleTurnMetric:
        return self._distance_measure

    @distance_measure.setter
    def distance_measure(self, distance_measure: DistanceMeasure) -> None:
        self._distance_measure = NonLLMStringSimilarity(
            distance_measure=distance_measure
        )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """For each reference context, find the best matching retrieved context. Try both main content
        and summary for each reference document. Return the similarity score of the best match.
        Don't collapse to a binary score because we aren't looking for exact document match but overall
        amount of matching information.
        """
        retrieved_contexts = sample.retrieved_contexts
        reference_contexts = sample.reference_contexts
        assert retrieved_contexts is not None, "retrieved_contexts is empty"
        assert reference_contexts is not None, "reference_contexts is empty"

        scores = []
        for ref, summary in reference_contexts:
            assert ref, "reference is empty"

            max_vs_ref = max(
                [
                    await self.distance_measure.single_turn_ascore(
                        SingleTurnSample(reference=rc, response=ref), callbacks
                    )
                    for rc in retrieved_contexts
                ]
            )
            if summary:
                max_vs_summary = max(
                    [
                        await self.distance_measure.single_turn_ascore(
                            SingleTurnSample(reference=rc, response=summary), callbacks
                        )
                        for rc in retrieved_contexts
                    ]
                )
            else:
                max_vs_summary = 0

            scores.append(max(max_vs_ref, max_vs_summary))
        return self._compute_score(scores)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)

    @staticmethod
    def _compute_score(verdict_list: t.List[float]) -> float:
        denom = len(verdict_list)
        numerator = sum(verdict_list)
        score = numerator / denom if denom > 0 else np.nan
        return score


@dataclass
class EmbeddingContextRecall(MetricWithEmbeddings, SingleTurnMetric):
    """
    Computes context recall using cosine similarity between context embeddings.
    Expects reference_contexts_embeddings as a list of (embedding for main content, embedding for summary) tuples and
    retrieved_contexts as a list of text strings to be embedded. Since it is cumbersome to try to define the exact
    correct set of embeddings for each testset sample, using the cluster of knowledge graph nodes used to generate the
    sample is the only viable option for a small project like this. Measuring whether retrieved context nodes match
    the cluster exactly is not a good measure of success, however. Comparing using embeddings and a continuous similarity
    metric is a much better approximation of the target semantic space.
    """

    name: str = "embedding_context_recall"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_contexts",  # list of text
                "reference_contexts_embeddings",  # list of embedding tuples
            }
        }
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS

    @staticmethod
    def cosine_sim(ref_emb: list[float], ret_emb: list[float]) -> float:
        return 1 - cosine(np.array(ref_emb), np.array(ret_emb))

    def _max_cosine_score(
        self,
        reference_embs_tuple: tuple[list[float], ...],
        retrieved_embeddings: list[list[float]],
    ) -> float:
        """
        Given a tuple of reference embeddings and a list of retrieved embeddings, return the maximum
        cosine similarity score for any element in the tuple against any retrieved embedding.
        """
        # inner: find the max similarity for each reference embedding type for a document
        # outer: find the max similarity for the document
        return max(
            max(self.cosine_sim(ref_emb, ret_emb) for ret_emb in retrieved_embeddings)
            for ref_emb in reference_embs_tuple
            if ref_emb is not None
        )

    @staticmethod
    def matmul_max_cosine_score(
        reference_embs_tuple: tuple[list[float], ...],
        retrieved_embeddings: list[list[float]],
    ) -> float:
        """
        Given a tuple of reference embeddings and a list of retrieved embeddings, return the maximum
        cosine similarity score for any element in the tuple against any retrieved embedding.
        """
        ref_arr = np.array(reference_embs_tuple)
        ret_arr = np.array(retrieved_embeddings)
        # Compute cosine similarity matrix
        norm_ref = ref_arr / np.linalg.norm(ref_arr, axis=1, keepdims=True)
        norm_ret = ret_arr / np.linalg.norm(ret_arr, axis=1, keepdims=True)
        sim_matrix = np.dot(norm_ref, norm_ret.T)
        return np.max(sim_matrix)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: t.Any
    ) -> float:
        retrieved_contexts = sample.retrieved_contexts
        reference_contexts_embeddings = sample.reference_contexts_embeddings
        assert (
            retrieved_contexts is not None and reference_contexts_embeddings is not None
        ), "Retrieved contexts or reference contexts embeddings are empty"

        # embed retrieved contexts
        assert (
            self.embeddings is not None
        ), "Embeddings model must be set for EmbeddingContextRecall."
        retrieved_embeddings = await self.embeddings.embed_texts(retrieved_contexts)

        scores = []
        for reference_embs_tuple in reference_contexts_embeddings:
            # Filter out None values from the reference embeddings tuple
            reference_embs_tuple = tuple(
                emb for emb in reference_embs_tuple if emb is not None
            )

            max_score = self.matmul_max_cosine_score(
                reference_embs_tuple, retrieved_embeddings
            )
            # assert math.isclose(
            #     max_score,
            #     self._max_cosine_score(reference_embs_tuple, retrieved_embeddings),
            #     abs_tol=1e-6,
            # ), f"max scores not equal using different methods"
            scores.append(max_score)

        return float(np.mean(scores))


context_recall = ContextRecall()

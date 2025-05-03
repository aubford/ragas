from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial.distance import cosine
import math

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._string import NonLLMStringSimilarity
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


class QAC(BaseModel):
    query: str = Field(..., description="Query")
    context: str = Field(..., description="Context")
    answer: str = Field(..., description="Answer")


class Verification(BaseModel):
    reason: str = Field(..., description="Explanation for the verdict")
    verdict: int = Field(
        ...,
        description='Binary (0/1) verdict. "1" if the provided context was useful in answering the query, "0" if not.',
    )


class ContextPrecisionPrompt(PydanticPrompt[QAC, Verification]):
    name: str = "context_precision"
    instruction: str = (
        'Given query, answer and context verify if the context was useful in arriving at the given answer for the query. Give verdict as "1" if useful and "0" if not.'
    )
    input_model = QAC
    output_model = Verification
    examples = [
        (
            QAC(
                query="What can you tell me about Albert Einstein?",
                context="Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",
                answer="Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
            ),
            Verification(
                reason="The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.",
                verdict=1,
            ),
        ),
        (
            QAC(
                query="who won 2020 icc world cup?",
                context="The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",
                answer="England",
            ),
            Verification(
                reason="the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",
                verdict=1,
            ),
        ),
        (
            QAC(
                query="What is the tallest mountain in the world?",
                context="The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.",
                answer="Mount Everest.",
            ),
            Verification(
                reason="the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the query about the world's tallest mountain.",
                verdict=0,
            ),
        ),
    ]


@dataclass
class LLMContextPrecisionWithReference(MetricWithLLM, SingleTurnMetric):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.

    Attributes
    ----------
    name : str
    evaluation_mode: EvaluationMode
    context_precision_prompt: Prompt
    """

    name: str = "llm_context_precision_with_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "retrieved_contexts",
                "reference",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS
    context_precision_prompt: PydanticPrompt = field(
        default_factory=ContextPrecisionPrompt
    )
    max_retries: int = 1

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        return row["user_input"], row["retrieved_contexts"], row["reference"]

    def _calculate_average_precision(
        self, verifications: t.List[Verification]
    ) -> float:
        score = np.nan

        verdict_list = [1 if ver.verdict else 0 for ver in verifications]
        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        if np.isnan(score):
            logger.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )
        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(
        self,
        row: t.Dict,
        callbacks: Callbacks,
    ) -> float:
        assert self.llm is not None, "LLM is not set"

        user_input, retrieved_contexts, reference = self._get_row_attributes(row)
        responses = []
        for context in retrieved_contexts:
            verdicts: t.List[Verification] = (
                await self.context_precision_prompt.generate_multiple(
                    data=QAC(
                        query=user_input,
                        context=context,
                        answer=reference,
                    ),
                    llm=self.llm,
                    callbacks=callbacks,
                )
            )

            responses.append([result.model_dump() for result in verdicts])

        answers = []
        for response in responses:
            agg_answer = ensembler.from_discrete([response], "verdict")
            answers.append(Verification(**agg_answer[0]))

        score = self._calculate_average_precision(answers)
        return score


@dataclass
class LLMContextPrecisionWithoutReference(LLMContextPrecisionWithReference):
    name: str = "llm_context_precision_without_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        return row["user_input"], row["retrieved_contexts"], row["response"]


@dataclass
class NonLLMContextPrecisionWithReference(SingleTurnMetric):
    name: str = "non_llm_context_precision_with_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_contexts",
                "reference_contexts",
            }
        }
    )
    distance_measure: SingleTurnMetric = field(
        default_factory=lambda: NonLLMStringSimilarity()
    )

    def __post_init__(self):
        if isinstance(self.distance_measure, MetricWithLLM):
            raise ValueError(
                "distance_measure must not be an instance of MetricWithLLM for NonLLMContextPrecisionWithReference"
            )

    def init(self, run_config: RunConfig) -> None: ...

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        sample = SingleTurnSample(**row)
        return await self._single_turn_ascore(sample, callbacks)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """For each retrieved context, find the best matching reference context. Try both main content
        and summary for each reference document. Return the similarity score of the best match.
        Don't collapse to a binary score because we aren't looking for exact doc match but overall
        amount of matching information.
        """
        retrieved_contexts = sample.retrieved_contexts
        reference_contexts = sample.reference_contexts
        assert retrieved_contexts is not None, "retrieved_contexts is empty"
        assert reference_contexts is not None, "reference_contexts is empty"

        scores = []
        for rc in retrieved_contexts:
            max_scores = []
            for ref, summary in reference_contexts:
                assert ref, "reference is empty"

                max_vs_ref = await self.distance_measure.single_turn_ascore(
                    SingleTurnSample(reference=rc, response=ref), callbacks
                )
                if summary:
                    max_vs_summary = await self.distance_measure.single_turn_ascore(
                        SingleTurnSample(reference=rc, response=summary), callbacks
                    )
                else:
                    max_vs_summary = 0

                max_scores.append(max(max_vs_ref, max_vs_summary))
            scores.append(max(max_scores))
        return self._calculate_average_precision(scores)

    def _calculate_average_precision(self, verdict_list: t.List[int]) -> float:
        score = np.nan

        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        return score


@dataclass
class EmbeddingContextPrecision(MetricWithEmbeddings, SingleTurnMetric):
    """
    Computes context precision using cosine similarity between context embeddings.
    Expects reference_contexts_embeddings as a list of (embedding for main content, embedding for summary) tuples and
    retrieved_contexts as a list of text strings to be embedded. Since it is cumbersome to try to define the exact
    correct set of embeddings for each testset sample, using the cluster of knowledge graph nodes used to generate the
    sample is the only viable option for a small project like this. Measuring whether retrieved context nodes match
    the cluster exactly is not a good measure of success, however. Comparing using embeddings and a continuous similarity
    metric is a much better approximation of the target semantic space.
    """

    name: str = "embedding_context_precision"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_contexts",  # list of text
                "reference_contexts_embeddings",  # list of (embedding, embedding, embedding)
            }
        }
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS

    @staticmethod
    def _max_cosine_score(retrieved_emb: list[float], reference_embs: tuple[list[float], ...]
    ) -> float:
        """
        Given a retrieved embedding and an iterable of reference embeddings, return the maximum
        cosine similarity score.
        """
        retrieved_emb_np = np.array(retrieved_emb)

        def cosine_sim(target_embs: list[float]) -> float:
            target_embs_np = np.array(target_embs)
            return 1 - cosine(target_embs_np, retrieved_emb_np)

        return max(
            (cosine_sim(emb) for emb in reference_embs),
            default=0.0,
        )

    @staticmethod
    def matmul_max_cosine_score_precision(
        retrieved_emb: list[float],
        reference_embs: tuple[list[float], ...],
    ) -> float:
        """
        Given a single retrieved embedding and a tuple of reference embeddings, return the maximum
        cosine similarity score between the retrieved embedding and any reference embedding.
        """
        retrieved_emb_np = np.array(retrieved_emb)
        ref_arr = np.array(reference_embs)
        # Normalize
        norm_ref = ref_arr / np.linalg.norm(ref_arr, axis=1, keepdims=True)
        norm_ret = retrieved_emb_np / np.linalg.norm(retrieved_emb_np)
        sim_scores = np.dot(norm_ref, norm_ret)
        return np.max(sim_scores)

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks) -> float:
        retrieved_contexts = sample.retrieved_contexts  # list of text
        reference_contexts_embeddings = sample.reference_contexts_embeddings
        assert (
            retrieved_contexts is not None and reference_contexts_embeddings is not None
        )

        assert (
            self.embeddings is not None
        ), "Embeddings model must be set for EmbeddingContextPrecision."
        # embed retrieved_contexts
        retrieved_embeddings = await self.embeddings.embed_texts(retrieved_contexts)

        scores = []
        for retrieved_emb in retrieved_embeddings:
            # Compute the max score across all reference_embs for this retrieved_emb
            max_score = max(
                (
                    self._max_cosine_score(retrieved_emb, reference_embs_tuple)
                    for reference_embs_tuple in reference_contexts_embeddings
                ),
                default=0.0,
            )
            max_score_fancy = max(
                (
                    self.matmul_max_cosine_score_precision(
                        retrieved_emb, reference_embs_tuple
                    )
                    for reference_embs_tuple in reference_contexts_embeddings
                ),
                default=0.0,
            )
            assert math.isclose(
                max_score, max_score_fancy, abs_tol=1e-6
            ), f"max_score: {max_score} not equal to max_score_fancy: {max_score_fancy}"
            scores.append(max_score)

        return float(np.mean(scores)) if scores else float("nan")


@dataclass
class ContextPrecision(LLMContextPrecisionWithReference):
    name: str = "context_precision"

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await super()._single_turn_ascore(sample, callbacks)

    @deprecated(
        since="0.2", removal="0.3", alternative="LLMContextPrecisionWithReference"
    )
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


@dataclass
class ContextUtilization(LLMContextPrecisionWithoutReference):
    name: str = "context_utilization"

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await super()._single_turn_ascore(sample, callbacks)

    @deprecated(
        since="0.2", removal="0.3", alternative="LLMContextPrecisionWithoutReference"
    )
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


context_precision = ContextPrecision()
context_utilization = ContextUtilization()

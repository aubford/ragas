from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


class ResponseRelevanceDiverseOutput(BaseModel):
    queries: list[str] = Field(
        description="List of different queries that the given answer directly addresses"
    )
    noncommittal: int = Field(
        description="Binary (0/1) verdict on whether the answer is committal (0) or noncommittal (1)"
    )


class ResponseRelevanceDiverseInput(BaseModel):
    answer: str = Field(
        description="The answer to generate queries for and analyze for responsiveness"
    )


class ResponseRelevanceDiversePrompt(
    PydanticPrompt[ResponseRelevanceDiverseInput, ResponseRelevanceDiverseOutput]
):
    instruction = """Given the answer provided, perform two tasks:

# Task 1
Generate a set of 3–4 queries that the answer could directly and correctly address. Each query in the set must be distinct from the others by either:
- Paraphrastic variance (same meaning, different phrasing), or
- Semantic variance (different meaning, but still is directly and correctly addressed by the answer).

Allow semantic variance only when the nature of the answer permits a wide range of plausible queries. Avoid introducing semantic drift when the answer is specific or narrow in scope.
Do not introduce variance for named entities. Named entities should remain intact.

# Task 2
Determine whether the answer is noncommittal. Examples of noncommittal answers are statements that include "I don't know" or "I'm not sure". Label it as:
   - 1 if the answer is noncommittal (evasive, vague, or ambiguous)
   - 0 if the answer is committal (direct, specific, and clear)
"""
    input_model = ResponseRelevanceDiverseInput
    output_model = ResponseRelevanceDiverseOutput
    examples = [
        (
            ResponseRelevanceDiverseInput(
                answer="Albert Einstein was born in Germany.",
            ),
            ResponseRelevanceDiverseOutput(
                queries=[
                    "Where was Albert Einstein born?",
                    "Which country was Albert Einstein's birthplace?",
                    "What was Albert Einstein's country of birth?",
                ],
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceDiverseInput(
                answer="""For optimizing a setup involving multiple 4x4 MIMO antennas, it is generally recommended to space the antennas 1-2 feet apart to minimize interference and maximize performance. This spacing helps to avoid transmit interference, which is particularly important when upload speeds are a priority. For a single antenna to serve two modems, make sure both modems don't use SIMs from the same carrier, as they might operate on the same frequencies, potentially causing interference. To further optimize performance, consider manually setting allowed frequencies per modem/element to prevent frequency clashing.""",
            ),
            ResponseRelevanceDiverseOutput(
                queries=[
                    "How can you optimize a setup involving multiple 4x4 MIMO antennas and what are the considerations for using a single antenna for two modems? What are the recommended spacing and placement strategies for these antennas?",
                    "What steps should you take to optimize performance for a 4x4 MIMO antenna setup for multiple antennas? What do you need to do to reduce interference? Also discuss the same for a single antenna setup. What situations could cause interference in that case and why?",
                    "Discuss 4x4 MIMO antenna setups for two scenarios. First, for a single antenna setup serving two modems. Second, for a multiple antenna setup. What pitfalls should be avoided in each case, specifically with regards to interference, placement, or SIMs? Explain the technical root causes of those pitfalls.",
                    "My upload speeds are painfully slow with my 4x4 MIMO antenna setup. What are some obvious things I should check to make sure I'm not doing something wrong?",
                ],
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceDiverseInput(
                answer="I don't know about the groundbreaking feature of the smartphone invented in 2023 as I am unaware of information beyond 2022.",
            ),
            ResponseRelevanceDiverseOutput(
                queries=[
                    "What was the groundbreaking feature of the smartphone invented in 2023?",
                    "Can you describe the groundbreaking feature implemented in the 2023 smartphone?",
                    "What new smartphone feature was released in 2023?",
                ],
                noncommittal=1,
            ),
        ),
    ]


@dataclass
class ResponseRelevancyDiverse(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    This strategy differs from ResponseRelevancy by utilizing a single prompt to request
    three diverse questions to be generated for the answer and then taking the maximum
    similarity of the three. This primarily allows for paraphrastic variance. It also
    mitigates semantic variance in cases where semantic variance is expected based on the
    answer.

    Attributes
    ----------
    name: string
        The name of the metrics
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy_diverse"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    question_generation: PydanticPrompt = ResponseRelevanceDiversePrompt()

    def calculate_similarity(self, question: str, generated_questions: list[str]):
        assert (
            self.embeddings is not None
        ), f"Error: '{self.name}' requires embeddings to be set."
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)
        ).reshape(len(generated_questions), -1)
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

    def _calculate_score(
        self, response: ResponseRelevanceDiverseOutput, row: t.Dict
    ) -> float:
        # Fast fail if response is noncommittal
        if response.noncommittal:
            return 0.0

        question = row["user_input"]
        gen_questions = response.queries
        if not gen_questions or all(q == "" for q in gen_questions):
            logger.warning("Invalid response. No valid questions were generated.")
            score = np.nan
        else:
            cosine_sim = self.calculate_similarity(question, gen_questions)
            # Use max similarity instead of mean
            score = cosine_sim.max()

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_input = ResponseRelevanceDiverseInput(
            answer=row["response"],
        )

        # Single LLM call to generate multiple questions
        response = await self.question_generation.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )

        return self._calculate_score(response, row)


class AnswerRelevancyDiverse(ResponseRelevancyDiverse):
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


answer_relevancy_diverse = AnswerRelevancyDiverse()

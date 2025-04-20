from __future__ import annotations

import asyncio
import logging
import typing as t
import time
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


class ResponseRelevanceOutput(BaseModel):
    question: str = Field(
        description="A question that the given answer directly addresses"
    )
    noncommittal: int = Field(
        description="Indicator for whether the answer is committal (0) or noncommittal (1)"
    )


class ResponseRelevanceInput(BaseModel):
    answer: str = Field(
        description="The answer to generate a question for and analyze for responsiveness"
    )


class ResponseRelevancePrompt(
    PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]
):
    temperature: float = 0.9

    @property
    def instruction(self):
        return f"""[Timestamp: {time.time()}]
    
Given the answer provided, perform two tasks:

1. Generate a question that the answer could directly and correctly address.

2. Determine whether the answer is noncommittal. Examples of noncommittal answers are statements that include "I don't know" or "I'm not sure". Label it as:
   - 1 if the answer is noncommittal (evasive, vague, or ambiguous)
   - 0 if the answer is committal (direct, specific, and clear)
"""

    input_model = ResponseRelevanceInput
    output_model = ResponseRelevanceOutput
    examples = [
        (
            ResponseRelevanceInput(
                answer="""Albert Einstein was born in Germany.""",
            ),
            ResponseRelevanceOutput(
                question="Where was Albert Einstein born?",
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceInput(
                answer="""I don't know about the groundbreaking feature of the smartphone invented in 2023 as I am unaware of information beyond 2022.""",
            ),
            ResponseRelevanceOutput(
                question="What was the groundbreaking feature of the smartphone invented in 2023?",
                noncommittal=1,
            ),
        ),
    ]


@dataclass
class ResponseRelevancy(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    question_generation: PydanticPrompt = ResponseRelevancePrompt()
    strictness: int = 3

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
        self, responses: t.Sequence[ResponseRelevanceOutput], row: t.Dict
    ) -> float:
        # Fast fail if any response is noncommittal
        has_noncommittal_response = np.any(
            [response.noncommittal for response in responses]
        )
        if has_noncommittal_response:
            return 0.0

        question = row["user_input"]
        gen_questions = [response.question for response in responses]
        if all(q == "" for q in gen_questions):
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'question'"
            )
            score = np.nan
        else:
            cosine_sim = self.calculate_similarity(question, gen_questions)
            score = cosine_sim.mean()

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_input = ResponseRelevanceInput(answer=row["response"])
        tasks = [
            self.question_generation.generate(
                data=prompt_input,
                llm=self.llm,
                callbacks=callbacks,
            )
            for _ in range(self.strictness)
        ]
        responses = await asyncio.gather(*tasks)

        return self._calculate_score(responses, row)


class AnswerRelevancy(ResponseRelevancy):
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


answer_relevancy = AnswerRelevancy()

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

import re
from typing import Sequence

import numpy as np
from wordfreq import word_frequency

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

_SIF_A: float = 5e-3
_TOKEN_RE = re.compile(r"[A-Za-z']+")


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

# Task 1: Guess the prompt.
Your task is to guess the prompt that resulted in the given answer in 3-4 attempts. Only one guess needs to be correct for you to be successful.
Generate a set of 3–4 prompts that are the most likely to have the highest cosine similarity with the real prompt that resulted in the given answer.

You can leverage the following strategies to create a diverse set of guesses, which will increase your odds of getting one correct:
- Paraphrastic variance (same meaning, different phrasing), or
- Semantic variance (different meaning, but still addresses all the content in the answer). Semantic variance is useful only when the nature of the answer is broad in scope and permits a wide range of plausible queries.

Follow these guidelines:
- Do not introduce variance for named entities. Maintain the full set of named entities from the answer in each query.
- Each question should address the content of the *entire* answer as a whole. Obviously, a question that only addresses a part of the answer will never be correct.

# Task 2: Is the answer noncommittal?
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

    def get_question_embeddings(
        self, question: str, generated_questions: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for the question and generated questions.
        Returns:
            Tuple of (question_vec, gen_question_vec)
        """
        assert (
            self.embeddings is not None
        ), f"Error: '{self.name}' requires embeddings to be set."
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)
        ).reshape(len(generated_questions), -1)
        return question_vec, gen_question_vec

    def calculate_similarity(
        self, question_vec: np.ndarray, gen_question_vec: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between the question embedding and generated question embeddings.
        Returns:
            Array of cosine similarities.
        """
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

    def _sentence_probability(self, sentence: str) -> float:
        """Return average word probability for a sentence using the `wordfreq` corpus."""
        tokens = _TOKEN_RE.findall(sentence.lower())
        if not tokens:
            return 1.0  # fall back to high probability for empty input
        freqs = [word_frequency(tok, "en") for tok in tokens]
        return float(np.mean(freqs))

    def combine_embeddings(
        self, texts: Sequence[str], embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Combine component-question embeddings into a single vector using
        Smooth Inverse Frequency (SIF) weights derived from the `wordfreq` list.

        This is an attempt to counter the LLM's occasional tendency to cheat its directive
        to generate variance by just chopping a single question into pieces.

        Parameters
        ----------
        texts
            Component question texts in the same order as `embeddings`.
        embeddings
            Pre-computed embeddings as a 2D NumPy array (n, d).

        Returns
        -------
        np.ndarray
            2D array of shape (1, d) representing the combined semantic content embedding.
        """
        if len(texts) != embeddings.shape[0]:
            raise ValueError("`texts` and `embeddings` must have the same length")

        E = embeddings.astype(np.float64)
        n, _ = E.shape

        # --- Smooth Inverse Frequency weights ----------------------------------
        probs = np.array(
            [self._sentence_probability(t) for t in texts], dtype=np.float64
        )
        weights = _SIF_A / (_SIF_A + probs)  # a / (a + p(w))
        weights /= weights.sum()  # normalise to sum to 1

        # --- Weighted mean -----------------------------------------------------
        v = (weights[:, None] * E).sum(axis=0)

        # --- Remove first principal component ----------------------------------
        E_centered = E - E.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(E_centered, full_matrices=False)
        pc1 = vh[0]
        v -= pc1 * np.dot(pc1, v)

        # --- L2 normalisation ---------------------------------------------------
        norm = np.linalg.norm(v)
        if norm == 0.0:
            raise ValueError("Resultant vector has zero magnitude")
        return (v / norm).reshape(1, -1)  # 2D array (1, d)

    def combine_embeddings_concat(self, texts: Sequence[str]) -> np.ndarray:
        """
        Combine all the questions into a single synthetic question. An alternative attempt to
        counter the LLM's tendency to cheat its directive to generate variance by just chopping
        a single question into pieces. Seems to work better than SIF when cheating
        is egregious.

        Parameters
        ----------
        texts
            Component question strings in their original order.

        Returns
        -------
        np.ndarray
            2‑D array of shape (1, d) containing a unit‑norm embedding.
        """
        assert (
            self.embeddings is not None
        ), f"Error: '{self.name}' requires embeddings to be set."
        synthetic_question: str = " ".join(filter(None, texts))
        vec = np.asarray(
            self.embeddings.embed_query(synthetic_question), dtype=np.float64
        )
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            raise ValueError("Zero‑norm embedding generated for synthetic question")
        return (vec / norm).reshape(1, -1)

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
            question_vec, gen_question_vec = self.get_question_embeddings(
                question, gen_questions
            )
            aggregate_embedding = self.combine_embeddings(
                gen_questions, gen_question_vec
            )
            # Two strategies to combine embeddings in case the model cheats
            concat_embedding = self.combine_embeddings_concat(gen_questions)
            cosine_sim = self.calculate_similarity(
                question_vec,
                np.vstack([gen_question_vec, aggregate_embedding, concat_embedding]),
            )

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

# TESTING
if __name__ == "__main__":
    from ragas.embeddings.base import (
        embedding_factory,
    )

    instance = ResponseRelevancyDiverse(embeddings=embedding_factory())

    user_input = "What troubleshooting steps and considerations should a technician take when experiencing a drop in carrier aggregation (CA) bands on a Peplink MAX BR1 5G Pro or BR2 5G Pro across multiple carriers, and what factors could be responsible for this behavior?"
    response = ResponseRelevanceDiverseOutput(
        queries=[
            "What troubleshooting steps should be taken when carrier aggregation drops occur on Peplink MAX BR1 5G Pro and BR2 5G Pro devices?",
            "How can a technician diagnose issues related to carrier aggregation reduction on Peplink 5G Pro devices across different carriers?",
            "What factors influence the occurrence of carrier aggregation drops on Peplink network devices?",
            "What are common reasons for carrier aggregation not functioning properly on Peplink MAX BR1 and BR2 devices?",
        ],
        noncommittal=0,
    )

    score = instance._calculate_score(response, {"user_input": user_input})
    print(score)

    user_input = "How can a technician monitor and interpret the power supply input voltage and GPIO status on a Peplink router, and what steps should they take if voltage-related instability or restarts occur?"
    # Reference for 99%: How can a technician monitor power supply input voltage and GPIO status on a Peplink router, and what steps should be taken if voltage-related instability or restarts occur?
    response = ResponseRelevanceDiverseOutput(
        queries=[
            "How can a technician monitor power supply input voltage and GPIO status on a Peplink router?",
            "What steps should be taken if a Peplink router experiences voltage-related instability or spontaneous restarts?",
            "How do you configure and use GPIO pins on Peplink routers for voltage monitoring?",
            "What are recommended practices to ensure stable power supply and prevent reboots on Peplink routers?",
        ],
        noncommittal=0,
    )

    score = instance._calculate_score(response, {"user_input": user_input})
    print(score)

    # USING THE NEW PROMPT WORKS GREAT!
    user_input = "How do you achieve reliable, high-bandwidth live video streaming from a mobile camera crew using multiple 3G/4G LTE connections, and what are the critical configuration steps and considerations for SpeedFusion bonding with Pepwave MAX On-The-Go and Peplink Balance devices?"
    response = ResponseRelevanceDiverseOutput(
        queries=[
            "How can I set up reliable high-bandwidth live video streaming from a mobile camera crew using multiple 3G/4G LTE connections with Pepwave and Peplink devices, and what are the key configuration steps involved?",
            "What is the recommended approach for achieving resilient live video streaming over multiple cellular links with Pepwave and Peplink devices, including device setup, bonding technology, and critical considerations?",
            "Can you explain the best practices for using SpeedFusion bonding with Pepwave MAX On-The-Go and Peplink Balance routers to enable high-quality live video streaming from a mobile camera crew over multiple 3G/4G LTE connections?",
            "What setup and configuration are necessary to use Pepwave and Peplink devices for bonding multiple cellular connections to support live video streaming from a mobile crew, and what are the important factors to consider?",
        ],
        noncommittal=0,
    )

    score = instance._calculate_score(response, {"user_input": user_input})
    print(score)

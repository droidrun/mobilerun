"""
StructuredOutputAgent - Extract structured data from final answers.

Takes a raw text answer and a Pydantic model, then returns a validated model
instance. Answers that already contain JSON are parsed locally first; otherwise
the agent falls back to LLM structured extraction.
"""

import json
import logging
import re
from collections.abc import Iterator
from typing import Any, Type, TypeVar

from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel, ValidationError

from mobilerun.agent.utils.inference import astructured_predict_with_retries

logger = logging.getLogger("mobilerun")

T = TypeVar("T", bound=BaseModel)

_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def coerce_structured_output_from_text(
    pydantic_model: Type[T], answer_text: str
) -> T | None:
    """Return a validated model when *answer_text* already contains JSON."""

    for candidate in _iter_json_candidates(answer_text):
        try:
            if isinstance(candidate, str):
                return pydantic_model.model_validate_json(candidate)
            return pydantic_model.model_validate(candidate)
        except (TypeError, ValueError, ValidationError):
            continue
    return None


def _iter_json_candidates(text: str) -> Iterator[str | Any]:
    stripped = text.strip()
    if not stripped:
        return

    yield stripped

    for match in _FENCED_JSON_RE.finditer(text):
        candidate = match.group(1).strip()
        if candidate:
            yield candidate

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        yield value


class StructuredOutputAgent(Workflow):
    """
    Agent that extracts structured output from text answers.

    Uses direct Pydantic validation for JSON answers, then
    LLM.structured_predict() for natural-language answers.
    """

    def __init__(
        self,
        llm: LLM | None,
        pydantic_model: Type[BaseModel],
        answer_text: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.pydantic_model = pydantic_model
        self.answer_text = answer_text

    @step
    async def extract_structured_output(
        self, ctx: Context, ev: StartEvent
    ) -> StopEvent:
        """
        Extract structured output using direct validation or structured_predict().
        """
        logger.debug("Extracting structured output from final answer...")

        try:
            direct_output = coerce_structured_output_from_text(
                self.pydantic_model,
                self.answer_text,
            )
            if direct_output is not None:
                logger.debug("Parsed structured output directly from final answer")
                return StopEvent(
                    result={
                        "structured_output": direct_output,
                        "success": True,
                        "error_message": "",
                    }
                )

            if self.llm is None:
                raise ValueError(
                    "No structured output LLM is configured and the final answer "
                    "does not contain valid JSON for the requested model"
                )

            prompt = PromptTemplate(
                "Extract structured information from the following text:\n\n{text}"
            )

            logger.info("StructuredOutput response:", extra={"color": "magenta"})
            structured_output = await astructured_predict_with_retries(
                self.llm, self.pydantic_model, prompt, text=self.answer_text
            )

            logger.debug("Successfully extracted structured output")

            return StopEvent(
                result={
                    "structured_output": structured_output,
                    "success": True,
                    "error_message": "",
                }
            )

        except Exception as e:
            logger.error(f"Failed to extract structured output: {e}")

            return StopEvent(
                result={
                    "structured_output": None,
                    "success": False,
                    "error_message": str(e),
                }
            )

"""
StructuredOutputAgent - Extract structured data from final answers.

Takes a raw text answer and a Pydantic model, uses structured_predict()
to extract structured data from the text.
"""

import logging
from typing import Type

from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel

from droidrun.agent.utils.inference import astructured_predict_with_retries

logger = logging.getLogger("droidrun")


class StructuredOutputAgent(Workflow):
    """
    Agent that extracts structured output from text answers.

    Uses LLM.structured_predict() to parse text into Pydantic models.
    """

    def __init__(
        self,
        llm: LLM,
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
        Extract structured output using structured_predict().
        """
        logger.debug("🔍 Extracting structured output from final answer...")

        try:
            # Create prompt for extraction
            prompt = PromptTemplate(
                "Extract structured information from the following text:\n\n{text}"
            )

            # Use structured_predict to extract data
            logger.info("🔍 StructuredOutput response:", extra={"color": "magenta"})
            structured_output = await astructured_predict_with_retries(
                self.llm, self.pydantic_model, prompt, text=self.answer_text
            )

            logger.debug("✅ Successfully extracted structured output")

            return StopEvent(
                result={
                    "structured_output": structured_output,
                    "success": True,
                    "error_message": "",
                }
            )

        except Exception as e:
            logger.error(f"❌ Failed to extract structured output: {e}")

            return StopEvent(
                result={
                    "structured_output": None,
                    "success": False,
                    "error_message": str(e),
                }
            )

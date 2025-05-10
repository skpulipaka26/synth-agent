"""
SynthAgent - A tool to synthesize responses from multiple LLMs

This module implements a system that queries multiple language models in parallel
and then synthesizes their responses into a cohesive answer.
"""
import os
import time
import asyncio
import logging
from typing import List, Optional

from openai import AsyncOpenAI, AsyncStream
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, Field
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

for logger_name in ["openai", "httpx", "urllib3", "openai.http_client"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


load_dotenv()

API_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SYNTHESIZER_MODEL = "qwen/qwen3-235b-a22b:free"

console = Console()

class SynthModel(BaseModel):
    """Model configuration for a language model to query"""
    name: str
    runs: int = Field(default=1, ge=1, description="Number of times to run this model")

    class Config:
        frozen = True


class SynthModelResult(BaseModel):
    """Result from a language model query"""
    name: str
    result: str = ""
    error: Optional[str] = None
    processing_time: Optional[float] = None


async def process_model(model: SynthModel, client: AsyncOpenAI, query: str) -> SynthModelResult:
    """Process a single model asynchronously

    Args:
        model: The model configuration
        client: The OpenAI client to use for queries
        query: The user query to process

    Returns:
        A SynthModelResult with the model response or error information
    """
    logger.info(f"Processing model - {model.name}")
    start_time = time.time()

    try:
        response = await client.chat.completions.create(
            model=model.name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant"
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
        )

        result = response.choices[0].message.content

        if result is None:
            raise ValueError(f"Received null response from model: {model.name}")

        processing_time = time.time() - start_time
        logger.info(f"Successfully processed model: {model.name} in {processing_time:.2f}s")

        return SynthModelResult(
            name=model.name,
            result=result,
            processing_time=processing_time
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing model {model.name}: {str(e)}")

        return SynthModelResult(
            name=model.name,
            result=f"ERROR: {str(e)}",
            error=str(e),
            processing_time=processing_time
        )


class SynthAgent:
    """Agent that synthesizes responses from multiple language models"""

    def __init__(self, models: List[SynthModel]):
        """Initialize the agent with a list of models

        Args:
            models: List of SynthModel configurations to use
        """
        if not models:
            raise ValueError("At least one model must be provided")

        self.models = models
        self.client = AsyncOpenAI(
            base_url=API_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )

    async def synthesize_async(self, query: str) -> Optional[AsyncStream[ChatCompletionChunk]]:
        """Synthesize responses from multiple models

        Args:
            query: The user query to process

        Returns:
            A synthesized response or None if processing failed
        """

        results = await self.start_async(query)

        if not results:
            logger.error("No valid responses were generated")
            raise ValueError("No responses were generated - check your models or API configuration")

        # Format results for the synthesizer prompt
        results_txt = "\n\n".join([
            f"Model: {result.name} \nResponse: {result.result}"
            for result in results
        ])

        user_content = f"""
            This is the original query:
            {query}

            These are the results you need to synthesize:
            {results_txt}
        """

        logger.info(f"Synthesizing results with {SYNTHESIZER_MODEL}")

        try:
            response = await self.client.chat.completions.create(
                model=SYNTHESIZER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are an advanced AI system designed to synthesize responses from multiple
                        large language models (LLMs) to provide a cohesive, accurate, and comprehensive
                        answer to user queries. Your goal is to leverage the strengths of each LLM,
                        mitigate their weaknesses, and ensure the final response is clear, concise,
                        and tailored to the user's intent. Follow these steps:

                        1. **Query Analysis**: Carefully analyze the user's query to understand its
                           intent, context, and specific requirements.
                        2. **Response Aggregation**: Analyze the collected responses from the selected LLMs.
                        3. **Synthesis Process**:
                           - **Cross-Validation**: Compare responses for consistency, accuracy, and relevance.
                           - **Conflict Resolution**: If discrepancies arise, prioritize responses based on factual accuracy.
                           - **Enhancement**: Combine the best elements of each response.
                        4. **Refinement**: Polish the synthesized response to ensure clarity, coherence, and
                           alignment with the user's requested tone and style.
                        5. **Output**: Deliver a single, seamless response that reflects the synthesized insights.
                        """
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                stream=True,
            )

            return response

        except Exception as e:
            logger.error(f"Error during synthesis: {str(e)}")
            # Return the best individual response as fallback
            return None

    async def start_async(self, query: str) -> List[SynthModelResult]:
        """Process all models concurrently

        Args:
            query: The user query to process

        Returns:
            A list of model results
        """
        if not self.models:
            logger.error("No models configured")
            return []

        # Create tasks for all models
        tasks = [
            process_model(model, self.client, query)
            for model in self.models
        ]

        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Model {self.models[i].name} failed: {str(result)}")
            else:
                processed_results.append(result)

        logger.info(f"Completed processing {len(processed_results)}/{len(self.models)} models")
        return processed_results



async def main():
    """Main entry point for the application"""

    # Configure models
    models = [
        SynthModel(name="qwen/qwen3-30b-a3b:free"),
        SynthModel(name="google/gemini-2.5-pro-exp-03-25"),
        SynthModel(name="deepseek/deepseek-chat-v3-0324"),
    ]

    try:
        synth_agent = SynthAgent(models)

        # Get user input
        query = input("Enter your query: ")
        if not query or len(query.strip()) < 5:
            print("Query too short. Please provide a more detailed query.")
            return

        print("\nProcessing your query. This may take a moment...\n")

        # Process query
        synthesized_result = await synth_agent.synthesize_async(query)

        if synthesized_result:
            print("\n==== Synthesized Result ====\n")

            synthesized_content = ""

            with Live(refresh_per_second=10) as live:
                async for chunk in synthesized_result:
                    try:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, "content") and delta.content is not None:
                                content = delta.content
                                synthesized_content += content
                                # Update live
                                live.update(Markdown(synthesized_content))
                    except Exception as e:
                        logger.warning(f"Error processing chunk: {str(e)}")

        else:
            print("No results were synthesized. Please check your API configuration.")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    """Script entry point"""
    asyncio.run(main())

"""
SynthAgent - A tool to synthesize responses from multiple LLMs

This module implements a system that queries multiple language models in parallel
and then synthesizes their responses into a cohesive answer.
"""
import os
import sys
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
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("synthagent.log", mode="w")]
)
logger = logging.getLogger("synthagent")

console = Console()

load_dotenv()

API_BASE_URL = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "qwen/qwen3-235b-a22b:free")

REQUIRED_ENV_VARS = ["OPENROUTER_API_KEY"]

class SynthAgentConfigError(Exception):
    """Raised when there is a configuration error in SynthAgent."""
    pass

class SynthModel(BaseModel):
    """Model configuration for a language model to query."""
    name: str
    runs: int = Field(default=1, ge=1, description="Number of times to run this model")

    class Config:
        frozen = True

class SynthModelResult(BaseModel):
    """Result from a language model query."""
    name: str
    result: str = ""
    error: Optional[str] = None
    processing_time: Optional[float] = None

def validate_env() -> None:
    """Ensure all required environment variables are set."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        msg = f"Missing required environment variables: {', '.join(missing)}"
        logger.critical(msg)
        console.print(Panel(Text(msg, style="bold red"), title="[red]Configuration Error"))
        sys.exit(1)

def get_default_models() -> List[SynthModel]:
    """Return a default list of models to use."""
    return [
        SynthModel(name="qwen/qwen3-30b-a3b:free", runs=5),
        SynthModel(name="google/gemini-2.5-pro-preview", runs=5),
        SynthModel(name="deepseek/deepseek-chat-v3-0324", runs=5),
    ]

async def process_model(model: SynthModel, client: AsyncOpenAI, query: str) -> SynthModelResult:
    """Process a single model asynchronously."""
    logger.info(f"Processing model - {model.name}")
    start_time = time.time()
    try:
        response = await client.chat.completions.create(
            model=model.name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
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
    """Agent that synthesizes responses from multiple language models."""
    def __init__(self, models: List[SynthModel], api_key: str, api_base: str = API_BASE_URL):
        if not models:
            raise SynthAgentConfigError("At least one model must be provided.")
        self.models = models
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key)

    async def synthesize_async(self, query: str) -> Optional[AsyncStream[ChatCompletionChunk]]:
        """Synthesize responses from multiple models."""
        results = await self.__start_async(query)
        if not results:
            logger.error("No valid responses were generated.")
            raise ValueError("No responses were generated - check your models or API configuration.")
        # Format results for the synthesizer prompt
        results_txt = "\n\n".join([
            f"Model: {result.name} \nResponse: {result.result}" for result in results
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
                        "content": (
                            "You are an advanced AI system designed to synthesize responses from multiple "
                            "large language models (LLMs) to provide a cohesive, accurate, and comprehensive "
                            "answer to user queries. Your goal is to leverage the strengths of each LLM, "
                            "mitigate their weaknesses, and ensure the final response is clear, concise, "
                            "and tailored to the user's intent. Follow these steps:\n\n"
                            "1. Query Analysis: Carefully analyze the user's query to understand its intent, context, and specific requirements.\n"
                            "2. Response Aggregation: Analyze the collected responses from the selected LLMs.\n"
                            "3. Synthesis Process:\n"
                            "   - Cross-Validation: Compare responses for consistency, accuracy, and relevance.\n"
                            "   - Conflict Resolution: If discrepancies arise, prioritize responses based on factual accuracy.\n"
                            "   - Enhancement: Combine the best elements of each response.\n"
                            "4. Refinement: Polish the synthesized response to ensure clarity, coherence, and alignment with the user's requested tone and style.\n"
                            "5. Output: Deliver a single, seamless response that reflects the synthesized insights."
                        )
                    },
                    {"role": "user", "content": user_content},
                ],
                stream=True,
            )
            return response
        except Exception as e:
            logger.error(f"Error during synthesis: {str(e)}")
            return None

    async def __start_async(self, query: str) -> List[SynthModelResult]:
        """Process all models concurrently."""
        if not self.models:
            logger.error("No models configured.")
            return []
        tasks = [
            process_model(model, self.client, query) 
            for model in self.models 
            for _ in range(model.runs)
            ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Model {self.models[i].name} failed: {str(result)}")
            else:
                processed_results.append(result)
        logger.info(f"Completed processing {len(processed_results)}/{len(self.models)} models.")
        return processed_results

async def main(models: Optional[List[SynthModel]] = None) -> None:
    """Main entry point for the application."""
    validate_env()

    api_key = OPENROUTER_API_KEY

    if models is None:
        models = get_default_models()

    synth_agent = SynthAgent(models, api_key)
    # Get user input
    console.print(Panel("[bold cyan]Welcome to SynthAgent![/bold cyan]", title="SynthAgent"))
    query = Prompt.ask("[bold green]Enter your query[/bold green]")
    
    if not query or len(query.strip()) < 5:
        console.print(Panel("Query too short. Please provide a more detailed query.", style="red"))
        return
    
    console.print(Panel("Processing your query. This may take a moment...", style="yellow"))
    
    try:
        synthesized_result = await synth_agent.synthesize_async(query)
        if synthesized_result:
            console.print(Panel("[bold green]==== Synthesized Result ====\n[/bold green]", style="green"))
            synthesized_content = ""
            with Live(refresh_per_second=10, console=console) as live:
                async for chunk in synthesized_result:
                    try:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, "content") and delta.content is not None:
                                content = delta.content
                                synthesized_content += content
                                live.update(Markdown(synthesized_content))
                    except Exception as e:
                        logger.warning(f"Error processing chunk: {str(e)}")
        else:
            console.print(Panel("No results were synthesized. Please check your API configuration.", style="red"))
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        console.print(Panel(f"An error occurred: {str(e)}", style="bold red"))

def cli():
    """Command-line interface entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Exiting SynthAgent. Goodbye![/bold yellow]")
        sys.exit(0)

if __name__ == "__main__":
    cli()

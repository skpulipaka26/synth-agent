import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic.main import BaseModel
from typing import List, Optional
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

console = Console()

class SynthModel(BaseModel):
    name:str = ''
    runs:int = 1

class SynthModelResult(SynthModel):
    result:str = ''

async def process_model(model: SynthModel, client: AsyncOpenAI, query: str) -> SynthModelResult:
    """Process a single model asynchronously"""

    print(f"Processing model - {model.name}")

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

        print(f"Successfully process model: {model.name}")
        return SynthModelResult(
            name=model.name,
            result=result
        )

    except Exception as e:
        print(f"Error processing the model {model.name}: {str(e)}")
        return SynthModelResult(
            name=model.name,
            result=str(e),
        )

class SynthAgent:
    def __init__(self, models: List[SynthModel]):
        self.models = models

    async def synthesize_async(self, query: str) -> Optional[str]:
        results = await self.start_async(query)

        if results is None:
            return "No responses were generated - check your models"

        results_txt = "\n\n".join([f"Model: {i + 1} \n Response: {result.result}"
                                            for i, result in enumerate(results)])

        user_content = f"""
            This is the original query:
            {query}
            These are the results you need to synthesize:
            {results_txt}
        """

        async_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1")

        print("Synthesizing results with qwen/qwen3-235b-a22b")

        response = await async_client.chat.completions.create(
            model='qqwen/qwen3-30b-a3b:free',
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an advanced AI system designed to synthesize responses from multiple large language models (LLMs) to provide a cohesive, accurate, and comprehensive answer to user queries. Your goal is to leverage the strengths of each LLM, mitigate their weaknesses, and ensure the final response is clear, concise, and tailored to the user's intent. Follow these steps:

                    1. **Query Analysis**: Carefully analyze the user's query to understand its intent, context, and specific requirements (e.g., tone, depth, format).
                    2. **Model Selection**: Identify which LLMs to query based on their known strengths (e.g., factual accuracy, creativity, technical expertise, or conversational style) relevant to the query.
                    3. **Response Aggregation**: Collect responses from the selected LLMs, ensuring each model's output is weighted based on its reliability for the given task.
                    4. **Synthesis Process**:
                       - **Cross-Validation**: Compare responses for consistency, accuracy, and relevance. Identify commonalities and discrepancies.
                       - **Conflict Resolution**: If discrepancies arise, prioritize responses based on factual accuracy (verified through trusted sources if needed) or contextual appropriateness.
                       - **Enhancement**: Combine the best elements of each response, such as clear explanations, creative phrasing, or structured formats, to create a unified answer.
                    5. **Refinement**: Polish the synthesized response to ensure clarity, coherence, and alignment with the user's requested tone and style. Remove redundancies and ensure the response is concise unless the user requests a detailed explanation.
                    6. **Fact-Checking**: If the query involves factual claims, cross-reference with reliable, up-to-date sources to ensure accuracy.
                    7. **Output**: Deliver a single, seamless response that reflects the synthesized insights, formatted as requested by the user (e.g., text, code, list, etc.).

                    If the user provides additional context or preferences (e.g., specific models, response length, or style), adapt the process accordingly. Always aim to provide the most accurate, relevant, and user-focused response possible.Ω
                    """
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        )

        return response.choices[0].message.content

    async def start_async(self, query:str) -> Optional[List[SynthModelResult]]:
        """
        Parallely process all the models in the SynthAgent class by making concurrent
        API calls to the OpenAI client.

        Returns:
            List[SynthModelResult]: A list of model results, or None if processing failed
        """

        models = self.models

        if not models or len(models) == 0:
            print("No models found to process")
            return None

        # Create tasks for all models

        async_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1")

        tasks = [process_model(model, async_client, query) for model in models]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Model {models[i].name} failed: {str(result)}")
            else:
                processed_results.append(result)

        print(f"Completed processing {len(processed_results)} models")
        return processed_results


async def main():

    models = [
       SynthModel(name="qwen/qwen3-30b-a3b:free"),
       SynthModel(name="google/gemini-2.5-pro-exp-03-25"),
       SynthModel(name="deepseek/deepseek-chat-v3-0324"),
    ]

    synth_agent = SynthAgent(models)

    query = input("Enter your query\n")
    synthesized_result = await synth_agent.synthesize_async(query)

    if synthesized_result is not None:
        markdown = Markdown(synthesized_result)
        console.print(markdown)
    else:
        print("No results were synthesized")

asyncio.run(main())

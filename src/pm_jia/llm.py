"""
Core LLM processing module for handling all direct AI interactions.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage
from pydantic import BaseModel

from src.pm_jia.config import AgentConfig
from src.pm_jia.logger import setup_logger
from src.pm_jia.model import TokenUsage
from src.pm_jia.tools import Tool

load_dotenv()

logger = setup_logger(__name__)


@dataclass
class LLMResponse:
    """Wrapper for LLM response including token usage."""

    message: Union[ChatCompletionMessage, ParsedChatCompletionMessage]
    token_usage: TokenUsage


@dataclass
class StreamingChunk:
    """Wrapper for a streaming chunk from the LLM."""

    content: str
    is_complete: bool
    token_usage: Optional[TokenUsage] = None


class LLMEngine:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.llm_model = os.getenv("LLM_MODEL", self.config.llm_model)
        self.llm_api_key = os.getenv("OPENAI_API_KEY", "")

        if not self.llm_api_key:
            raise ValueError("LLM API KEY is not set")

        self.llm_client = AsyncOpenAI(api_key=self.llm_api_key)

    async def generate_response(
        self,
        name: str,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        llm_gen_type: Optional[str] = None,
        json_schema: Optional[BaseModel] = None,
        tools: Optional[List[Tool]] = [],
    ) -> LLMResponse:
        """
        Generate a response using the LLM.
        """
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        if llm_gen_type == "json_schema":
            return await self.generate_response_json_schema(
                name, messages, json_schema, temperature, max_tokens, tools
            )
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
            )

            # Extract token usage
            usage = response.usage
            token_usage = TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )

            return LLMResponse(message=response.choices[0].message, token_usage=token_usage)

        except Exception as e:
            logger.error(f"Agent {name} - Error generating response: {e}")
            raise e

    async def generate_response_json_schema(
        self,
        name: str,
        messages: List[Dict],
        json_schema: BaseModel,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Tool]] = [],
    ) -> LLMResponse:
        """
        Generate a response using the LLM and return the response as a JSON object.
        """
        try:
            response = await self.llm_client.beta.chat.completions.parse(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                response_format=json_schema,
            )

            # Extract token usage
            usage = response.usage
            token_usage = TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )

            return LLMResponse(message=response.choices[0].message, token_usage=token_usage)
        except Exception as e:
            logger.error(f"Agent {name} - Error generating response in json schema: {e}")
            raise e

    async def generate_streaming_response(
        self,
        name: str,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Tool]] = [],
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Generate a streaming response using the LLM.

        Args:
            name: Agent name for logging
            messages: Chat messages
            temperature: Response temperature
            max_tokens: Maximum tokens
            tools: Available tools

        Yields:
            StreamingChunk objects containing content and completion status
        """
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        try:
            stream = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=True,
                stream_options={"include_usage": True},
            )

            accumulated_content = ""

            async for chunk in stream:
                # Handle content chunks
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:

                    content = chunk.choices[0].delta.content
                    accumulated_content += content

                    yield StreamingChunk(content=content, is_complete=False)

                # Handle completion and token usage
                elif chunk.choices and chunk.choices[0].finish_reason:
                    token_usage = None
                    if chunk.usage:
                        token_usage = TokenUsage(
                            input_tokens=chunk.usage.prompt_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                        )

                    yield StreamingChunk(content="", is_complete=True, token_usage=token_usage)
                    break

        except Exception as e:
            logger.error(f"Agent {name} - Error in streaming response: {e}")
            yield StreamingChunk(content=f"Error: {str(e)}", is_complete=True)
            raise e


if __name__ == "__main__":
    from src.pm_jia.model import MarketAnalysis, MessageRole, SafetyCheckResult

    llm = LLMEngine()
    messages = [
        {"role": MessageRole.SYSTEM, "content": "You are a helpful assistant."},
        {"role": MessageRole.USER, "content": "What is the tourism industry in China?"},
    ]
    message = asyncio.run(
        llm.generate_response(
            "test_agent_name",
            messages,
            json_schema=MarketAnalysis,
            llm_gen_type="json_schema",
        )
    )
    logger.info(message)

"""
Message and memory management for agents.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openai.types.chat.chat_completion_message import ChatCompletionMessageToolCall

from src.pm_jia.memory import Memory, MemoryStepType, Timing, ToolCall
from src.pm_jia.model import MessageRole


class MessageManager:
    """
    Message and memory management system.
    Centralizes all memory operations and message handling.
    """

    def __init__(self, memory: Memory, agent_name: str):
        self.memory = memory
        self.agent_name = agent_name

    def add_system_message(self, content: str) -> Dict[str, Any]:
        """Add system message to memory and return formatted message."""

        # Directly return if the same system message and memory step already exist
        for step in self.memory.memory_steps:
            if step.step_type == MemoryStepType.SYSTEM and step.content == content:
                return step.raw_message

        message = {"role": MessageRole.SYSTEM, "content": content}

        self.memory.add_step(
            id=str(uuid4()),
            step_type=MemoryStepType.SYSTEM,
            content=content,
            timing=Timing(start_time=datetime.now(), end_time=datetime.now()),
            metadata={"agent": self.agent_name},
            raw_message=message,
        )
        return message

    def add_user_message(self, content: str) -> Dict[str, Any]:
        """Add user message to memory and return formatted message."""

        # Directly return if the same user message and memory step already exist
        for step in self.memory.memory_steps:
            if step.step_type == MemoryStepType.USER and step.content == content:
                return step.raw_message

        message = {"role": MessageRole.USER, "content": content}

        self.memory.add_step(
            id=str(uuid4()),
            step_type=MemoryStepType.USER,
            content=content,
            timing=Timing(start_time=datetime.now(), end_time=datetime.now()),
            metadata={"agent": self.agent_name},
            raw_message=message,
        )

        return message

    def add_assistant_message(
        self,
        content: str,
        token_usage: Optional[Dict] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Add assistant message to memory and return formatted message."""
        start_time = start_time or datetime.now()
        end_time = end_time or datetime.now()
        timing = Timing(start_time=start_time, end_time=end_time)

        message = {"role": MessageRole.ASSISTANT, "content": content}

        self.memory.add_step(
            id=str(uuid4()),
            step_type=MemoryStepType.ASSISTANT,
            content=content,
            timing=timing,
            metadata={"agent": self.agent_name},
            token_usage=token_usage,
            raw_message=message,
        )

        return message

    def add_tool_call_message(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        token_usage: Optional[Dict] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Add tool call message to memory and return formatted message."""
        start_time = start_time or datetime.now()
        end_time = end_time or datetime.now()
        timing = Timing(start_time=start_time, end_time=end_time)

        message = {
            "role": MessageRole.ASSISTANT,
            "content": "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }

        tool_calls_list = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                args=tc.function.arguments,
            )
            for tc in tool_calls
        ]

        self.memory.add_step(
            id=str(uuid4()),
            step_type=MemoryStepType.TOOL_CALL,
            content=None,  # Tool calls don't have content
            timing=timing,
            tool_calls=tool_calls_list,
            metadata={"agent": self.agent_name},
            token_usage=token_usage,
            raw_message=message,
        )

        return message

    def add_tool_response(
        self,
        tool_call_id: str,
        result: Any,
        tool_call: ChatCompletionMessageToolCall,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Add tool response to memory and return formatted message."""
        start_time = start_time or datetime.now()
        end_time = end_time or datetime.now()
        timing = Timing(start_time=start_time, end_time=end_time)

        message = {
            "role": MessageRole.TOOL,
            "content": str(result),
            "tool_call_id": tool_call_id,
        }

        self.memory.add_step(
            id=str(uuid4()),
            step_type=MemoryStepType.TOOL_RESPONSE,
            content=str(result),
            timing=timing,
            tool_calls=[
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args=tool_call.function.arguments,
                )
            ],
            metadata={"agent": self.agent_name},
            raw_message=message,
        )

        return message

    def get_conversation_messages(
        self, depends_on: List[str], orchestrator_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get conversation messages with proper filtering and function call preservation.
        """
        messages = []

        if not self.memory.memory_steps:
            return messages

        for step in self.memory.memory_steps:

            # Skip error steps
            if step.step_type == MemoryStepType.ERROR:
                continue

            # Filter by dependencies
            if (
                step.metadata
                and step.metadata.get("agent") not in depends_on
                and step.metadata.get("agent") != orchestrator_name
            ):
                continue

            if step.raw_message:
                messages.append(step.raw_message)
            else:
                messages.append(
                    {
                        "role": step.step_type.to_message_role(),
                        "content": step.content,
                    }
                )

        return messages

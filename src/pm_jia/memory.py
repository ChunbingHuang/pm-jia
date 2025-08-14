"""
Memory management module for storing and retrieving agent steps and system prompts.
This module provides functionality to track, store, and replay agent actions and their outcomes.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from src.pm_jia.logger import setup_logger
from src.pm_jia.model import MessageRole

logger = setup_logger(__name__)


class MemoryStepType(str, Enum):
    """Types of steps that can be stored in memory."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    ERROR = "error"

    def to_message_role(self) -> MessageRole:
        return {
            MemoryStepType.SYSTEM: MessageRole.SYSTEM,
            MemoryStepType.USER: MessageRole.USER,
            MemoryStepType.ASSISTANT: MessageRole.ASSISTANT,
            MemoryStepType.TOOL_CALL: MessageRole.TOOL,
            MemoryStepType.TOOL_RESPONSE: MessageRole.TOOL,
        }[self]


@dataclass
class Timing:
    """
    Elapsed time information for a step or run.
    """

    start_time: datetime
    end_time: datetime | None = None

    @property
    def duration(self):
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration": self.duration,
        }


@dataclass
class ToolCall:
    """Represents a tool/function call made by an agent."""

    id: str
    name: str
    args: Any

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "args": self.args,
            },
        }


@dataclass
class MemoryStep:
    """Represents a single step in the agent's memory."""

    id: str
    step_type: MemoryStepType
    content: Any
    timing: Timing
    tool_calls: Optional[List[ToolCall]] = None
    metadata: Optional[Dict] = None
    token_usage: Optional[Dict] = None  # Store token usage as dict for flexibility
    raw_message: Optional[Dict] = None  # Store the actual OpenAI message format

    def __post_init__(self):
        """Ensure tool_calls are always ToolCall objects."""
        if self.tool_calls:
            normalized_calls = []
            for tool_call in self.tool_calls:
                if isinstance(tool_call, ToolCall):
                    normalized_calls.append(tool_call)
                elif isinstance(tool_call, dict):
                    # Convert dict to ToolCall
                    if "function" in tool_call and isinstance(tool_call["function"], dict):
                        # OpenAI format
                        normalized_calls.append(
                            ToolCall(
                                id=tool_call.get("id", str(uuid4())),
                                name=tool_call["function"]["name"],
                                args=tool_call["function"]["args"],
                            )
                        )
                    else:
                        # Direct format
                        normalized_calls.append(
                            ToolCall(
                                id=tool_call.get("id", str(uuid4())),
                                name=tool_call.get("name", "unknown"),
                                args=tool_call.get("args", {}),
                            )
                        )
                else:
                    # Create a placeholder ToolCall for invalid data
                    normalized_calls.append(
                        ToolCall(
                            id=str(uuid4()),
                            name="invalid_tool",
                            args={"error": f"Invalid tool_call data: {tool_call}"},
                        )
                    )
            self.tool_calls = normalized_calls

    def to_dict(self) -> Dict:
        # Convert tool_calls to dict format - they should always be ToolCall objects
        tool_calls_dict = []
        if self.tool_calls:
            for tool_call in self.tool_calls:
                if isinstance(tool_call, ToolCall):
                    tool_calls_dict.append(tool_call.to_dict())
                elif isinstance(tool_call, dict):
                    # This shouldn't happen, but handle it gracefully
                    tool_calls_dict.append(tool_call)
                else:
                    # Log this as it indicates a data integrity issue
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Unexpected tool_call type: {type(tool_call)}")
                    tool_calls_dict.append({"error": f"Invalid tool_call type: {type(tool_call)}"})

        return {
            "id": self.id,
            "step_type": self.step_type,
            "content": self.content,
            "timing": self.timing.to_dict(),
            "tool_calls": tool_calls_dict,
            "metadata": self.metadata,
            "token_usage": self.token_usage,
            "raw_message": self.raw_message,
        }


class Memory:
    """
    Memory management system for agents.

    Stores and manages the agent's steps, system prompt, and execution history.
    """

    def __init__(self, system_prompt: Optional[str] = None):
        self.memory_steps: List[MemoryStep] = []
        self.system_prompt = system_prompt

    def add_step(
        self,
        id: str,
        step_type: MemoryStepType,
        content: Any,
        timing: Optional[Timing] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        metadata: Optional[Dict] = None,
        token_usage: Optional[Dict] = None,
        raw_message: Optional[Dict] = None,
    ):
        """
        Add a new step to memory.

        Args:
            id: Unique identifier for the step
            step_type: Type of the step
            content: Content of the step
            timing: Optional timing information for the step
            tool_calls: Optional list of tool calls for the step
            metadata: Optional metadata for the step
            token_usage: Optional token usage information for the step
        """
        step = MemoryStep(
            id=id,
            step_type=step_type,
            content=content,
            metadata=metadata,
            timing=timing,
            tool_calls=tool_calls,
            token_usage=token_usage,
            raw_message=raw_message,
        )
        self.memory_steps.append(step)

    def add_step_batch(self, steps: List[MemoryStep]):
        """
        Add a batch of steps to memory.
        """
        self.memory_steps.extend(steps)

    async def get_step(self, step_id: str) -> Optional[MemoryStep]:
        """
        Retrieve a specific step by ID.

        Args:
            step_id: ID of the step to retrieve

        Returns:
            The step if found, None otherwise
        """
        for step in self.memory_steps:
            if step.id == step_id:
                return step
        return None

    def get_step_sync(self, step_id: str) -> Optional[MemoryStep]:
        """
        Synchronous version of get_step for easier timing updates.

        Args:
            step_id: ID of the step to retrieve

        Returns:
            The step if found, None otherwise
        """
        for step in self.memory_steps:
            if step.id == step_id:
                return step
        return None

    def update_step_token_usage(self, step_id: str, token_usage: Dict) -> bool:
        """
        Update the token usage for a specific step.

        Args:
            step_id: ID of the step to update
            token_usage: Token usage dictionary

        Returns:
            True if step was found and updated, False otherwise
        """
        step = self.get_step_sync(step_id)
        if step:
            step.token_usage = token_usage
            logger.debug(f"Updated token usage for step {step_id}: {token_usage}")
            return True
        return False

    def get_steps_by_type(self, step_type: MemoryStepType) -> List[MemoryStep]:
        """
        Get all steps of a specific type.

        Args:
            step_type: Type of steps to retrieve

        Returns:
            List of matching steps
        """
        return [step for step in self.memory_steps if step.step_type == step_type]

    async def get_recent_steps(self, count: int = 5) -> List[MemoryStep]:
        """
        Get the most recent steps (default: 5)

        Args:
            count: Number of recent steps to retrieve

        Returns:
            List of recent steps
        """
        return self.memory_steps[-count:] if self.memory_steps else []

    async def get_steps_in_timeframe(
        self, start_time: datetime, end_time: Optional[datetime] = None
    ) -> List[MemoryStep]:
        """
        Get steps within a specific timeframe.

        Args:
            start_time: Start of the timeframe
            end_time: End of the timeframe (defaults to now)

        Returns:
            List of steps within the timeframe
        """
        end_time = end_time or datetime.now()
        return [
            step for step in self.memory_steps if start_time <= step.timing.start_time <= end_time
        ]

    async def get_step_ids(self) -> List[str]:
        """
        Get all step IDs.

        Returns:
            List of step IDs
        """
        return [step.id for step in self.memory_steps]

    async def clear(self) -> None:
        """Clear all steps from memory except system prompt."""
        if self.system_prompt:
            self.memory_steps = [
                step for step in self.memory_steps if step.step_type == MemoryStepType.SYSTEM_PROMPT
            ]
        else:
            self.memory_steps = []

    async def get_summary(self, include_metadata: bool = False) -> List[Dict]:
        """
        Get a summary of all steps.

        Args:
            include_metadata: Whether to include step metadata

        Returns:
            List of step summaries
        """
        summaries = []
        for step in self.memory_steps:
            summary = {
                "type": step.step_type.value,
                "timing": step.timing.to_dict(),
            }

            if step.step_type == MemoryStepType.TOOL_CALL:
                summary["tool"] = step.content.get("name")
            elif step.step_type in [MemoryStepType.AGENT_THOUGHT, MemoryStepType.USER_INPUT]:
                summary["preview"] = str(step.content)[:100] + "..."

            if include_metadata:
                summary["metadata"] = step.metadata

            summaries.append(summary)

        return summaries

    async def replay_steps(
        self,
        start_step: Optional[Union[str, int]] = None,
        end_step: Optional[Union[str, int]] = None,
    ) -> List[MemoryStep]:
        """
        Replay steps in memory.

        Args:
            start_step: Starting step (ID or index)
            end_step: Ending step (ID or index)

        Returns:
            List of steps in the replay sequence
        """
        steps = self.memory_steps[:]

        if isinstance(start_step, str):
            start_idx = next((i for i, s in enumerate(steps) if s.id == start_step), 0)
        else:
            start_idx = start_step or 0

        if isinstance(end_step, str):
            end_idx = next((i for i, s in enumerate(steps) if s.id == end_step), len(steps))
        else:
            end_idx = end_step or len(steps)

        return steps[start_idx:end_idx]

    async def export_to_dict(self) -> Dict:
        """
        Export memory contents to a dictionary.

        Returns:
            Dictionary containing all memory data
        """
        return {
            "system_prompt": self.system_prompt,
            "steps": [step.to_dict() for step in self.memory_steps],
            "metadata": {
                "total_steps": len(self.memory_steps),
                "types_distribution": {
                    step_type.value: len(self.get_steps_by_type(step_type))
                    for step_type in MemoryStepType
                },
            },
        }

    async def import_from_dict(self, data: Dict) -> None:
        """
        Import memory contents from a dictionary.

        Args:
            data: Dictionary containing memory data
        """
        self.system_prompt = data.get("system_prompt")
        self.memory_steps = []

        for step_data in data.get("steps", []):
            step_type = MemoryStepType(step_data["step_type"])
            content = step_data["content"]
            metadata = step_data.get("metadata", {})

            # Create timing object without duration (computed property)
            timing_data = step_data["timing"].copy()
            timing_data.pop("duration", None)  # Remove duration if present

            # Convert ISO strings back to datetime objects
            if "start_time" in timing_data and isinstance(timing_data["start_time"], str):
                timing_data["start_time"] = datetime.fromisoformat(timing_data["start_time"])
            if "end_time" in timing_data and isinstance(timing_data["end_time"], str):
                timing_data["end_time"] = datetime.fromisoformat(timing_data["end_time"])

            timing = Timing(**timing_data)

            step_id = step_data.get("id", str(uuid4()))

            tool_calls = step_data.get("tool_calls", None)
            token_usage = step_data.get("token_usage", None)
            raw_message = step_data.get("raw_message", None)

            step = MemoryStep(
                id=step_id,
                step_type=step_type,
                content=content,
                timing=timing,
                tool_calls=tool_calls,
                metadata=metadata,
                token_usage=token_usage,
                raw_message=raw_message,
            )
            self.memory_steps.append(step)

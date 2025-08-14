"""
Implementation of base agent and all other agents.
"""

import asyncio
import json
from abc import ABC
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union
from uuid import uuid4

from dotenv import load_dotenv
from openai.types.chat.chat_completion_message import ChatCompletionMessageToolCall
from pydantic import BaseModel

from src.pm_jia.config import AgentConfig
from src.pm_jia.llm import LLMEngine
from src.pm_jia.logger import setup_logger
from src.pm_jia.memory import Memory, MemoryStep, MemoryStepType, Timing, ToolCall
from src.pm_jia.message_manager import MessageManager
from src.pm_jia.model import MessageRole, SafetyCheckResult, WorkflowPlan, WorkflowStepType
from src.pm_jia.progress_manager import (
    report_calling_function,
    report_custom,
    report_error,
    report_stage_complete,
    report_stage_start,
    report_step_complete,
    report_success,
    report_warning,
)
from src.pm_jia.prompt import PromptTemplates
from src.pm_jia.stats import AgentStatistics
from src.pm_jia.tools import Tool, get_available_tools, tools_to_openai_format
from src.pm_jia.utils import generate_workflow_diagram_cli
from src.pm_jia.workflow_templates import get_default_workflow

load_dotenv()

logger = setup_logger(__name__)

prompt_templates = PromptTemplates()


class BaseAgent(ABC):
    """
    Base agent class that all other agents inherit from.
    Designed for multi-agent systems with shared memory.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm_engine: LLMEngine,
        llm_gen_type: Literal["json_schema", "text"],
        llm_json_schema: Optional[BaseModel] = None,
        memory: Optional[Memory] = None,
        tools: Optional[List[Tool]] = [],
        depends_on: Optional[List[str]] = [],
        orchestrator_name: Optional[str] = "orchestrator",
        **kwargs,
    ):
        self.config = AgentConfig()
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm_engine
        self.shared_memory = memory
        self.tools = tools or []
        self.depends_on = depends_on or []
        self.orchestrator_name = orchestrator_name

        self.tool_names = [tool.name for tool in self.tools]
        self._tool_map = {tool.name: tool for tool in self.tools}

        self.max_steps = self.config.max_steps
        self.llm_gen_type = llm_gen_type
        self.llm_json_schema = llm_json_schema

        if self.llm_gen_type == "json_schema" and self.llm_json_schema is None:
            raise ValueError("LLM JSON schema is required for json_schema generation type")

        self.kwargs = kwargs

        if self.shared_memory:
            self.message_manager = MessageManager(self.shared_memory, self.name)

    async def _build_messages_from_memory(self) -> List[Dict[str, Any]]:
        """Build LLM messages from shared memory and agent context."""
        # Add from memory
        if self.shared_memory and hasattr(self, "message_manager"):
            messages = self.message_manager.get_conversation_messages(
                self.depends_on, self.orchestrator_name
            )
        else:
            messages = []

        # Add agent-specific system prompt
        system_content = self.system_prompt

        # if self.tools:
        #     tool_descriptions = [f"- {tool.name}: {tool.description}" for tool in self.tools]
        #     system_content += f"\n\nAvailable tools:\n" + "\n".join(tool_descriptions)
        #     system_content += f"\n\nTool usage restrictions: Do not use the same tool more than {self.config.max_tool_usage} times."

        messages.append({"role": MessageRole.SYSTEM, "content": system_content})

        return messages

    async def _add_user_request(
        self,
        messages: List[Dict[str, Any]],
        user_input: Optional[str] = None,
        additional_materials: Optional[Dict[str, str]] = None,
    ):
        """Add user request to messages."""
        user_request = ""
        if user_input:
            user_request += f"User input: {user_input}\n"
        if additional_materials:
            user_request += f"Additional materials: {additional_materials}\n"

        if hasattr(self, "message_manager"):
            user_message = self.message_manager.add_user_message(user_request)
        else:
            user_message = {"role": MessageRole.USER, "content": user_request}

        messages.append(user_message)
        return messages

    async def execute(
        self,
        user_input: Optional[str] = None,
        additional_materials: Optional[Dict[str, str]] = None,
    ) -> Union[str, BaseModel]:
        """
        Main method to execute the agent's functionality.

        Args:
            user_input: User's input message
            additional_materials: Additional materials provided by user
        """
        messages = await self._build_messages_from_memory()

        messages = await self._add_user_request(messages, user_input, additional_materials)

        step = 0
        while step <= self.max_steps:

            # Use strict mode for tools when using json_schema generation
            tools_strict = self.llm_gen_type == "json_schema"

            start_time_llm = datetime.now()
            llm_response = await self.llm.generate_response(
                name=self.name,
                messages=messages,
                llm_gen_type=self.llm_gen_type,
                json_schema=self.llm_json_schema,
                tools=(
                    tools_to_openai_format(self.tools, strict=tools_strict) if self.tools else []
                ),
            )
            end_time_llm = datetime.now()

            response_message = llm_response.message
            token_usage_dict = llm_response.token_usage.dict() if llm_response.token_usage else None

            if response_message.content:
                assistant_message = self._handle_text_response(
                    response_message.content, token_usage_dict, start_time_llm, end_time_llm
                )
                messages.append(assistant_message)
                break

            tool_calls = response_message.tool_calls
            if not tool_calls:
                break

            # Execute tool calls
            tool_tasks = [self._execute_tool_call(tool_call) for tool_call in tool_calls]

            for tool_call in tool_calls:
                report_calling_function(
                    stage="Workflow execution",
                    message=f"Args: {tool_call.function.arguments}",
                    agent_name=self.name,
                    function_name=tool_call.function.name,
                )

            tool_results_raw = await asyncio.gather(*tool_tasks, return_exceptions=True)

            tool_call_response_msgs, tool_call_response_memory_steps = (
                self._handle_tool_calls_and_responses(
                    tool_calls, tool_results_raw, token_usage_dict, start_time_llm, end_time_llm
                )
            )

            self.shared_memory.add_step_batch(tool_call_response_memory_steps)
            messages.extend(tool_call_response_msgs)
            step += 1

            # Validate tool call/response matching
            if len(tool_call_response_msgs) - 1 != len(tool_calls):
                logger.warning(
                    f"Tool call/response mismatch: {len(tool_calls)} calls, {len(tool_call_response_msgs)-1} responses"
                )

        return (
            response_message.parsed
            if self.llm_gen_type == "json_schema"
            else response_message.content
        )

    def _handle_text_response(
        self,
        content: str,
        token_usage_dict: Optional[Dict] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Handle text response from LLM."""
        if hasattr(self, "message_manager"):
            return self.message_manager.add_assistant_message(
                content, token_usage_dict, start_time, end_time
            )
        else:
            return {"role": MessageRole.ASSISTANT, "content": content}

    def _handle_tool_calls_and_responses(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        tool_results_raw: List[Any],
        token_usage_dict: Optional[Dict] = None,
        start_time_llm: Optional[datetime] = None,
        end_time_llm: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Handle tool calls and responses from LLM.
        Add tool call and tool response to memory in correct order.
        This is important as it ensures the tool call/response pairs are valid in the LLM's context.
        """
        tool_call_msg = {
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
        tool_call_memory = MemoryStep(
            id=str(uuid4()),
            step_type=MemoryStepType.TOOL_CALL,
            content=None,  # Tool calls don't have content
            timing=Timing(start_time=start_time_llm, end_time=end_time_llm),
            tool_calls=tool_calls_list,
            metadata={"agent": self.name},
            token_usage=token_usage_dict,
            raw_message=tool_call_msg,
        )
        tool_call_response_msgs = [tool_call_msg]
        tool_call_response_memory_steps = [tool_call_memory]

        for i, tool_call in enumerate(tool_calls):
            result = tool_results_raw[i] if i < len(tool_results_raw) else None
            if result is None or isinstance(result, Exception):
                # No result found - create error response
                error_msg = f"Tool execution failed: No result returned - Exception: {str(result)}"
                logger.error(
                    f"Agent: {self.name} - Tool {tool_call.function.name} failed: {error_msg}"
                )
                tool_response_msg = {
                    "role": MessageRole.TOOL,
                    "content": error_msg,
                    "tool_call_id": tool_call.id,
                }
            else:
                func_result, start_time, end_time = result
                tool_response_msg = {
                    "role": MessageRole.TOOL,
                    "content": str(func_result),
                    "tool_call_id": tool_call.id,
                }
                tool_response_memory = MemoryStep(
                    id=str(uuid4()),
                    step_type=MemoryStepType.TOOL_RESPONSE,
                    content=str(func_result),
                    timing=Timing(start_time=start_time, end_time=end_time),
                    tool_calls=[
                        ToolCall(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            args=tool_call.function.arguments,
                        )
                    ],
                    metadata={"agent": self.name},
                    raw_message=tool_response_msg,
                )
            tool_call_response_msgs.append(tool_response_msg)
            tool_call_response_memory_steps.append(tool_response_memory)
        return tool_call_response_msgs, tool_call_response_memory_steps

    async def execute_streaming(
        self,
        user_input: Optional[str] = None,
        additional_materials: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Execute the agent's functionality with streaming output.
        Only supports text generation type.

        Args:
            user_input: User's input message
            additional_materials: Additional materials provided by user

        Yields:
            String chunks of the response as they are generated
        """
        if self.llm_gen_type != "text":
            raise ValueError("Streaming is only supported for text generation type")

        messages = await self._build_messages_from_memory()
        user_request = await self._format_user_request(user_input, additional_materials)

        if user_request:
            if hasattr(self, "message_manager"):
                user_message = self.message_manager.add_user_message(user_request)
            else:
                user_message = {"role": MessageRole.USER, "content": user_request}

            messages.append(user_message)

        # For streaming, we only support simple text responses without tool calls
        accumulated_content = ""
        token_usage_dict = None

        async for chunk in self.llm.generate_streaming_response(
            name=self.name, messages=messages, tools=[]  # No tools for streaming for now
        ):
            if not chunk.is_complete:
                accumulated_content += chunk.content
                yield chunk.content
            else:
                # Final chunk with token usage
                if chunk.token_usage:
                    token_usage_dict = chunk.token_usage.dict()
                break

        # Add the complete response to memory
        if hasattr(self, "message_manager"):
            self.message_manager.add_assistant_message(accumulated_content, token_usage_dict)

    async def _execute_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> Any:
        """
        Execute a single tool call safely.

        Args:
            tool_call: OpenAI tool call object

        Returns:
            Result of tool execution
        """
        function_name = tool_call.function.name

        if function_name not in self._tool_map:
            raise ValueError(
                f"Tool '{function_name}' not found. Available tools: {self.tool_names}"
            )

        tool = self._tool_map[function_name]
        function_args = json.loads(tool_call.function.arguments)

        logger.info(f"{self.name} - Executing tool '{function_name}' with args: {function_args}")

        start_time = datetime.now()
        if asyncio.iscoroutinefunction(tool.func):
            result = await tool.func(**function_args)
        else:
            result = tool.func(**function_args)
        end_time = datetime.now()

        logger.debug(f"{self.name} - Tool '{function_name}' result: {result}")
        return result, start_time, end_time

    def get_memory_summary(self, agent_filter: Optional[str] = None) -> str:
        """
        Get a summary of the shared memory, optionally filtered by agent.

        Args:
            agent_filter: Only include steps from this agent

        Returns:
            Formatted summary of memory steps
        """
        if not self.shared_memory or not self.shared_memory.memory_steps:
            return "No conversation history available."

        summary_parts = []
        for step in sorted(self.shared_memory.memory_steps, key=lambda x: x.timing.start_time):

            # Filter by agent if specified
            if agent_filter and step.metadata:
                if step.metadata.get("agent") != agent_filter:
                    continue

            agent_name = (
                step.metadata.get("agent", MessageRole.SYSTEM)
                if step.metadata
                else MessageRole.SYSTEM
            )
            timestamp = step.timing.start_time.strftime("%H:%M:%S")

            summary_parts.append(
                f"[{timestamp}] {agent_name} ({step.step_type}): {step.content[:100]}..."
            )

        return "\n".join(summary_parts)

    def get_last_agent_output(self, agent_name: Optional[str] = None) -> Optional[str]:
        """
        Get the last output from a specific agent or any agent.

        Args:
            agent_name: Name of agent to get output from, or None for any agent

        Returns:
            Last agent output content or None
        """
        if not self.shared_memory or not self.shared_memory.memory_steps:
            return None

        # Filter assistant steps and sort by time (newest first)
        assistant_steps = [
            step
            for step in self.shared_memory.memory_steps
            if step.step_type == MemoryStepType.ASSISTANT
        ]

        if agent_name:
            assistant_steps = [
                step
                for step in assistant_steps
                if step.metadata and step.metadata.get("agent") == agent_name
            ]

        if not assistant_steps:
            return None

        # Return most recent output
        latest_step = max(assistant_steps, key=lambda x: x.timing.start_time)
        return latest_step.content


class DynamicAgent(BaseAgent):
    """
    Dynamic agent that acts as different roles based on the workflow plan.
    """

    def __init__(self, name: str, system_prompt: str, **kwargs):

        # Use provided tools or default to all available tools
        tools = kwargs.pop("tools", get_available_tools())

        super().__init__(
            name=name,
            system_prompt=system_prompt,
            tools=tools,
            **kwargs,
        )


class WorkflowPlanner:
    """
    LLM-driven workflow planner that determines the optimal sequence of tasks
    for any product design request.
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        memory: Optional[Memory] = None,
        config: Optional[AgentConfig] = None,
    ):
        self.config = config or AgentConfig()
        self.system_prompt = prompt_templates.get_template("workflow_planner").replace(
            "[max_workflow_steps_placeholder]", str(self.config.max_workflow_steps)
        )

        self.planner_agent = DynamicAgent(
            name="workflow_planner",
            system_prompt=self.system_prompt,
            llm_engine=llm_engine,
            llm_gen_type="json_schema",
            llm_json_schema=WorkflowPlan,
            memory=memory,
            tools=[],
        )

    async def plan_workflow(
        self, user_request: str, additional_materials: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze user request and create an optimal workflow plan.

        Args:
            user_request: User's product design request
            additional_materials: Any uploaded materials

        Returns:
            Dict containing workflow steps and execution plan
        """

        try:
            workflow_plan_result = await self.planner_agent.execute(
                user_request, additional_materials
            )

            # The result should be a parsed WorkflowPlan object
            # If return a string, try to parse it as JSON (fallback)
            if isinstance(workflow_plan_result, str):
                workflow_data = json.loads(workflow_plan_result)
                workflow_plan = WorkflowPlan(**workflow_data)
            else:
                workflow_plan = workflow_plan_result

            workflow_steps = []
            for step in workflow_plan.workflow:
                workflow_steps.append(
                    {
                        "step_name": step.step_name,
                        "step_type": step.step_type.value,
                        "step_description": step.step_description,
                        "role_description": step.role_description,
                        "expertise": step.expertise,
                        "depends_on": step.depends_on,
                    }
                )

            return {
                "success": True,
                "workflow": workflow_steps,
                "reasoning": workflow_plan.reasoning,
                "estimated_steps": len(workflow_steps),
            }

        except Exception as e:
            logger.error(f"Error in workflow planning: {e}, falling back to default workflow")

            self.planner_agent.shared_memory.add_step(
                id=str(uuid4()),
                step_type=MemoryStepType.ERROR,
                content=f"Error in workflow planning: {e}",
                timing=Timing(start_time=datetime.now(), end_time=datetime.now()),
                metadata={"agent": self.planner_agent.name},
            )

            # Fallback to default workflow
            return get_default_workflow()


class IntelligentOrchestrator:
    """
    Intelligent orchestrator that uses LLM-driven workflow planning
    and different dynamic agents to execute product design workflows.
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        memory: Optional[Memory] = None,
        config: Optional[AgentConfig] = None,
    ):
        self.shared_memory = memory or Memory()
        self.config = config or AgentConfig()

        self.general_agent = DynamicAgent(
            name="general_agent",
            system_prompt=None,
            llm_engine=llm_engine,
            llm_gen_type="text",
            memory=self.shared_memory,
            tools=[],
        )

        # Safety check agent (optional)
        self.safety_agent = None
        if self.config.enable_safety_check:
            self.safety_agent = DynamicAgent(
                name="safety_agent",
                system_prompt=prompt_templates.get_template("safety_agent"),
                llm_engine=llm_engine,
                llm_gen_type="json_schema",
                llm_json_schema=SafetyCheckResult,
                memory=self.shared_memory,
                tools=[],
            )

        # LLM-driven workflow planner
        self.workflow_planner = WorkflowPlanner(llm_engine, self.shared_memory, self.config)
        self.workflow_plan = None

        # Add current time to memory
        self._add_current_time_to_memory()

        # Initialize message manager for clean memory operations
        self.message_manager = MessageManager(self.shared_memory, "orchestrator")

    def _add_current_time_to_memory(self):
        """Add current time to memory."""
        current_time = datetime.now()
        self.shared_memory.add_step(
            id=str(uuid4()),
            step_type=MemoryStepType.SYSTEM,
            content=f"CURRENT TIME: {current_time.strftime('%Y-%m-%d %H:%M:%S')}",
            timing=Timing(start_time=current_time, end_time=current_time),
            metadata={"agent": "orchestrator"},
            raw_message={
                "role": MessageRole.SYSTEM,
                "content": f"CURRENT TIME: {current_time.strftime('%Y-%m-%d %H:%M:%S')}",
            },
        )

    async def chat_streaming(
        self, user_input: str, additional_materials: Optional[Dict] = None
    ) -> AsyncGenerator[str, None]:
        """
        Handle normal chat conversations with streaming output.

        Args:
            user_input: User's message
            additional_materials: Any uploaded materials

        Yields:
            Response chunks as they are generated
        """
        try:
            messages = self.message_manager.get_conversation_messages([], "orchestrator")
            chat_system_prompt = prompt_templates.get_template("chat_agent")
            system_message = self.message_manager.add_system_message(chat_system_prompt)
            user_message = self.message_manager.add_user_message(user_input)
            messages.append(system_message)
            messages.append(user_message)

            accumulated_content = ""
            token_usage_dict = None

            async for chunk in self.general_agent.llm.generate_streaming_response(
                name="chat_agent", messages=messages
            ):
                if not chunk.is_complete:
                    accumulated_content += chunk.content
                    yield chunk.content
                else:
                    if chunk.token_usage:
                        token_usage_dict = chunk.token_usage.dict()
                    break

            # Add complete response to memory
            self.message_manager.add_assistant_message(accumulated_content, token_usage_dict)

        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            yield f"I apologize, but I encountered an error: {str(e)}"

    async def chat(self, user_input: str, additional_materials: Optional[Dict] = None) -> str:
        """
        Handle normal chat conversations with the user.

        Args:
            user_input: User's message
            additional_materials: Any uploaded materials

        Returns:
            Assistant's response as string
        """
        try:
            messages = self.message_manager.get_conversation_messages([], "orchestrator")
            chat_system_prompt = prompt_templates.get_template("chat_agent")
            system_message = self.message_manager.add_system_message(chat_system_prompt)
            user_message = self.message_manager.add_user_message(user_input)
            messages.append(system_message)
            messages.append(user_message)

            llm_response = await self.general_agent.llm.generate_response(
                name="chat_agent",
                messages=messages,
                llm_gen_type="text",
            )

            response_content = llm_response.message.content
            token_usage_dict = llm_response.token_usage.dict() if llm_response.token_usage else None

            self.message_manager.add_assistant_message(response_content, token_usage_dict)

            return response_content

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

    async def create_product_design_document(
        self, user_input: str, additional_materials: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute an intelligent, LLM-planned product design workflow.

        Args:
            user_input: User's product requirements
            additional_materials: Any uploaded materials

        Returns:
            Dict with final document and process information
        """

        try:
            completed_steps = set()
            workflow_results = {}

            # Optional safety check before any processing
            if self.config.enable_safety_check and self.safety_agent:
                report_stage_start(
                    "Safety Check", "Performing safety validation...", "safety_agent"
                )

                safety_result: SafetyCheckResult = await self.safety_agent.execute(
                    user_input,
                    additional_materials,
                )

                if not safety_result.is_safe:
                    report_error(
                        "Safety Check",
                        f"Safety validation failed: {safety_result.message}",
                        "safety_agent",
                    )
                    return {
                        "success": False,
                        "stage": "safety_check",
                        "message": f"Safety validation failed: {safety_result}",
                        "document": None,
                    }

                completed_steps.add("safety_check")
                workflow_results["safety_check"] = {
                    "is_safe": True,
                    "message": safety_result.message,
                    "confidence_score": safety_result.confidence_score,
                    "approved_content": user_input,
                    "approved_files": additional_materials,
                }
                report_success("Safety Check", "Safety validation passed", "safety_agent")
            else:
                report_warning("Safety Check", "Safety check disabled, skipping validation", None)

            # Plan the workflow using LLM
            report_stage_start("Workflow Planning", "Planning workflow...", "workflow_planner")
            self.workflow_plan = await self.workflow_planner.plan_workflow(
                user_input, additional_materials
            )
            if not self.workflow_plan["success"]:
                report_error(
                    "Workflow Planning", "Failed to create workflow plan", "workflow_planner"
                )
                return {
                    "success": False,
                    "stage": "workflow_planning",
                    "message": "Failed to create workflow plan",
                    "document": None,
                }

            workflow_plan_diagram = generate_workflow_diagram_cli(self.workflow_plan)
            report_success(
                "Workflow Planning",
                f"Workflow planned successfully",
                "workflow_planner",
            )
            report_custom(
                "Workflow Planning",
                f"\n{workflow_plan_diagram}\n",
                "workflow_planner",
                custom_color="light_green",
            )

            # Execute the planned workflow
            report_stage_start("Workflow Execution", "Starting workflow execution...", None)
            remaining_steps = {step["step_name"]: step for step in self.workflow_plan["workflow"]}
            total_steps = len(remaining_steps)
            completed_count = 0

            while remaining_steps:

                ready_steps = []
                for step_name, step in remaining_steps.items():
                    depends_on = step.get("depends_on", [])
                    if not depends_on or all(dep in completed_steps for dep in depends_on):
                        ready_steps.append(step)

                if not ready_steps:
                    # No steps can be executed - circular dependency detected
                    report_error(
                        "Workflow Execution",
                        f"Circular dependency detected. Remaining steps: {list(remaining_steps.keys())}",
                        None,
                    )
                    break

                # Report which steps are starting
                step_names = [step["step_name"] for step in ready_steps]
                if len(step_names) == 1:
                    report_stage_start(
                        "Workflow Execution",
                        f"Processing: {step_names[0]}",
                        f"{step_names[0]}_agent",
                    )
                else:
                    report_stage_start(
                        "Workflow Execution",
                        f"Processing in parallel: {', '.join(step_names)}",
                        None,
                    )

                tasks = []
                for step in ready_steps:
                    task = self._execute_workflow_step(step, user_input, workflow_results)
                    tasks.append(task)

                results = await asyncio.gather(*tasks)

                for i, step_name in enumerate(step_names):
                    workflow_results[step_name] = results[i]
                    completed_steps.add(step_name)
                    remaining_steps.pop(step_name)
                    completed_count += 1
                    report_step_complete(
                        "Workflow Execution",
                        f"Completed: {step_name}",
                        f"{step_name}_agent",
                        completed_count,
                        total_steps,
                    )

                if len(step_names) > 1:
                    report_stage_complete(
                        "Workflow Execution",
                        f"Completed parallel execution of: {', '.join(step_names)}",
                        None,
                    )

            # Generate final document using all workflow results
            report_stage_start(
                "Document Generation",
                "Generating final product design document...",
                "document_generator",
            )
            final_document = await self._generate_final_document(user_input, workflow_results)

            # Get validation result if available
            validation_result = workflow_results.get("validation")
            validation_info = None
            if validation_result and hasattr(validation_result, "model_dump_json"):
                validation_info = validation_result.model_dump_json()

            report_success(
                "Document Generation",
                "Product design document completed successfully",
                "document_generator",
            )

            return {
                "success": True,
                "stage": "completed",
                "document": final_document,
                "workflow_plan": self.workflow_plan,
                "workflow_results": workflow_results,
                "validation_info": validation_info,
            }

        except Exception as e:
            logger.error(f"Error in intelligent workflow: {e}")
            return {
                "success": False,
                "stage": "error",
                "message": f"Workflow error: {str(e)}",
                "document": None,
            }

    async def _execute_workflow_step(
        self,
        step: Dict[str, Any],
        user_input: str,
        previous_results: Optional[Dict] = None,
    ) -> Any:
        """Execute a single workflow step with dependency-aware context."""
        step_name = step["step_name"]
        step_type = step["step_type"]
        step_description = step["step_description"]
        role_description = step["role_description"]
        expertise = step.get("expertise", [])
        depends_on: List[str] = step.get("depends_on", [])

        # Get the appropriate JSON schema for this step type
        json_schema = WorkflowStepType.get_json_schema(step_type)

        # Build context from dependencies
        context = ""
        context += f"Your role description is: {role_description}\n"
        context += f"Your step description is: {step_description}\n"
        context += f"Your expertise is: {expertise}\n"

        if depends_on and previous_results:
            dependency_context = []
            for dep_step in depends_on:
                if dep_step in previous_results:
                    dep_result = previous_results[dep_step]
                    if isinstance(dep_result, str):
                        dependency_context.append(f"{dep_step.upper()} ANALYSIS:\n{dep_result}")
                    elif isinstance(dep_result, BaseModel):
                        dependency_context.append(
                            f"{dep_step.upper()} ANALYSIS:\n{dep_result.model_dump_json()}"
                        )

            if dependency_context:
                context += f"\n\nPREVIOUS ANALYSIS RESULTS:\n" + "\n\n".join(dependency_context)

        # Create dynamic agent for this step
        agent = DynamicAgent(
            name=step_name,
            system_prompt=context,
            llm_engine=LLMEngine(),
            llm_gen_type="json_schema",
            llm_json_schema=json_schema,
            memory=self.shared_memory,
            depends_on=depends_on,
            orchestrator_name="orchestrator",
        )

        # Execute the agent with the original user input and built context
        return await agent.execute(user_input)

    async def _generate_final_document(
        self, user_input: str, workflow_results: Dict[str, Any]
    ) -> str:
        """
        Generate final product design document by synthesizing all workflow results.

        Args:
            user_input: Original user request
            workflow_results: Results from all workflow steps

        Returns:
            Final formatted product design document as markdown
        """
        try:
            analysis_results = self._build_analysis_summary(workflow_results)

            document_prompt_template = prompt_templates.get_template(
                "final_product_design_document"
            )

            final_prompt = document_prompt_template.replace(
                "[analysis_results_placeholder]", analysis_results
            )

            # Increase max tokens for final document generation
            config = AgentConfig(max_tokens=3000)

            document_agent = DynamicAgent(
                name="final_document_generator",
                system_prompt=final_prompt,
                llm_engine=LLMEngine(config=config),
                llm_gen_type="text",
                memory=self.shared_memory,
                depends_on=list(workflow_results.keys()),
                orchestrator_name="orchestrator",
            )
            document_content = await document_agent.execute(
                f"Generate the final product design document for: {user_input}"
            )

            logger.info("✅ Final product design document generated successfully")
            return document_content

        except Exception as e:
            logger.error(f"Error generating final document: {e}")
            return f"Error generating document: {str(e)}"

    def _build_analysis_summary(self, workflow_results: Dict[str, Any]) -> str:
        """
        Build a comprehensive summary of all analysis results for document generation.
        """
        summary_parts = []

        for step_name, result in workflow_results.items():
            if result is None:
                continue

            summary_parts.append(f"\n## {step_name.upper()} ANALYSIS:")

            if isinstance(result, str):
                summary_parts.append(result)
            elif hasattr(result, "model_dump_json"):
                # For Pydantic models, get JSON representation
                try:
                    result_dict = result.model_dump()
                    formatted_result = self._format_model_result(step_name, result_dict)
                    summary_parts.append(formatted_result)
                except Exception as e:
                    logger.warning(f"Could not format {step_name} result: {e}")
                    summary_parts.append(str(result))
            else:
                summary_parts.append(str(result))

            summary_parts.append("\n" + "-" * 50)

        return "\n".join(summary_parts)

    def _format_model_result(self, step_name: str, result_dict: Dict) -> str:
        """
        Format a Pydantic model result for better readability.
        """
        formatted_parts = []

        for key, value in result_dict.items():
            if isinstance(value, list) and value:
                formatted_parts.append(f"**{key.replace('_', ' ').title()}:**")
                for item in value:
                    formatted_parts.append(f"- {item}")
                formatted_parts.append("")
            elif isinstance(value, dict) and value:
                formatted_parts.append(f"**{key.replace('_', ' ').title()}:**")
                for sub_key, sub_value in value.items():
                    formatted_parts.append(f"- {sub_key}: {sub_value}")
                formatted_parts.append("")
            elif value and not isinstance(value, (list, dict)):
                formatted_parts.append(f"**{key.replace('_', ' ').title()}:** {value}")
                formatted_parts.append("")

        return "\n".join(formatted_parts)

    async def update_document_with_feedback(
        self,
        current_document: str,
        user_feedback: str,
        additional_materials: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Update document based on user feedback."""
        try:
            # Optional safety check first
            if self.config.enable_safety_check and self.safety_agent:
                safety_result: SafetyCheckResult = await self.safety_agent.execute(
                    f"Validate this feedback: {user_feedback}",
                    additional_materials,
                )

                if not safety_result.is_safe:
                    return {
                        "success": False,
                        "stage": "safety_check",
                        "message": f"Safety validation failed: {safety_result.message}",
                        "document": current_document,
                    }

            # Update the system prompt
            system_prompt = f"""
            Update this product design document based on user feedback:
            
            CURRENT DOCUMENT:
            {current_document}
            
            Provide an improved version addressing the feedback.
            """

            self.general_agent.system_prompt = system_prompt
            updated_document = await self.general_agent.execute(
                user_feedback,
                additional_materials,
            )

            return {
                "success": True,
                "stage": "updated",
                "document": updated_document,
            }

        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return {
                "success": False,
                "stage": "error",
                "message": f"Update error: {str(e)}",
                "document": current_document,
            }

    def get_statistics_report(self) -> Dict[str, Any]:
        """Get comprehensive statistics report."""
        if not self.shared_memory:
            return {"message": "No memory available for statistics"}

        stats_calculator = AgentStatistics(self.shared_memory)
        return stats_calculator.export_statistics_report()

    def get_efficiency_report(self) -> Dict[str, Any]:
        """Get agent efficiency analysis."""
        if not self.shared_memory:
            return {"message": "No memory available for efficiency analysis"}

        stats_calculator = AgentStatistics(self.shared_memory)
        return stats_calculator.get_agent_efficiency_report()

    def get_cost_estimate(
        self, cost_per_1k_input: float = 0.0015, cost_per_1k_output: float = 0.002
    ) -> Dict[str, Any]:
        """Get cost estimate based on token usage."""
        if not self.shared_memory:
            return {"message": "No memory available for cost estimation"}

        stats_calculator = AgentStatistics(self.shared_memory)
        return stats_calculator.get_cost_estimate(cost_per_1k_input, cost_per_1k_output)


if __name__ == "__main__":

    async def example_usage():

        llm_engine = LLMEngine()
        config = AgentConfig(enable_safety_check=False)

        orchestrator = IntelligentOrchestrator(llm_engine, memory=Memory(), config=config)

        user_input = "I want to create a mobile app for fitness tracking with social features"

        print("\n\n🚀 Starting product design...\n\n")
        result = await orchestrator.create_product_design_document(user_input)
        print(result)

    asyncio.run(example_usage())

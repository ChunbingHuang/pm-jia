"""
Tool management system for agent capabilities.
"""

import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Union, get_type_hints

from dotenv import load_dotenv
from tavily import TavilyClient

from src.pm_jia.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)


class ToolCategory(str, Enum):
    """Categories for organizing tools."""

    MATH = "math"
    WEB = "web"
    FILE = "file"
    API = "api"
    UTILITY = "utility"


class ToolRequirement(str, Enum):
    """Requirements for tool operation."""

    NONE = "none"
    API_KEY = "api_key"
    FILE_SYSTEM = "file_system"


@dataclass
class ToolMetadata:
    """Metadata for tool configuration and requirements."""

    name: str
    description: str
    category: ToolCategory
    requirements: Set[ToolRequirement] = field(default_factory=set)
    required_env_vars: Set[str] = field(default_factory=set)
    enabled: bool = True


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata
        self._validated = False

    def validate_requirements(self) -> bool:
        """Validate that all tool requirements are met."""
        if self._validated:
            return True

        try:
            # Check environment variables
            for env_var in self.metadata.required_env_vars:
                if not os.getenv(env_var):
                    logger.warning(
                        f"Tool '{self.metadata.name}' missing required env var: {env_var}"
                    )
                    return False

            # Check specific requirements
            if ToolRequirement.API_KEY in self.metadata.requirements:
                if not self._has_api_key():
                    logger.warning(f"Tool '{self.metadata.name}' requires API key but none found")
                    return False

            self._validated = True
            return True

        except Exception as e:
            logger.error(f"Error validating tool '{self.metadata.name}': {e}")
            return False

    def _has_api_key(self) -> bool:
        """Check if required API keys are available."""
        return len(self.metadata.required_env_vars) == 0 or any(
            os.getenv(var) for var in self.metadata.required_env_vars
        )

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the tool's main functionality."""
        pass


def tool_wrapper(func):
    """Decorator to wrap tool functions with validation and error handling."""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not self.validate_requirements():
            raise RuntimeError(f"Tool '{self.metadata.name}' requirements not met")

        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing tool '{self.metadata.name}': {e}")
            raise

    return wrapper


# This is a tool for testing only
# class MathTool(BaseTool):
#     """Mathematical operations tool."""

#     def __init__(self):
#         metadata = ToolMetadata(
#             name="add_numbers",
#             description="Add two numbers together",
#             category=ToolCategory.MATH,
#             requirements={ToolRequirement.NONE},
#         )
#         super().__init__(metadata)

#     @tool_wrapper
#     async def execute(self, a: float, b: float) -> float:
#         """
#         Add two numbers together.

#         Args:
#             a: The first number to add
#             b: The second number to add

#         Returns:
#             The sum of the two numbers
#         """
#         return float(a) + float(b)


class WebSearchTool(BaseTool):
    """Web search tool requiring API key."""

    def __init__(self):
        metadata = ToolMetadata(
            name="web_search",
            description="Search the web for information",
            category=ToolCategory.WEB,
            requirements={ToolRequirement.API_KEY},
            required_env_vars={"TAVILY_API_KEY"},
            enabled=False,  # Default to disabled until API key is configured
        )
        super().__init__(metadata)

    @tool_wrapper
    async def execute(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Perform a web search using the Tavily API.

        This function initializes a Tavily client with an API key and performs a search
        query using their search engine. Tavily specializes in providing AI-optimized
        search results with high accuracy and relevance.

        Args:
            query (str): The search query string to be processed by the Tavily's search engine
            max_results (int): Maximum number of search results to return (default: 5)

        Returns:
            dict: A dictionary containing the search results.
            The dictionary includes:
            - title: The title of the search result
            - url: The URL of the search result
            - content: A snippet or content preview
            - score: A relevance score for the search result
            - published_date: The date the content was published (if available)

        Example:
            >>> results = web_search("artificial intelligence trends in 2024")
            >>> print(results)
        """
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        search_results = client.search(
            query,
            max_results=max_results,
        )

        result = ""
        for search_result in search_results["results"]:
            result += f"Title: {search_result.get('title', 'No title')} - Content: {search_result.get('content', 'No content')}\n"

        return result


@dataclass
class Tool:
    """Tool definition for agent integration."""

    name: str
    func: Callable
    description: str
    category: ToolCategory
    requirements: Set[ToolRequirement]
    available: bool = True


class ToolRegistry:
    """Central registry for managing tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools."""
        default_tools = [
            # MathTool(),
            WebSearchTool(),
        ]

        for tool in default_tools:
            self.register_tool(tool)

    def register_tool(self, tool: BaseTool):
        """Register a tool in the registry."""
        if not isinstance(tool, BaseTool):
            raise TypeError("Tool must inherit from BaseTool")

        # Validate requirements during registration
        tool.metadata.enabled = tool.validate_requirements()
        self._tools[tool.metadata.name] = tool

        if tool.metadata.enabled:
            logger.info(f"Registered tool: {tool.metadata.name}")
        else:
            logger.warning(
                f"Registered tool '{tool.metadata.name}' but disabled due to unmet requirements"
            )

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a specific tool by name."""
        return self._tools.get(name)

    def get_available_tools(
        self, category: Optional[ToolCategory] = None, include_disabled: bool = False
    ) -> List[Tool]:
        """Get list of available tools for agents."""
        tools = []

        for tool in self._tools.values():
            if not include_disabled and not tool.metadata.enabled:
                continue

            if category and tool.metadata.category != category:
                continue

            tools.append(
                Tool(
                    name=tool.metadata.name,
                    func=tool.execute,
                    description=tool.metadata.description,
                    category=tool.metadata.category,
                    requirements=tool.metadata.requirements,
                    available=tool.metadata.enabled,
                )
            )

        return tools

    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get tools filtered by category."""
        return self.get_available_tools(category=category)

    def get_enabled_tools(self) -> List[Tool]:
        """Get only enabled tools."""
        return self.get_available_tools(include_disabled=False)

    def list_tool_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all registered tools."""
        status = {}
        for name, tool in self._tools.items():
            status[name] = {
                "enabled": tool.metadata.enabled,
                "category": tool.metadata.category.value,
                "requirements": [req.value for req in tool.metadata.requirements],
                "required_env_vars": list(tool.metadata.required_env_vars),
                "description": tool.metadata.description,
            }
        return status


# Global tool registry instance
_tool_registry = ToolRegistry()


def get_available_tools(
    category: Optional[ToolCategory] = None, include_disabled: bool = False
) -> List[Tool]:
    """Get list of available tools for agents."""
    return _tool_registry.get_available_tools(category, include_disabled)


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _tool_registry


def register_custom_tool(tool: BaseTool):
    """Register a custom tool with the global registry."""
    _tool_registry.register_tool(tool)


def _python_type_to_json_schema(python_type: type) -> Dict[str, Any]:
    """Convert Python type annotations to JSON Schema format."""
    type_mapping = {
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        List: {"type": "array"},
        Dict: {"type": "object"},
    }

    # Handle Union types (e.g., Optional[str] = Union[str, None])
    if hasattr(python_type, "__origin__"):
        if python_type.__origin__ is Union:
            # For Optional types, get the non-None type
            non_none_types = [t for t in python_type.__args__ if t is not type(None)]
            if len(non_none_types) == 1:
                return _python_type_to_json_schema(non_none_types[0])
            else:
                # Multiple types, use string as fallback
                return {"type": "string"}
        elif python_type.__origin__ is list:
            return {"type": "array"}
        elif python_type.__origin__ is dict:
            return {"type": "object"}

    return type_mapping.get(python_type, {"type": "string"})


def _parse_docstring_params(docstring: str) -> Dict[str, str]:
    """Parse parameter descriptions from docstring Args section."""
    param_descriptions = {}
    if not docstring:
        return param_descriptions

    lines = docstring.strip().split("\n")
    in_args_section = False

    for line in lines:
        line = line.strip()
        if line.lower().startswith("args:"):
            in_args_section = True
            continue
        elif line.lower().startswith(
            ("returns:", "return:", "raises:", "raise:", "examples:", "example:")
        ):
            in_args_section = False
            continue

        if in_args_section and ":" in line:
            # Parse parameter line like "param_name: Description"
            parts = line.split(":", 1)
            if len(parts) == 2:
                param_name = parts[0].strip()
                description = parts[1].strip()
                param_descriptions[param_name] = description

    return param_descriptions


def _extract_function_schema(func: Callable) -> Dict[str, Any]:
    """Extract function schema from callable for OpenAI tool format."""
    try:
        # Get function signature and type hints
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Extract function name and docstring
        func_name = func.__name__ if hasattr(func, "__name__") else "unknown_function"
        if func_name == "execute":
            # For tool execute methods, use the tool's metadata name
            if hasattr(func, "__self__") and hasattr(func.__self__, "metadata"):
                func_name = func.__self__.metadata.name

        docstring = func.__doc__ or "No description available"
        # Get main description (first paragraph)
        description_lines = []
        for line in docstring.strip().split("\n"):
            line = line.strip()
            if not line:
                break
            if line.lower().startswith(("args:", "returns:", "raises:")):
                break
            description_lines.append(line)

        description = " ".join(description_lines) if description_lines else func_name

        # Parse parameter descriptions from docstring
        param_descriptions = _parse_docstring_params(docstring)

        # Build parameters schema
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip 'self' parameter
            if param_name == "self":
                continue

            # Get type annotation
            param_type = type_hints.get(param_name, str)

            # Convert to JSON schema
            param_schema = _python_type_to_json_schema(param_type)

            # Add parameter description from docstring or default
            param_schema["description"] = param_descriptions.get(
                param_name, f"The {param_name} parameter"
            )

            properties[param_name] = param_schema

            # Check if parameter is required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": func_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    except Exception as e:
        logger.error(f"Error extracting schema from function {func}: {e}")
        return {
            "name": "unknown_function",
            "description": "Function schema extraction failed",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }


def tools_to_openai_format(
    tools: Optional[List[Tool]] = None,
    category: Optional[ToolCategory] = None,
    include_disabled: bool = False,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert tools to OpenAI function calling format.

    Args:
        tools: List of tools to convert. If None, gets all available tools.
        category: Filter tools by category
        include_disabled: Whether to include disabled tools
        strict: Whether to make tools strict for structured output compatibility

    Returns:
        List of tool definitions in OpenAI format

    Example:
        >>> tools = tools_to_openai_format()
        >>> print(tools[0])
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two numbers together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "Parameter a"},
                        "b": {"type": "number", "description": "Parameter b"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    """
    if tools is None:
        tools = get_available_tools(category=category, include_disabled=include_disabled)

    openai_tools = []

    for tool in tools:
        try:
            function_schema = _extract_function_schema(tool.func)

            openai_tool = {"type": "function", "function": function_schema}

            # Add strict mode if requested
            if strict:
                openai_tool["function"]["strict"] = True
                openai_tool["function"]["parameters"]["additionalProperties"] = False
                all_properties = list(openai_tool["function"]["parameters"]["properties"].keys())
                openai_tool["function"]["parameters"]["required"] = all_properties

            openai_tools.append(openai_tool)

        except Exception as e:
            logger.error(f"Error converting tool '{tool.name}' to OpenAI format: {e}")
            continue

    return openai_tools


def get_openai_tools(
    category: Optional[ToolCategory] = None, include_disabled: bool = False
) -> List[Dict[str, Any]]:
    """
    Convenience function to get tools in OpenAI format.

    Args:
        category: Filter tools by category
        include_disabled: Whether to include disabled tools

    Returns:
        List of tool definitions in OpenAI format
    """
    return tools_to_openai_format(category=category, include_disabled=include_disabled)


if __name__ == "__main__":
    tools = get_openai_tools()
    print(tools)

"""
Contains all configuration values and settings for the project.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GeneralConfig:
    project_root: str = Path(__file__).parent.parent.parent
    prompt_template_name: str = "general_prompt_light.yaml"
    prompt_template_path: str = project_root / "src" / "pm_jia" / "prompts" / prompt_template_name
    sessions_path: str = project_root / "sessions"


@dataclass
class LoggerConfig:
    name: str = "pm_jia"
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"  # 2025-06-28 10:00:00
    save_to_file: bool = False


@dataclass
class AgentConfig:
    """Agent system configuration."""

    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.05
    max_tokens: int = 1000
    confidence_threshold: float = 0.8
    max_steps: int = 5
    verbose: bool = True
    enable_safety_check: bool = True
    document_approval_score: int = 7
    max_workflow_steps: int = 5
    max_tool_usage: int = 2


@dataclass
class ProcessorConfig:
    """Content processor configuration."""

    supported_image_types: tuple = ("jpg", "jpeg", "png")
    supported_table_types: tuple = ("csv", "xlsx", "xls")
    supported_text_types: tuple = ("txt", "md", "json")
    max_file_size_mb: int = 2
    gemini_model: str = "gemini-2.5-flash"

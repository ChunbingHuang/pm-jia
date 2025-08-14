"""
Progress Management System
Centralized progress reporting system that can be used across different interfaces (CLI, Web, API)
"""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from rich.markup import escape


class ProgressType(Enum):
    """Types of progress messages."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    IN_PROGRESS = "in_progress"
    CALLING_FUNCTION = "calling_function"
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    CUSTOM = "custom"


@dataclass
class ProgressMessage:
    """Structured progress message."""

    timestamp: datetime
    stage: str
    message: str
    progress_type: ProgressType
    agent_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    step_count: Optional[int] = None
    total_steps: Optional[int] = None
    custom_color: Optional[str] = None


class ProgressHandler(ABC):
    """Abstract base class for progress handlers."""

    @abstractmethod
    def handle_progress(self, progress_msg: ProgressMessage):
        """Handle a progress message."""
        pass


class ConsoleProgressHandler(ProgressHandler):
    """Handler for console/CLI output using rich formatting."""

    def __init__(self, console=None):
        if not console:
            from rich.console import Console
        self.console = console or Console()

    def handle_progress(self, progress_msg: ProgressMessage):
        """Handle progress with rich console formatting."""
        formatted_msg = self._format_message(progress_msg)
        self.console.print(formatted_msg)

    def _format_message(self, msg: ProgressMessage) -> str:
        """Format message with rich markup."""
        if msg.progress_type == ProgressType.SUCCESS:
            return f"[bold green]{msg.message}[/bold green]"
        elif msg.progress_type == ProgressType.WARNING:
            return f"[yellow]{msg.message}[/yellow]"
        elif msg.progress_type == ProgressType.ERROR:
            return f"[red]{msg.message}[/red]"
        elif msg.progress_type in [ProgressType.STAGE_START, ProgressType.STEP_START]:
            return f"[cyan][{msg.stage}][/cyan] {msg.message}"
        elif msg.progress_type in [ProgressType.STAGE_COMPLETE, ProgressType.STEP_COMPLETE]:
            return f"[green][{msg.stage}][/green] {msg.message}"
        elif msg.progress_type == ProgressType.CALLING_FUNCTION:
            function_name = (
                msg.metadata.get("function_name", "unknown") if msg.metadata else "unknown"
            )
            message = f"- Agent: {msg.agent_name} - Calling function: <{function_name}>"
            return f"       [blue]{escape(message)}[/blue] - {msg.message}"
        elif msg.progress_type == ProgressType.CUSTOM:
            return f"[{msg.custom_color}]{msg.message}[/{msg.custom_color}]"
        else:
            return f"[cyan][{msg.stage}][/cyan] {msg.message}"


class WebProgressHandler(ProgressHandler):
    """Handler for web interface using WebSocket or SSE."""

    def __init__(self, websocket_sender: Optional[Callable] = None):
        self.websocket_sender = websocket_sender
        self.messages: List[ProgressMessage] = []

    def handle_progress(self, progress_msg: ProgressMessage):
        """Handle progress for web interface."""
        self.messages.append(progress_msg)

        # Send via WebSocket if available
        if self.websocket_sender:
            message_data = {
                "timestamp": progress_msg.timestamp.isoformat(),
                "stage": progress_msg.stage,
                "message": progress_msg.message,
                "type": progress_msg.progress_type.value,
                "agent_name": progress_msg.agent_name,
                "step_count": progress_msg.step_count,
                "total_steps": progress_msg.total_steps,
                "metadata": progress_msg.metadata,
            }
            self.websocket_sender(message_data)


class LoggerProgressHandler(ProgressHandler):
    """Handler that writes to logger for debugging/persistence."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def handle_progress(self, progress_msg: ProgressMessage):
        """Handle progress via logger."""
        log_msg = f"[{progress_msg.stage}] {progress_msg.message}"

        if progress_msg.progress_type == ProgressType.ERROR:
            self.logger.error(log_msg)
        elif progress_msg.progress_type == ProgressType.WARNING:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)


class ProgressManager:
    """Centralized progress manager that can be used globally."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for global access."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if not self._initialized:
            self.handlers: List[ProgressHandler] = []
            self.active = True
            self._initialized = True

    def add_handler(self, handler: ProgressHandler):
        """Add a progress handler."""
        self.handlers.append(handler)

    def remove_handler(self, handler: ProgressHandler):
        """Remove a progress handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)

    def clear_handlers(self):
        """Clear all handlers."""
        self.handlers.clear()

    def set_active(self, active: bool):
        """Enable or disable progress reporting."""
        self.active = active

    def report_progress(
        self,
        stage: str,
        message: str,
        progress_type: ProgressType = ProgressType.INFO,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        step_count: Optional[int] = None,
        total_steps: Optional[int] = None,
        custom_color: Optional[str] = None,
    ):
        """Report progress to all registered handlers."""
        if not self.active or not self.handlers:
            return

        progress_msg = ProgressMessage(
            timestamp=datetime.now(),
            stage=stage,
            message=message,
            progress_type=progress_type,
            agent_name=agent_name,
            metadata=metadata,
            step_count=step_count,
            total_steps=total_steps,
            custom_color=custom_color,
        )

        for handler in self.handlers:
            try:
                handler.handle_progress(progress_msg)
            except Exception as e:
                # Handler errors don't break the main process
                print(f"Error in progress handler: {e}")

    def report_stage_start(self, stage: str, message: str, agent_name: Optional[str] = None):
        """Report stage start."""
        self.report_progress(stage, message, ProgressType.STAGE_START, agent_name)

    def report_stage_complete(self, stage: str, message: str, agent_name: Optional[str] = None):
        """Report stage completion."""
        self.report_progress(stage, message, ProgressType.STAGE_COMPLETE, agent_name)

    def report_step_start(self, stage: str, message: str, agent_name: Optional[str] = None):
        """Report step start."""
        self.report_progress(stage, message, ProgressType.STEP_START, agent_name)

    def report_step_complete(
        self,
        stage: str,
        message: str,
        agent_name: Optional[str] = None,
        step_count: Optional[int] = None,
        total_steps: Optional[int] = None,
    ):
        """Report step completion."""
        self.report_progress(
            stage,
            message,
            ProgressType.STEP_COMPLETE,
            agent_name,
            step_count=step_count,
            total_steps=total_steps,
        )

    def report_success(self, stage: str, message: str, agent_name: Optional[str] = None):
        """Report success message."""
        self.report_progress(stage, message, ProgressType.SUCCESS, agent_name)

    def report_warning(self, stage: str, message: str, agent_name: Optional[str] = None):
        """Report warning message."""
        self.report_progress(stage, message, ProgressType.WARNING, agent_name)

    def report_error(self, stage: str, message: str, agent_name: Optional[str] = None):
        """Report error message."""
        self.report_progress(stage, message, ProgressType.ERROR, agent_name)

    def report_calling_function(
        self,
        stage: str,
        message: str,
        agent_name: Optional[str] = None,
        function_name: Optional[str] = None,
    ):
        """Report calling function."""
        self.report_progress(
            stage,
            message,
            ProgressType.CALLING_FUNCTION,
            agent_name,
            metadata={"function_name": function_name},
        )

    def report_custom(
        self,
        stage: str,
        message: str,
        agent_name: Optional[str] = None,
        custom_color: Optional[str] = None,
    ):
        """Report custom message."""
        self.report_progress(
            stage, message, ProgressType.CUSTOM, agent_name, custom_color=custom_color
        )


# Global instance
progress_manager = ProgressManager()


# Functions for global access
def report_progress(
    stage: str,
    message: str,
    progress_type: ProgressType = ProgressType.INFO,
    agent_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    step_count: Optional[int] = None,
    total_steps: Optional[int] = None,
):
    """Global function to report progress."""
    progress_manager.report_progress(
        stage, message, progress_type, agent_name, metadata, step_count, total_steps
    )


def report_stage_start(stage: str, message: str, agent_name: Optional[str] = None):
    """Global function to report stage start."""
    progress_manager.report_stage_start(stage, message, agent_name)


def report_stage_complete(stage: str, message: str, agent_name: Optional[str] = None):
    """Global function to report stage completion."""
    progress_manager.report_stage_complete(stage, message, agent_name)


def report_step_complete(
    stage: str,
    message: str,
    agent_name: Optional[str] = None,
    step_count: Optional[int] = None,
    total_steps: Optional[int] = None,
):
    """Global function to report step completion."""
    progress_manager.report_step_complete(stage, message, agent_name, step_count, total_steps)


def report_success(stage: str, message: str, agent_name: Optional[str] = None):
    """Global function to report success."""
    progress_manager.report_success(stage, message, agent_name)


def report_warning(stage: str, message: str, agent_name: Optional[str] = None):
    """Global function to report warning."""
    progress_manager.report_warning(stage, message, agent_name)


def report_error(stage: str, message: str, agent_name: Optional[str] = None):
    """Global function to report error."""
    progress_manager.report_error(stage, message, agent_name)


def report_calling_function(
    stage: str, message: str, agent_name: Optional[str] = None, function_name: Optional[str] = None
):
    """Global function to report calling function."""
    progress_manager.report_calling_function(stage, message, agent_name, function_name)


def report_custom(
    stage: str, message: str, agent_name: Optional[str] = None, custom_color: Optional[str] = None
):
    """Global function to report custom message."""
    progress_manager.report_custom(stage, message, agent_name, custom_color)

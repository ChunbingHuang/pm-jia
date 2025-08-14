"""
Session Manager for PM-JIA

Handles session management for both CLI and API modes with consistent behavior.
Manages session persistence, memory handling, and data storage.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import GeneralConfig
from .logger import setup_logger


class SessionManager:
    """Session manager for both CLI and API modes."""

    def __init__(self, mode: str = "cli"):
        """
        Initialize session manager.

        Args:
            mode: Either "cli" or "api" to handle logging/output differently
        """
        self.mode = mode
        self.sessions_root = GeneralConfig.sessions_path
        self.sessions_root.mkdir(exist_ok=True, parents=True)

        if mode == "api":
            self.logger = setup_logger(__name__)
            self._log_info = self.logger.info
            self._log_debug = self.logger.debug
            self._log_error = self.logger.error
            self._log_warning = self.logger.warning
        else:
            from rich.console import Console

            self._console = Console()
            self._log_info = lambda msg: self._console.print(f"[blue]{msg}[/blue]")
            self._log_debug = lambda msg: self._console.print(f"[dim]{msg}[/dim]")
            self._log_error = lambda msg: self._console.print(f"[red]{msg}[/red]")
            self._log_warning = lambda msg: self._console.print(f"[yellow]{msg}[/yellow]")

    def list_sessions(self) -> List[Dict]:
        """List all existing sessions."""
        sessions = []
        for session_dir in self.sessions_root.iterdir():
            if session_dir.is_dir():
                session_file = session_dir / "session.json"
                if session_file.exists():
                    try:
                        with open(session_file) as f:
                            session_data = json.load(f)
                            sessions.append(session_data)
                    except Exception as e:
                        self._log_warning(f"Could not load session {session_file}: {e}")
        return sorted(sessions, key=lambda x: x.get("created_at", ""))

    def load_session(self, session_id: str) -> Optional[Dict]:
        """Load a specific session."""
        session_dir = self.sessions_root / session_id
        session_file = session_dir / "session.json"
        if session_file.exists():
            try:
                with open(session_file) as f:
                    return json.load(f)
            except Exception as e:
                self._log_error(f"Error loading session: {e}")
        return None

    def save_session_metadata(self, session_id: str, session_data: Dict[str, Any]):
        """Save session metadata to disk."""
        session_dir = self.sessions_root / session_id
        session_dir.mkdir(exist_ok=True, parents=True)

        # Handle different data formats for compatibility
        if isinstance(session_data.get("created_at"), datetime):
            created_at = session_data["created_at"].isoformat()
        else:
            created_at = session_data.get("created_at", datetime.now().isoformat())

        if isinstance(session_data.get("last_activity"), datetime):
            last_activity = session_data["last_activity"].isoformat()
        elif "last_updated" in session_data:
            last_activity = session_data["last_updated"]
        else:
            last_activity = datetime.now().isoformat()

        metadata = {
            "session_id": session_data.get("session_id") or session_data.get("id"),
            "project_name": session_data["project_name"],
            "created_at": created_at,
            "last_activity": last_activity,
            "status": session_data.get("status", "active"),
        }

        session_file = session_dir / "session.json"
        with open(session_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def create_new_session(self, project_name: str) -> Dict:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "project_name": project_name,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "status": "active",
        }
        self.save_session_metadata(session_id, session_data)
        return session_data

    def save_session_memory(self, session_id: str, memory_data: Dict[str, Any]):
        """Save session memory to disk."""
        session_dir = self.sessions_root / session_id
        session_dir.mkdir(exist_ok=True, parents=True)

        memory_file = session_dir / "memory.json"
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, default=str)

        self._log_debug(f"Saved memory for session {session_id}")

    def load_session_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session memory from disk."""
        session_dir = self.sessions_root / session_id
        memory_file = session_dir / "memory.json"

        if memory_file.exists():
            try:
                with open(memory_file, "r", encoding="utf-8") as f:
                    memory_data = json.load(f)
                self._log_debug(f"Loaded memory for session {session_id}")
                return memory_data
            except Exception as e:
                self._log_error(f"Error loading memory for session {session_id}: {e}")

        return None

    async def auto_save_session_memory(self, session_id: str, orchestrator):
        """Automatically save session memory if session exists (API mode)."""
        if orchestrator and orchestrator.shared_memory:
            try:
                if asyncio.iscoroutinefunction(orchestrator.shared_memory.export_to_dict):
                    memory_data = await orchestrator.shared_memory.export_to_dict()
                else:
                    memory_data = orchestrator.shared_memory.export_to_dict()

                self.save_session_memory(session_id, memory_data)
                self._log_debug(f"Auto-saved memory for session {session_id}")
            except Exception as e:
                self._log_error(f"Error auto-saving memory for session {session_id}: {e}")

    def save_document(self, session_id: str, document: str, version: int = 0) -> str:
        """Save a generated document."""
        session_dir = self.sessions_root / session_id
        doc_dir = session_dir / "documents"
        doc_dir.mkdir(exist_ok=True, parents=True)

        filename = f"{session_id}_v{version}.md"
        doc_path = doc_dir / filename

        # Version +1 if file already exists
        while doc_path.exists():
            version += 1
            filename = f"{session_id}_v{version}.md"
            doc_path = doc_dir / filename

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(document)

        return str(doc_path)

    def get_session_document(self, session_id: str) -> Optional[str]:
        """Load existing session document from disk."""
        session_dir = self.sessions_root / session_id
        doc_dir = session_dir / "documents"

        if doc_dir.exists():
            try:
                # Find the latest document file
                doc_files = list(doc_dir.glob(f"{session_id}_v*.md"))
                if doc_files:
                    # Sort by version number and get the latest
                    latest_doc = max(doc_files, key=lambda x: int(x.stem.split("_v")[1]))
                    with open(latest_doc, "r", encoding="utf-8") as f:
                        document = f.read()
                    self._log_debug(f"Loaded document for session {session_id}")
                    return document
            except Exception as e:
                self._log_error(f"Error loading document for session {session_id}: {e}")

        return None

    def save_session_stats(self, session_id: str, stats_data: Dict[str, Any]):
        """Save session statistics to disk."""
        session_dir = self.sessions_root / session_id
        session_dir.mkdir(exist_ok=True, parents=True)

        stats_file = session_dir / "stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2, default=str)

        self._log_debug(f"Saved stats for session {session_id}")

    def load_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session statistics from disk."""
        session_dir = self.sessions_root / session_id
        stats_file = session_dir / "stats.json"

        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    stats_data = json.load(f)
                self._log_debug(f"Loaded stats for session {session_id}")
                return stats_data
            except Exception as e:
                self._log_error(f"Error loading stats for session {session_id}: {e}")

        return None

    def save_workflow_plan(self, session_id: str, workflow_plan_data: Dict[str, Any]):
        """Save workflow plan data to disk."""
        session_dir = self.sessions_root / session_id
        session_dir.mkdir(exist_ok=True, parents=True)

        workflow_file = session_dir / "workflow.json"
        with open(workflow_file, "w", encoding="utf-8") as f:
            json.dump(workflow_plan_data, f, indent=2, default=str)

        self._log_debug(f"Saved workflow plan for session {session_id}")

    def load_workflow_plan(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load workflow plan data from disk."""
        session_dir = self.sessions_root / session_id
        workflow_file = session_dir / "workflow.json"

        if workflow_file.exists():
            try:
                with open(workflow_file, "r", encoding="utf-8") as f:
                    workflow_data = json.load(f)
                self._log_debug(f"Loaded workflow plan for session {session_id}")
                return workflow_data
            except Exception as e:
                self._log_error(f"Error loading workflow plan for session {session_id}: {e}")

        return None

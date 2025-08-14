"""
FastAPI web API for PM-JIA product management assistant.
Provides endpoints for session management, chat, and document generation.
"""

import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.pm_jia.agent import IntelligentOrchestrator
from src.pm_jia.config import AgentConfig, GeneralConfig, ProcessorConfig
from src.pm_jia.llm import LLMEngine
from src.pm_jia.logger import setup_logger
from src.pm_jia.memory import Memory
from src.pm_jia.model import (
    ChatRequest,
    ChatResponse,
    DocumentGenerationRequest,
    DocumentGenerationResponse,
    SessionCreateRequest,
    SessionResponse,
    SessionSaveRequest,
)
from src.pm_jia.processor import MaterialProcessor
from src.pm_jia.progress_manager import progress_manager
from src.pm_jia.session import SessionManager
from src.pm_jia.utils import clean_markdown_document, generate_workflow_diagram_web, get_stats_data

logger = setup_logger(__name__)

app = FastAPI(
    title="PM-JIA API", description="AI-powered product design assistant", version="1.0.0"
)

# Enable CORS for web client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for active sessions
sessions: Dict[str, Dict[str, Any]] = {}


session_manager = SessionManager(mode="api")


def create_orchestrator(config: Optional[Dict[str, Any]] = None) -> IntelligentOrchestrator:
    """Create a new orchestrator instance with optional configuration."""
    try:
        llm_engine = LLMEngine()
        memory = Memory()

        # Apply custom config if provided
        agent_config = AgentConfig()
        if config:
            for key, value in config.items():
                if hasattr(agent_config, key):
                    setattr(agent_config, key, value)

        return IntelligentOrchestrator(llm_engine, memory, agent_config)
    except Exception as e:
        logger.error(f"Error creating orchestrator: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create orchestrator: {str(e)}")


async def process_uploaded_materials(files: List[UploadFile]) -> Dict[str, str]:
    """Process uploaded files and return extracted content."""
    materials = {}
    processor_config = ProcessorConfig()
    processor = MaterialProcessor(processor_config)

    for file in files:
        try:
            file_content = await file.read()
            file.file.seek(0)

            if file.filename:
                processed_content = await processor.process_single_material(
                    file.filename, file_content
                )
                materials[file.filename] = processed_content
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            materials[file.filename] = f"Error processing file: {str(e)}"

    return materials


# API Endpoints


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "1.0.0"}


@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new chat session."""
    try:
        session_id = str(uuid.uuid4())
        orchestrator = create_orchestrator(request.config)

        session_data = {
            "id": session_id,
            "project_name": request.project_name,
            "orchestrator": orchestrator,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "status": "active",
        }

        sessions[session_id] = session_data

        session_manager.save_session_metadata(session_id, session_data)

        logger.info(f"Created session {session_id} for project {request.project_name}")

        return SessionResponse(
            session_id=session_id,
            project_name=request.project_name,
            created_at=session_data["created_at"].isoformat(),
            status="active",
        )

    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session information."""

    if session_id in sessions:
        session = sessions[session_id]
        return SessionResponse(
            session_id=session_id,
            project_name=session["project_name"],
            created_at=session["created_at"].isoformat(),
            status=session["status"],
        )

    session_dir = GeneralConfig.sessions_path / session_id
    session_file = session_dir / "session.json"

    if session_file.exists():
        try:
            with open(session_file) as f:
                session_data = json.load(f)

            orchestrator = create_orchestrator()

            memory_data = session_manager.load_session_memory(session_id)
            if memory_data and orchestrator.shared_memory:
                try:
                    if asyncio.iscoroutinefunction(orchestrator.shared_memory.import_from_dict):
                        await orchestrator.shared_memory.import_from_dict(memory_data)
                    else:
                        if hasattr(orchestrator.shared_memory, "import_from_dict"):
                            orchestrator.shared_memory.import_from_dict(memory_data)
                    logger.info(f"Loaded memory for session {session_id}")
                except Exception as e:
                    logger.error(f"Error loading memory for session {session_id}: {e}")

            sessions[session_id] = {
                "id": session_id,
                "project_name": session_data["project_name"],
                "orchestrator": orchestrator,
                "created_at": datetime.fromisoformat(session_data["created_at"]),
                "last_activity": datetime.now(),
                "status": "active",
            }

            return SessionResponse(
                session_id=session_id,
                project_name=session_data["project_name"],
                created_at=session_data["created_at"],
                status="active",
            )
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load session: {str(e)}")

    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/sessions/{session_id}/save")
async def save_session(session_id: str, request: SessionSaveRequest):
    """Save additional data to session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        sessions[session_id]["saved_data"] = request.data
        sessions[session_id]["last_activity"] = datetime.now()

        return {"message": "Session data saved successfully"}

    except Exception as e:
        logger.error(f"Error saving session data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/switch")
async def switch_to_session(session_id: str, current_session_id: Optional[str] = None):
    """Switch to a session, auto-saving the current session's memory."""
    try:
        # Auto-save current session's memory if provided
        if current_session_id and current_session_id in sessions:
            current_session = sessions[current_session_id]
            orchestrator = current_session["orchestrator"]
            await session_manager.auto_save_session_memory(current_session_id, orchestrator)
            logger.info(f"Auto-saved memory for previous session {current_session_id}")

        # Load the target session (this will also load its memory)
        session_response = await get_session(session_id)

        return {
            "message": f"Switched to session {session_id}",
            "session": session_response,
            "previous_session_saved": current_session_id is not None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching to session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    try:
        # Check if session exists in memory first
        if session_id in sessions:
            # Auto-save memory before deletion
            session = sessions[session_id]
            orchestrator = session["orchestrator"]
            await session_manager.auto_save_session_memory(session_id, orchestrator)
            del sessions[session_id]

        # Check if session exists on disk
        session_dir = GeneralConfig.sessions_path / session_id
        session_file = session_dir / "session.json"

        if not session_file.exists() and session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Delete session files from disk
        if session_dir.exists():
            import shutil

            shutil.rmtree(session_dir)
            logger.info(f"Deleted session directory: {session_dir}")

        return {"message": "Session deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat messages with the AI assistant."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        session = sessions[request.session_id]
        orchestrator = session["orchestrator"]
        session["last_activity"] = datetime.now()

        response = await orchestrator.chat(request.message, request.additional_materials)

        # Auto-save session memory after chat
        await session_manager.auto_save_session_memory(request.session_id, orchestrator)

        return ChatResponse(
            session_id=request.session_id, response=response, timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream/{session_id}")
async def chat_stream(session_id: str, message: str, additional_materials: Optional[str] = None):
    """Stream chat responses in real-time."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    async def generate_stream():
        try:
            session = sessions[session_id]
            orchestrator = session["orchestrator"]
            session["last_activity"] = datetime.now()

            materials = json.loads(additional_materials) if additional_materials else None

            # Get the streaming response from the orchestrator
            async for chunk in orchestrator.chat_streaming(message, materials):
                chunk_data = {
                    "session_id": session_id,
                    "chunk": chunk,
                    "is_complete": False,
                    "timestamp": datetime.now().isoformat(),
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            # Auto-save session memory after streaming chat
            await session_manager.auto_save_session_memory(session_id, orchestrator)

            # Send completion signal
            final_chunk_data = {
                "session_id": session_id,
                "chunk": "",
                "is_complete": True,
                "timestamp": datetime.now().isoformat(),
            }
            yield f"data: {json.dumps(final_chunk_data)}\n\n"

        except Exception as e:
            error_data = {
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/documents/generate", response_model=DocumentGenerationResponse)
async def generate_document(request: DocumentGenerationRequest):
    """Generate a product design document."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        session = sessions[request.session_id]
        orchestrator = session["orchestrator"]
        session["last_activity"] = datetime.now()

        result = await orchestrator.create_product_design_document(
            request.user_input, request.additional_materials
        )

        if result["success"] and result.get("document"):
            result["document"] = clean_markdown_document(result["document"])
            document_path = session_manager.save_document(request.session_id, result["document"])
            logger.info(f"Document saved to: {document_path}")

        # Auto-save session memory after document generation
        await session_manager.auto_save_session_memory(request.session_id, orchestrator)

        return DocumentGenerationResponse(
            session_id=request.session_id,
            success=result["success"],
            document=result.get("document"),
            stage=result["stage"],
            message=result.get("message"),
            workflow_plan=result.get("workflow_plan"),
        )

    except Exception as e:
        logger.error(f"Error generating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/generate/stream/{session_id}")
async def generate_document_stream(
    session_id: str, user_input: str, additional_materials: Optional[str] = None
):
    """Generate a product design document with streaming progress updates."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    async def generate_progress_stream():
        try:
            session = sessions[session_id]
            orchestrator = session["orchestrator"]
            session["last_activity"] = datetime.now()

            # Create an async queue for real-time progress streaming
            progress_queue = asyncio.Queue()

            class StreamingWebProgressHandler:
                def __init__(self, queue):
                    self.queue = queue
                    self.messages = []

                def handle_progress(self, progress_msg):
                    """Handle progress and add to queue for immediate streaming."""
                    self.messages.append(progress_msg)
                    progress_data = {
                        "session_id": session_id,
                        "timestamp": progress_msg.timestamp.isoformat(),
                        "stage": progress_msg.stage,
                        "message": progress_msg.message,
                        "type": progress_msg.progress_type.value,
                        "agent_name": progress_msg.agent_name,
                        "step_count": progress_msg.step_count,
                        "total_steps": progress_msg.total_steps,
                        "is_complete": False,
                    }
                    try:
                        self.queue.put_nowait(progress_data)
                    except asyncio.QueueFull:
                        pass  # Skip if queue is full

            streaming_handler = StreamingWebProgressHandler(progress_queue)
            progress_manager.clear_handlers()
            progress_manager.add_handler(streaming_handler)
            progress_manager.set_active(True)

            materials = json.loads(additional_materials) if additional_materials else None

            # Start document generation as a background task
            generation_task = asyncio.create_task(
                orchestrator.create_product_design_document(user_input, materials)
            )

            # Stream progress messages in real-time
            result = None
            while not generation_task.done():
                try:
                    progress_data = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
                    yield f"data: {json.dumps(progress_data)}\n\n"
                except asyncio.TimeoutError:
                    # No progress message, continue checking if task is done
                    pass

            result = await generation_task

            # Send any remaining progress messages in the queue
            while not progress_queue.empty():
                try:
                    progress_data = progress_queue.get_nowait()
                    yield f"data: {json.dumps(progress_data)}\n\n"
                except asyncio.QueueEmpty:
                    break

            if result["success"] and result.get("document"):
                result["document"] = clean_markdown_document(result["document"])
                document_path = session_manager.save_document(session_id, result["document"])
                logger.info(f"Document saved to: {document_path}")

                # Save stats and workflow diagram
                try:
                    stats_report = orchestrator.get_statistics_report()
                    efficiency_report = orchestrator.get_efficiency_report()
                    cost_estimate = orchestrator.get_cost_estimate()

                    stats_data = get_stats_data(
                        session_id,
                        stats_report,
                        efficiency_report,
                        cost_estimate,
                    )

                    session_manager.save_session_stats(session_id, stats_data)

                    # Save workflow plan
                    if hasattr(orchestrator, "workflow_plan") and orchestrator.workflow_plan:
                        session_manager.save_workflow_plan(
                            session_id, {"workflow_plan": orchestrator.workflow_plan}
                        )

                except Exception as e:
                    logger.error(f"Error saving stats/workflow for session {session_id}: {e}")

            # Auto-save session memory after document generation
            await session_manager.auto_save_session_memory(session_id, orchestrator)

            final_data = {
                "session_id": session_id,
                "success": result["success"],
                "document": result.get("document"),
                "stage": result["stage"],
                "message": result.get("message"),
                "workflow_plan": result.get("workflow_plan"),
                "is_complete": True,
                "timestamp": datetime.now().isoformat(),
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            error_data = {
                "error": str(e),
                "session_id": session_id,
                "is_complete": True,
                "timestamp": datetime.now().isoformat(),
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            # Clean up progress manager
            progress_manager.clear_handlers()

    return StreamingResponse(
        generate_progress_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/documents/upload/{session_id}")
async def upload_materials(session_id: str, files: List[UploadFile] = File(...)):
    """Upload and process materials for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        materials = await process_uploaded_materials(files)

        session = sessions[session_id]
        if "uploaded_materials" not in session:
            session["uploaded_materials"] = {}
        session["uploaded_materials"].update(materials)
        session["last_activity"] = datetime.now()

        return {
            "session_id": session_id,
            "processed_files": list(materials.keys()),
            "message": f"Successfully processed {len(materials)} files",
        }

    except Exception as e:
        logger.error(f"Error uploading materials: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/memory")
async def get_session_memory(session_id: str):
    """Get session memory/conversation history."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        session = sessions[session_id]
        orchestrator = session["orchestrator"]

        if hasattr(orchestrator.shared_memory, "export_to_dict"):
            if asyncio.iscoroutinefunction(orchestrator.shared_memory.export_to_dict):
                memory_data = await orchestrator.shared_memory.export_to_dict()
            else:
                memory_data = orchestrator.shared_memory.export_to_dict()
        else:
            # Fallback: create memory data from memory steps
            memory_data = {
                "steps": [step.to_dict() for step in orchestrator.shared_memory.memory_steps],
                "system_prompt": orchestrator.shared_memory.system_prompt,
            }

        return {"session_id": session_id, "memory": memory_data}

    except Exception as e:
        logger.error(f"Error getting session memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/statistics")
async def get_session_statistics(session_id: str):
    """Get comprehensive session statistics and usage reports."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        session = sessions[session_id]
        orchestrator = session["orchestrator"]

        stats_report = orchestrator.get_statistics_report()
        efficiency_report = orchestrator.get_efficiency_report()
        cost_estimate = orchestrator.get_cost_estimate()

        agent_stats = stats_report.get("agent_statistics", {})
        summary = agent_stats.get("summary", {})

        # Get workflow diagram - load workflow plan and generate diagram
        workflow_data = session_manager.load_workflow_plan(session_id)
        workflow_diagram = None

        if workflow_data and "workflow_plan" in workflow_data:
            workflow_diagram = generate_workflow_diagram_web(workflow_data["workflow_plan"])
        elif hasattr(orchestrator, "workflow_plan") and orchestrator.workflow_plan:
            workflow_diagram = generate_workflow_diagram_web(orchestrator.workflow_plan)

        response = {
            "session_id": session_id,
            "summary": {
                "total_agents_used": summary.get("total_agents_used", 0),
                "total_execution_time_seconds": round(
                    summary.get("total_execution_time_ms", 0) / 1000, 2
                ),
                "total_tokens": summary.get("total_tokens_used", 0),
                "input_tokens": summary.get("total_input_tokens", 0),
                "output_tokens": summary.get("total_output_tokens", 0),
            },
            "cost_estimate": {
                "input_cost_usd": cost_estimate.get("input_cost_usd", 0),
                "output_cost_usd": cost_estimate.get("output_cost_usd", 0),
                "total_cost_usd": cost_estimate.get("total_cost_usd", 0),
            },
            "agents_breakdown": agent_stats.get("by_agent", {}),
            "detailed_statistics": stats_report,
            "efficiency_report": efficiency_report,
            "detailed_cost_estimate": cost_estimate,
            "workflow_diagram": workflow_diagram,
        }

        return response

    except Exception as e:
        logger.error(f"Error getting session statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/document")
async def get_session_document_data(session_id: str):
    """Get existing session document with stats and workflow if available."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        document = session_manager.get_session_document(session_id)
        stats = session_manager.load_session_stats(session_id)
        workflow_data = session_manager.load_workflow_plan(session_id)

        # Generate workflow diagram from saved plan
        workflow_diagram = None
        if workflow_data and "workflow_plan" in workflow_data:
            workflow_diagram = generate_workflow_diagram_web(workflow_data["workflow_plan"])

        return {
            "session_id": session_id,
            "document": document,
            "statistics": stats,
            "workflow_diagram": workflow_diagram,
            "has_document": document is not None,
            "has_statistics": stats is not None,
            "has_workflow": workflow_diagram is not None,
        }

    except Exception as e:
        logger.error(f"Error getting session document data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """List all sessions (both active and saved)."""
    session_list = []

    # Add active in-memory sessions
    for session_id, session_data in sessions.items():
        session_list.append(
            {
                "session_id": session_id,
                "project_name": session_data["project_name"],
                "created_at": session_data["created_at"].isoformat(),
                "last_activity": session_data["last_activity"].isoformat(),
                "status": session_data["status"],
            }
        )

    # Add saved sessions from disk
    sessions_root = GeneralConfig.sessions_path
    if sessions_root.exists():
        for session_dir in sessions_root.iterdir():
            if session_dir.is_dir():
                session_file = session_dir / "session.json"
                if session_file.exists():
                    try:
                        with open(session_file) as f:
                            session_data = json.load(f)
                            # Avoid duplicates from in-memory sessions
                            if not any(
                                s["session_id"] == session_data["session_id"] for s in session_list
                            ):
                                session_list.append(
                                    {
                                        "session_id": session_data["session_id"],
                                        "project_name": session_data["project_name"],
                                        "created_at": session_data["created_at"],
                                        "last_activity": session_data.get(
                                            "last_updated", session_data["created_at"]
                                        ),
                                        "status": session_data.get("status", "saved"),
                                    }
                                )
                    except Exception as e:
                        logger.warning(f"Could not load session {session_file}: {e}")

    return {"sessions": sorted(session_list, key=lambda x: x["created_at"], reverse=True)}


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return {"error": exc.detail, "status_code": exc.status_code}


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}


if __name__ == "__main__":
    uvicorn.run("src.pm_jia.api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

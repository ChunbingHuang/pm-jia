#!/usr/bin/env python3
"""
PM-JIA Command Line Interface

Interactive CLI for the PM-JIA product design assistant with session management,
real-time chat, and document generation capabilities.
"""

import asyncio
import os
import sys
from typing import Dict, List

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Set Cli mode to True
os.environ["CLI_MODE"] = "True"

from src.pm_jia.agent import IntelligentOrchestrator
from src.pm_jia.config import AgentConfig
from src.pm_jia.llm import LLMEngine
from src.pm_jia.memory import Memory
from src.pm_jia.progress_manager import ConsoleProgressHandler, progress_manager
from src.pm_jia.session import SessionManager
from src.pm_jia.utils import clean_markdown_document, generate_workflow_diagram_cli, get_stats_data

console = Console()


class PMJIAConsole:
    """Main PM-JIA CLI application."""

    def __init__(self):
        self.session_manager = SessionManager(mode="api")
        self.current_session = None
        self.orchestrator = None

    def display_banner(self):
        """Display welcome banner."""
        banner = Text("PM-JIA", style="bold cyan")
        banner.append(" - AI-Powered Product Design Assistant", style="bold white")

        panel = Panel(banner, subtitle="Interactive Command Line Interface", border_style="cyan")
        console.print(panel)
        console.print()

    def display_sessions_table(self, sessions: List[Dict]):
        """Display sessions in a formatted table."""
        if not sessions:
            console.print("[yellow]No existing sessions found.[/yellow]")
            return

        table = Table(title="Available Sessions")
        table.add_column("#", style="cyan", no_wrap=True, width=3)
        table.add_column("ID", style="cyan", no_wrap=True, width=8)
        table.add_column("Project Name", style="green")
        table.add_column("Created", style="blue")
        table.add_column("Last Updated", style="magenta")
        table.add_column("Status", style="yellow")

        for i, session in enumerate(sessions, 1):
            session_id_short = session["session_id"][:8]
            created = session.get("created_at", "Unknown")[:19].replace("T", " ")
            updated = session.get("last_updated", "Unknown")[:19].replace("T", " ")

            table.add_row(
                str(i),
                session_id_short,
                session["project_name"],
                created,
                updated,
                session.get("status", "unknown"),
            )

        console.print(table)
        console.print()

    def display_generation_stats(self, stats_data: Dict):
        """Display brief statistics after document generation."""
        if not stats_data:
            return

        console.print()
        console.print(
            Panel(
                "[bold green]Generation Statistics[/bold green]",
                title="Performance Summary",
                border_style="green",
            )
        )

        # Summary statistics table
        summary_table = Table(title="Overall Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        if "summary" in stats_data:
            summary = stats_data["summary"]

            summary_table.add_row("Total Agents Used", str(summary.get("total_agents_used", 0)))
            summary_table.add_row(
                "Total Execution Time", f"{summary.get('total_execution_time_seconds', 0):.2f} s"
            )
            summary_table.add_row("Total Tokens", f"{summary.get('total_tokens', 0):,}")
            summary_table.add_row("Input Tokens", f"{summary.get('input_tokens', 0):,}")
            summary_table.add_row("Output Tokens", f"{summary.get('output_tokens', 0):,}")
        elif "agent_statistics" in stats_data and "summary" in stats_data["agent_statistics"]:
            # Fallback for old format
            summary = stats_data["agent_statistics"]["summary"]

            summary_table.add_row("Total Agents Used", str(summary.get("total_agents_used", 0)))
            summary_table.add_row(
                "Total Execution Time", f"{summary.get('total_execution_time_ms', 0) / 1000:.2f} s"
            )
            summary_table.add_row("Total Tokens", f"{summary.get('total_tokens_used', 0):,}")
            summary_table.add_row("Input Tokens", f"{summary.get('total_input_tokens', 0):,}")
            summary_table.add_row("Output Tokens", f"{summary.get('total_output_tokens', 0):,}")

        console.print(summary_table)

        # Cost estimation
        if "cost_estimate" in stats_data:
            cost_data = stats_data["cost_estimate"]
            if "total_cost_usd" in cost_data:
                console.print()
                cost_table = Table(title="Cost Estimation")
                cost_table.add_column("Cost Type", style="cyan")
                cost_table.add_column("Amount (USD)", style="green")

                cost_table.add_row("Input Tokens", f"${cost_data.get('input_cost_usd', 0):.6f}")
                cost_table.add_row("Output Tokens", f"${cost_data.get('output_cost_usd', 0):.6f}")
                cost_table.add_row(
                    "Total Cost", f"[bold]${cost_data.get('total_cost_usd', 0):.6f}[/bold]"
                )

                console.print(cost_table)

        # Agent breakdown
        agent_data = None
        if "agents_breakdown" in stats_data:
            agent_data = stats_data["agents_breakdown"]
        elif "agent_statistics" in stats_data and "by_agent" in stats_data["agent_statistics"]:
            # Fallback for old format
            agent_data = stats_data["agent_statistics"]["by_agent"]

        if agent_data:
            console.print()
            agent_table = Table(title="Agents by Token Usage")
            agent_table.add_column("Agent", style="cyan")
            agent_table.add_column("Tokens", style="green")
            agent_table.add_column("Steps", style="blue")
            agent_table.add_column("Time (s)", style="magenta")

            sorted_agents = sorted(
                agent_data.items(), key=lambda x: x[1].get("total_tokens", 0), reverse=True
            )

            for agent_name, agent_stats in sorted_agents:
                agent_table.add_row(
                    agent_name,
                    f"{agent_stats.get('total_tokens', 0):,}",
                    str(agent_stats.get("total_steps", 0)),
                    f"{agent_stats.get('total_time_ms', 0) / 1000:.2f}",
                )

            console.print(agent_table)

        console.print()

    async def select_or_create_session(self) -> Dict:
        """Allow user to select existing session or create new one."""
        sessions = self.session_manager.list_sessions()

        console.print("[bold]Session Management[/bold]")
        console.print()

        self.display_sessions_table(sessions)

        if sessions:
            choice = Prompt.ask("Choose an option", choices=["new", "select"], default="new")

            if choice == "select":
                session_numbers = [str(i) for i in range(1, len(sessions) + 1)]
                session_choice = Prompt.ask(
                    f"Enter session number (1-{len(sessions)})", choices=session_numbers
                )
                selected_index = int(session_choice) - 1
                return sessions[selected_index]

        # Create new session
        project_name = Prompt.ask("[green]Enter project name[/green]")
        return self.session_manager.create_new_session(project_name)

    async def initialize_orchestrator(self):
        """Initialize the orchestrator for the current session."""
        try:
            console.print("[blue]Initializing AI orchestrator...[/blue]")

            llm_engine = LLMEngine()
            memory = Memory()

            # Load existing memory if available
            session_id = self.current_session["session_id"]
            existing_memory = self.session_manager.load_session_memory(session_id)
            if existing_memory:
                console.print("[blue]Loading previous conversation history...[/blue]")
                try:
                    if hasattr(memory, "import_from_dict"):
                        await memory.import_from_dict(existing_memory)
                    else:
                        console.print(
                            "[yellow]Note: Memory loading not fully supported yet[/yellow]"
                        )
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not restore memory: {e}[/yellow]")

            config = AgentConfig(enable_safety_check=False)
            self.orchestrator = IntelligentOrchestrator(llm_engine, memory, config)

            console.print("[green]Orchestrator initialized successfully![/green]")

        except Exception as e:
            console.print(f"[red]Error initializing orchestrator: {e}[/red]")
            console.print("[yellow]Please check your API configuration and try again.[/yellow]")
            sys.exit(1)

    async def show_greeting_message(self):
        """Show greeting or welcome back message."""
        session_id = self.current_session["session_id"]
        project_name = self.current_session["project_name"]

        existing_memory = self.session_manager.load_session_memory(session_id)

        if existing_memory:
            # Welcome back message
            console.print()
            console.print("[blue]Assistant[/blue]: ", end="")
            welcome_msg = f"Welcome back to your project '{project_name}'! I remember our previous conversations. How can I help you today?"

            for char in welcome_msg:
                console.print(char, end="")
            console.print()
            console.print()
        else:
            # First time greeting
            console.print()
            console.print("[blue]Assistant[/blue]: ", end="")
            greeting_msg = f"Hello! Welcome to your new project '{project_name}'. I'm here to help you with product design and development. What would you like to work on today?"

            for char in greeting_msg:
                console.print(char, end="")
            console.print()
            console.print()

    async def chat_mode(self):
        """Interactive chat mode with streaming responses."""
        console.print()
        console.print(
            Panel(
                "[bold green]Chat Mode[/bold green]\n"
                "Type your messages and get real-time responses.\n\n"
                "Commands:\n"
                "   /generate (document generation)\n"
                "   /session (session management)\n"
                "   /stats (view statistics)\n"
                "   /workflow (view workflow)\n"
                "   /document (view document)\n"
                "   /quit (exit)\n",
                title="Interactive Chat",
                border_style="green",
            )
        )
        console.print()

        await self.show_greeting_message()

        while True:
            try:
                user_input = Prompt.ask("[cyan]You[/cyan]")

                if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
                    break
                elif user_input.lower() == "/generate":
                    await self.document_generation_mode()
                    continue
                elif user_input.lower() == "/session":
                    await self.session_management_mode()
                    continue
                elif user_input.lower() == "/stats":
                    await self.display_stats_mode()
                    continue
                elif user_input.lower() == "/workflow":
                    await self.display_workflow_mode()
                    continue
                elif user_input.lower() == "/document":
                    await self.display_document_mode()
                    continue
                elif user_input.strip() == "":
                    continue

                console.print("[blue]Assistant[/blue]: ", end="")

                accumulated_response = ""
                async for chunk in self.orchestrator.chat_streaming(user_input):
                    console.print(chunk, end="")
                    accumulated_response += chunk

                console.print()
                console.print()

                # Auto-save memory after each chat interaction
                if self.orchestrator and self.orchestrator.shared_memory:
                    try:
                        memory_data = await self.orchestrator.shared_memory.export_to_dict()
                        self.session_manager.save_session_memory(
                            self.current_session["session_id"], memory_data
                        )
                    except Exception as e:
                        # Silent save - don't interrupt the chat flow
                        pass

            except KeyboardInterrupt:
                console.print("\n[yellow]Chat interrupted. Type /quit to exit.[/yellow]")
            except EOFError:
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

    async def session_management_mode(self):
        """Session management mode for switching sessions."""
        console.print()
        console.print(
            Panel(
                "[bold cyan]Session Management Mode[/bold cyan]\n"
                "Switch to a different session or create a new one.",
                title="Session Management",
                border_style="cyan",
            )
        )
        console.print()

        # Save current session's memory before potentially switching
        if self.current_session and self.orchestrator and self.orchestrator.shared_memory:
            try:
                console.print("[blue]Saving current session's memory...[/blue]")
                memory_data = await self.orchestrator.shared_memory.export_to_dict()
                self.session_manager.save_session_memory(
                    self.current_session["session_id"], memory_data
                )
                console.print("[green]Current session memory saved![/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save current memory: {e}[/yellow]")

        # Present session options
        sessions = self.session_manager.list_sessions()

        console.print("[bold]What would you like to do?[/bold]")
        console.print("1. Switch to existing session")
        console.print("2. Create new session")
        console.print("3. Return to current session")
        console.print()

        choice = Prompt.ask("Choose an option", choices=["1", "2", "3"], default="3")

        if choice == "3":
            console.print("[green]Returning to current session...[/green]")
            return

        new_session = None

        if choice == "1" and sessions:
            # Switch to existing session
            self.display_sessions_table(sessions)

            session_numbers = [str(i) for i in range(1, len(sessions) + 1)]
            session_choice = Prompt.ask(
                f"Enter session number (1-{len(sessions)})", choices=session_numbers
            )
            selected_index = int(session_choice) - 1
            new_session = sessions[selected_index]

        elif choice == "2":
            # Create new session
            project_name = Prompt.ask("[green]Enter project name[/green]")
            new_session = self.session_manager.create_new_session(project_name)

        elif choice == "1" and not sessions:
            console.print(
                "[yellow]No existing sessions found. Creating new session instead.[/yellow]"
            )
            project_name = Prompt.ask("[green]Enter project name[/green]")
            new_session = self.session_manager.create_new_session(project_name)

        if new_session and new_session["session_id"] != self.current_session["session_id"]:
            # Switch to new session
            console.print(f"[green]Switching to session:[/green] {new_session['project_name']}")
            self.current_session = new_session

            # Reinitialize orchestrator with new session
            await self.initialize_orchestrator()

            console.print("[green]Session switched successfully![/green]")

            await self.show_greeting_message()
        else:
            console.print("[green]Staying in current session...[/green]")

    async def document_generation_mode(self):
        """Document generation mode with progress display."""
        console.print()
        console.print(
            Panel(
                "[bold magenta]Document Generation Mode[/bold magenta]\n"
                "Generate a comprehensive product design document.",
                title="Document Generation",
                border_style="magenta",
            )
        )
        console.print()

        product_idea = Prompt.ask(
            "[green]What kind of product do you want to design?[/green]\n"
            "Describe your idea in detail"
        )

        if not product_idea.strip():
            console.print("[yellow]No product idea provided. Returning to chat mode.[/yellow]")
            return

        console.print()
        console.print(f"[blue]Generating product design document for:[/blue] {product_idea}")
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:

            task = progress.add_task("[bold green]Generation started...[/bold green] ", total=None)

            try:
                result = await self._generate_document_with_progress(product_idea, progress, task)

                if result["success"]:
                    console.print("[green]Document generation completed successfully![/green]")
                    cleaned_document = clean_markdown_document(result["document"])
                    document_path = self.session_manager.save_document(
                        self.current_session["session_id"], cleaned_document
                    )
                    console.print(f"[blue]Document saved to:[/blue] {document_path}")

                    # Auto-save memory after document generation
                    if self.orchestrator and self.orchestrator.shared_memory:
                        try:
                            memory_data = await self.orchestrator.shared_memory.export_to_dict()
                            self.session_manager.save_session_memory(
                                self.current_session["session_id"], memory_data
                            )
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not save memory: {e}[/yellow]")

                    # Generate and save generation stats and workflow diagram
                    if self.orchestrator and self.orchestrator.shared_memory:
                        try:
                            stats_report = self.orchestrator.get_statistics_report()
                            efficiency_report = self.orchestrator.get_efficiency_report()
                            cost_estimate = self.orchestrator.get_cost_estimate()

                            stats_data = get_stats_data(
                                self.current_session["session_id"],
                                stats_report,
                                efficiency_report,
                                cost_estimate,
                            )

                            self.session_manager.save_session_stats(
                                self.current_session["session_id"], stats_data
                            )
                            console.print(f"[green]Stats saved.[/green]")
                        except Exception as e:
                            console.print(
                                f"[yellow]Warning: Could not generate stats: {e}[/yellow]"
                            )
                            stats_data = {}

                        try:
                            workflow_data = {
                                "workflow_plan": self.orchestrator.workflow_plan,
                            }
                            self.session_manager.save_workflow_plan(
                                self.current_session["session_id"], workflow_data
                            )
                            console.print(f"[green]Workflow saved.[/green]")
                        except Exception as e:
                            console.print(
                                f"[yellow]Warning: Could not generate workflow: {e}[/yellow]"
                            )
                            workflow_data = {}

                    # Display generation statistics
                    if stats_data:
                        try:
                            self.display_generation_stats(stats_data)
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not display stats: {e}[/yellow]")

                else:
                    console.print(
                        f"[red]Document generation failed:[/red] {result.get('message', 'Unknown error')}"
                    )

            except Exception as e:
                progress.update(task, description="Error occurred during generation")
                console.print(f"[red]Error during document generation: {e}[/red]")

        console.print()
        console.print("[blue]Returning to chat mode...[/blue]")
        console.print()

    async def _generate_document_with_progress(self, product_idea: str, progress: Progress, task):
        """Generate document while updating progress display."""

        # Set up progress manager with console handler
        console_handler = ConsoleProgressHandler(console)
        progress_manager.clear_handlers()
        progress_manager.add_handler(console_handler)
        progress_manager.set_active(True)

        progress.update(task, description="[bold green]Agent working...[/bold green]")

        try:
            result = await self.orchestrator.create_product_design_document(product_idea)
        finally:
            progress_manager.clear_handlers()

        progress.update(task, description="[bold green]Document generation completed![/bold green]")

        return result

    async def display_stats_mode(self):
        """Display session statistics."""
        session_id = self.current_session["session_id"]
        try:
            stats = self.session_manager.load_session_stats(session_id)
        except Exception as e:
            console.print(f"[red]Error loading stats: {e}[/red]")
            return

        if stats:
            self.display_generation_stats(stats)
        else:
            console.print("[yellow]No stats found for this session.[/yellow]")

    async def display_workflow_mode(self):
        """Display workflow diagram."""
        session_id = self.current_session["session_id"]
        try:
            workflow_data = self.session_manager.load_workflow_plan(session_id)
        except Exception as e:
            console.print(f"[red]Error loading workflow: {e}[/red]")
            return

        if workflow_data and "workflow_plan" in workflow_data:
            # Generate diagram from saved workflow plan
            workflow_diagram = generate_workflow_diagram_cli(workflow_data["workflow_plan"])
            console.print(workflow_diagram, style="green")
        else:
            console.print("[yellow]No workflow found for this session.[/yellow]")

        console.print()

    async def display_document_mode(self):
        """Display document."""
        session_id = self.current_session["session_id"]
        try:
            document = self.session_manager.get_session_document(session_id)
        except Exception as e:
            console.print(f"[red]Error loading document: {e}[/red]")
            return

        if document:
            console.print(document)
        else:
            console.print("[yellow]No document found for this session.[/yellow]")

    async def run(self):
        """Main application loop."""
        self.display_banner()

        try:
            # Session selection/creation
            self.current_session = await self.select_or_create_session()

            console.print(
                f"[green]Selected session:[/green] {self.current_session['project_name']}"
            )
            console.print(f"[blue]Session ID:[/blue] {self.current_session['session_id'][:8]}...")

            # Initialize orchestrator
            await self.initialize_orchestrator()

            # Start chat mode
            await self.chat_mode()

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
        finally:
            # Save session data and memory
            if self.current_session:
                self.session_manager.save_session_metadata(
                    self.current_session["session_id"], self.current_session
                )

                # Save memory if orchestrator exists
                if self.orchestrator and self.orchestrator.shared_memory:
                    try:
                        console.print()
                        console.print()
                        console.print("[blue]Saving conversation history...[/blue]")
                        memory_data = await self.orchestrator.shared_memory.export_to_dict()
                        self.session_manager.save_session_memory(
                            self.current_session["session_id"], memory_data
                        )
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not save memory: {e}[/yellow]")


@click.command()
@click.option("--debug", is_flag=True, help="Enable debug mode")
def main(debug):
    """PM-JIA Interactive Command Line Interface."""
    if debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    app = PMJIAConsole()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()

"""
Utility functions.
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_yaml(file_path: Union[str, Path]) -> Dict:
    """Load YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, file_path: Union[str, Path]) -> None:
    """Save data to YAML file."""
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def clean_markdown_document(document: str) -> str:
    """Clean up the generated markdown document by removing code blocks and explanations."""
    if not document:
        return document

    # Remove markdown code blocks that wrap the entire document
    lines = document.strip().split("\n")
    cleaned_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip opening markdown code blocks
        if line.startswith("```markdown") or line == "```":
            i += 1
            continue

        # Skip obvious explanations at the start
        if (
            i < 3
            and any(
                phrase in line.lower()
                for phrase in [
                    "here is your",
                    "here's your",
                    "generated document",
                    "markdown document",
                    "document has been generated",
                    "this document",
                    "below is",
                ]
            )
            and not line.startswith("#")
        ):
            i += 1
            continue

        cleaned_lines.append(lines[i])
        i += 1

    # Join the cleaned lines
    cleaned_document = "\n".join(cleaned_lines)

    # Remove trailing explanations and metadata
    import re

    # Remove lines that look like AI generation metadata
    trailing_patterns = [
        r"<final_product_design_document>.*",
        r"</final_product_design_document>.*",
        r"\n.*generated with.*$",
        r"\n.*ai assistance.*$",
        r"\n.*created by.*$",
        r"\n.*this document.*generated.*$",
        r"\n.*markdown.*generated.*$",
    ]

    for pattern in trailing_patterns:
        cleaned_document = re.sub(pattern, "", cleaned_document, flags=re.IGNORECASE | re.MULTILINE)

    # Remove excessive empty lines (more than 2 consecutive)
    cleaned_document = re.sub(r"\n{3,}", "\n\n", cleaned_document)

    # Ensure document starts with meaningful content (not empty lines)
    cleaned_document = cleaned_document.strip()

    return cleaned_document


def generate_workflow_diagram_cli(workflow_plan: Dict[str, Any]) -> str:
    """Generate a text-based diagram from workflow plan.

    Args:
        workflow_plan: Dictionary containing workflow structure with steps and dependencies

    Returns:
        String representation of the workflow diagram
    """
    if not workflow_plan or "workflow" not in workflow_plan:
        return "No workflow plan available"

    steps = workflow_plan["workflow"]
    if not steps:
        return "Empty workflow"

    step_map = {step["step_name"]: step for step in steps}

    root_steps = []
    for step in steps:
        depends_on = step.get("depends_on", []) or []
        if not depends_on:
            root_steps.append(step["step_name"])

    # If no root steps found, use first step
    if not root_steps:
        root_steps = [steps[0]["step_name"]]

    levels = []
    processed = set()
    current_level = root_steps[:]

    while current_level:
        levels.append(current_level[:])
        processed.update(current_level)

        next_level = []
        for step in steps:
            step_name = step["step_name"]
            if step_name in processed:
                continue

            depends_on = step.get("depends_on", []) or []
            if all(dep in processed for dep in depends_on):
                next_level.append(step_name)

        current_level = next_level

    # Generate diagram
    diagram_lines = []
    diagram_lines.append("Workflow Diagram:")
    diagram_lines.append("=" * 50)

    for i, level in enumerate(levels):
        if len(level) == 1:
            # Single step
            step_name = level[0]
            step_info = step_map[step_name]
            diagram_lines.append(f"[{step_name}: {step_info['step_type']}]")
        else:
            # Parallel steps
            step_infos = [f"{name}: {step_map[name]['step_type']}" for name in level]
            diagram_lines.append(f"[{' | '.join(step_infos)}]")

        # Add arrow to next level (except for last level)
        if i < len(levels) - 1:
            diagram_lines.append("    |")
            diagram_lines.append("    v")

    # Add reasoning if available
    if "reasoning" in workflow_plan and workflow_plan["reasoning"]:
        diagram_lines.append("")
        diagram_lines.append("Reasoning:")
        diagram_lines.append("-" * 20)
        diagram_lines.append(workflow_plan["reasoning"])

    return "\n".join(diagram_lines)


def generate_workflow_diagram_web(workflow_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a web-friendly workflow diagram structure."""
    if not workflow_plan or "workflow" not in workflow_plan:
        return {"nodes": [], "edges": [], "layout": "hierarchical"}

    steps = workflow_plan["workflow"]
    if not steps:
        return {"nodes": [], "edges": [], "layout": "hierarchical"}

    step_map = {step["step_name"]: step for step in steps}

    # Create nodes
    nodes = []
    for i, step in enumerate(steps):
        nodes.append(
            {
                "id": step["step_name"],
                "label": step["step_name"],
                "type": step["step_type"],
                "description": step["step_description"],
                "role": step["role_description"],
                "expertise": step.get("expertise", []),
                "level": 0,
            }
        )

    # Create edges based on dependencies
    edges = []
    for step in steps:
        depends_on = step.get("depends_on") or []
        # Ensure depends_on is a list and not None
        if depends_on and isinstance(depends_on, (list, tuple)):
            for dependency in depends_on:
                if dependency:  # Make sure dependency is not None or empty
                    edges.append(
                        {"from": dependency, "to": step["step_name"], "type": "dependency"}
                    )

    # Calculate levels for hierarchical layout
    processed = set()
    level = 0

    # Find root nodes (no dependencies)
    current_level_nodes = []
    for step in steps:
        depends_on = step.get("depends_on") or []
        if not depends_on or not isinstance(depends_on, (list, tuple)):
            current_level_nodes.append(step["step_name"])

    # If no root nodes found, use first step
    if not current_level_nodes and steps:
        current_level_nodes = [steps[0]["step_name"]]

    while current_level_nodes:
        for node_name in current_level_nodes:
            for node in nodes:
                if node["id"] == node_name:
                    node["level"] = level

        processed.update(current_level_nodes)
        level += 1

        # Find next level nodes
        next_level_nodes = []
        for step in steps:
            step_name = step["step_name"]
            if step_name in processed:
                continue

            depends_on = step.get("depends_on") or []
            if isinstance(depends_on, (list, tuple)) and all(
                dep in processed for dep in depends_on if dep
            ):
                next_level_nodes.append(step_name)

        current_level_nodes = next_level_nodes

    return {
        "nodes": nodes,
        "edges": edges,
        "layout": "hierarchical",
        "reasoning": workflow_plan.get("reasoning", ""),
        "total_steps": len(steps),
    }


def get_stats_data(
    session_id: str,
    stats_report: Dict[str, Any],
    efficiency_report: Dict[str, Any],
    cost_estimate: Dict[str, Any],
) -> Dict[str, Any]:
    """Get stats data from stats report."""
    agent_stats = stats_report.get("agent_statistics", {})
    summary = agent_stats.get("summary", {})

    stats_data = {
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
    }
    return stats_data

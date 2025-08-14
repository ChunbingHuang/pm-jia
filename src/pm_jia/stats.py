"""
Statistics calculation module for agent performance analysis.
"""

from typing import Any, Dict, Optional

from src.pm_jia.memory import Memory


class AgentStatistics:
    """
    Calculate and manage agent performance statistics.
    """

    def __init__(self, memory: Optional[Memory] = None):
        self.memory = memory

    def calculate_agent_statistics(self) -> Dict[str, Any]:
        """Calculate agent-specific statistics from memory steps."""
        if not self.memory or not self.memory.memory_steps:
            return {}

        agent_stats = {}

        # Group steps by agent
        for step in self.memory.memory_steps:
            if not step.metadata or "agent" not in step.metadata:
                continue

            agent_name = step.metadata["agent"]
            if agent_name not in agent_stats:
                agent_stats[agent_name] = {
                    "total_steps": 0,
                    "assistant_steps": 0,  # Only steps with content (not tool calls)
                    "tool_call_steps": 0,  # Steps where content is None but tools are used
                    "total_time_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "step_types": {},
                }

            # Count step types
            step_type = step.step_type.value
            agent_stats[agent_name]["step_types"][step_type] = (
                agent_stats[agent_name]["step_types"].get(step_type, 0) + 1
            )

            # Count total steps
            agent_stats[agent_name]["total_steps"] += 1

            # Distinguish between assistant steps with content vs tool call steps
            if step.step_type.value == "assistant":
                agent_stats[agent_name]["assistant_steps"] += 1
            elif step.step_type.value == "tool_call":
                agent_stats[agent_name]["tool_call_steps"] += 1

            # Calculate time used (convert to milliseconds for consistency)
            if step.timing and step.timing.duration:
                duration_ms = step.timing.duration * 1000  # Convert seconds to ms
                agent_stats[agent_name]["total_time_ms"] += duration_ms

            # Calculate token usage
            if step.token_usage:
                agent_stats[agent_name]["input_tokens"] += step.token_usage.get("input_tokens", 0)
                agent_stats[agent_name]["output_tokens"] += step.token_usage.get("output_tokens", 0)
                agent_stats[agent_name]["total_tokens"] += step.token_usage.get("total_tokens", 0)

        # Add summary statistics
        total_agents = len(agent_stats)
        total_time_all_agents = sum(stats["total_time_ms"] for stats in agent_stats.values())
        total_tokens_all_agents = sum(stats["total_tokens"] for stats in agent_stats.values())
        total_input_tokens = sum(stats["input_tokens"] for stats in agent_stats.values())
        total_output_tokens = sum(stats["output_tokens"] for stats in agent_stats.values())

        return {
            "by_agent": agent_stats,
            "summary": {
                "total_agents_used": total_agents,
                "total_execution_time_ms": total_time_all_agents,
                "average_time_per_agent_ms": (
                    total_time_all_agents / total_agents if total_agents > 0 else 0
                ),
                "total_tokens_used": total_tokens_all_agents,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "average_tokens_per_agent": (
                    total_tokens_all_agents / total_agents if total_agents > 0 else 0
                ),
            },
        }

    def get_agent_efficiency_report(self) -> Dict[str, Any]:
        """Generate an efficiency report for all agents."""
        stats = self.calculate_agent_statistics()
        if not stats or "by_agent" not in stats:
            return {"message": "No data available for efficiency report"}

        efficiency_report = {
            "most_efficient_agent": None,
            "least_efficient_agent": None,
            "token_efficiency_ranking": [],
            "time_efficiency_ranking": [],
            "recommendations": [],
        }

        # Calculate efficiency metrics
        agent_efficiency = {}
        for agent_name, agent_data in stats["by_agent"].items():
            if agent_data["total_tokens"] > 0 and agent_data["total_time_ms"] > 0:
                # Efficiency = tokens per millisecond (higher is more efficient)
                token_efficiency = agent_data["total_tokens"] / agent_data["total_time_ms"]
                # Time per step (lower is better)
                time_per_step = agent_data["total_time_ms"] / agent_data["total_steps"]

                agent_efficiency[agent_name] = {
                    "token_efficiency": token_efficiency,
                    "time_per_step": time_per_step,
                    "total_tokens": agent_data["total_tokens"],
                    "total_time_ms": agent_data["total_time_ms"],
                    "total_steps": agent_data["total_steps"],
                }

        if agent_efficiency:
            # Token efficiency ranking (higher is better)
            token_ranking = sorted(
                agent_efficiency.items(),
                key=lambda x: x[1]["token_efficiency"],
                reverse=True,
            )
            efficiency_report["token_efficiency_ranking"] = [
                {
                    "agent": agent,
                    "tokens_per_ms": round(data["token_efficiency"], 4),
                    "total_tokens": data["total_tokens"],
                }
                for agent, data in token_ranking
            ]

            # Time efficiency ranking (lower time per step is better)
            time_ranking = sorted(agent_efficiency.items(), key=lambda x: x[1]["time_per_step"])
            efficiency_report["time_efficiency_ranking"] = [
                {
                    "agent": agent,
                    "time_per_step_ms": round(data["time_per_step"], 2),
                    "total_steps": data["total_steps"],
                }
                for agent, data in time_ranking
            ]

            # Identify most and least efficient
            if token_ranking:
                efficiency_report["most_efficient_agent"] = {
                    "name": token_ranking[0][0],
                    "tokens_per_ms": round(token_ranking[0][1]["token_efficiency"], 4),
                }
                efficiency_report["least_efficient_agent"] = {
                    "name": token_ranking[-1][0],
                    "tokens_per_ms": round(token_ranking[-1][1]["token_efficiency"], 4),
                }

            # Generate recommendations
            recommendations = []

            # Check for high token usage
            avg_tokens = stats["summary"]["average_tokens_per_agent"]
            for agent_name, agent_data in stats["by_agent"].items():
                if agent_data["total_tokens"] > avg_tokens * 1.5:
                    recommendations.append(
                        f"Agent '{agent_name}' uses {agent_data['total_tokens']} tokens "
                        f"({round(agent_data['total_tokens']/avg_tokens, 1)}x average). "
                        "Consider optimizing prompts or reducing context length."
                    )

            # Check for slow execution
            avg_time = stats["summary"]["average_time_per_agent_ms"]
            for agent_name, agent_data in stats["by_agent"].items():
                if agent_data["total_time_ms"] > avg_time * 1.5:
                    recommendations.append(
                        f"Agent '{agent_name}' takes {agent_data['total_time_ms']}ms "
                        f"({round(agent_data['total_time_ms']/avg_time, 1)}x average). "
                        "Consider reducing task complexity or optimizing workflows."
                    )

            efficiency_report["recommendations"] = recommendations

        return efficiency_report

    def get_cost_estimate(
        self,
        cost_per_1k_input_tokens: float = 0.0006,
        cost_per_1k_output_tokens: float = 0.0024,
    ) -> Dict[str, Any]:
        """
        Estimate costs based on token usage.

        Args:
            cost_per_1k_input_tokens: Cost per 1000 input tokens (default: GPT-4o-mini pricing)
            cost_per_1k_output_tokens: Cost per 1000 output tokens (default: GPT-4o-mini pricing)
        """
        stats = self.calculate_agent_statistics()
        if not stats or "summary" not in stats:
            return {"message": "No data available for cost estimation"}

        summary = stats["summary"]

        # Calculate costs
        input_cost = (summary["total_input_tokens"] / 1000) * cost_per_1k_input_tokens
        output_cost = (summary["total_output_tokens"] / 1000) * cost_per_1k_output_tokens
        total_cost = input_cost + output_cost

        cost_breakdown = {
            "total_cost_usd": round(total_cost, 6),
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_input_tokens": summary["total_input_tokens"],
            "total_output_tokens": summary["total_output_tokens"],
            "total_tokens": summary["total_tokens_used"],
            "cost_per_1k_input": cost_per_1k_input_tokens,
            "cost_per_1k_output": cost_per_1k_output_tokens,
        }

        # Per-agent cost breakdown
        if "by_agent" in stats:
            agent_costs = {}
            for agent_name, agent_data in stats["by_agent"].items():
                agent_input_cost = (agent_data["input_tokens"] / 1000) * cost_per_1k_input_tokens
                agent_output_cost = (agent_data["output_tokens"] / 1000) * cost_per_1k_output_tokens
                agent_total_cost = agent_input_cost + agent_output_cost

                agent_costs[agent_name] = {
                    "total_cost_usd": round(agent_total_cost, 6),
                    "input_cost_usd": round(agent_input_cost, 6),
                    "output_cost_usd": round(agent_output_cost, 6),
                    "input_tokens": agent_data["input_tokens"],
                    "output_tokens": agent_data["output_tokens"],
                    "total_tokens": agent_data["total_tokens"],
                }

            cost_breakdown["by_agent"] = agent_costs

        return cost_breakdown

    def export_statistics_report(self) -> Dict[str, Any]:
        """Export comprehensive statistics report."""
        return {
            "agent_statistics": self.calculate_agent_statistics(),
            "efficiency_report": self.get_agent_efficiency_report(),
            "cost_estimate": self.get_cost_estimate(),
        }

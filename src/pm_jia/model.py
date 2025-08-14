"""
Pydantic models for data validation and serialization.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# General Models


class MessageRole(str, Enum):
    """Message role enumeration."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class TokenUsage:
    """Token usage information from LLM calls."""

    input_tokens: int
    output_tokens: int
    total_tokens: int

    def dict(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class Priority(Enum):
    """Priority levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Document(BaseModel):
    """Project document."""

    name: str
    content: str
    version: str = Field(default="1.0.0")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# Workflow Planning Models


class WorkflowStepType(str, Enum):
    """Types of workflow steps."""

    SAFETY = "safety"
    DESIGN = "design"
    PRODUCT = "product"
    TECHNICAL = "technical"
    MARKET = "market"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"

    def get_json_schema(self):
        return {
            "safety": SafetyCheckResult,
            "market": MarketAnalysis,
            "design": DesignSpecification,
            "product": ProductRequirement,
            "technical": TechnicalSpecification,
            "synthesis": SynthesisResult,
            "validation": ValidationResult,
        }.get(self)


class WorkflowStep(BaseModel):
    """Individual step in a workflow plan."""

    step_name: str = Field(..., description="Unique identifier for this step")
    step_type: WorkflowStepType = Field(..., description="Type of analysis or work to be done")
    step_description: str = Field(..., description="What this step accomplishes")
    role_description: str = Field(..., description="What role this step is in the workflow")
    expertise: List[str] = Field(
        default_factory=list, description="Expertise required for this step"
    )
    depends_on: Optional[List[str]] = Field(
        default=None, description="Steps that must be completed before this one"
    )


class WorkflowPlan(BaseModel):
    """Complete workflow plan for product design."""

    workflow: List[WorkflowStep] = Field(..., description="Ordered list of workflow steps")
    reasoning: str = Field(
        ..., description="Explanation of why this workflow is optimal for the request"
    )


# Agent Response Models


class SafetyCheckResult(BaseModel):
    """Result of safety validation."""

    is_safe: bool
    message: str
    confidence_score: float
    risk_flags: List[str] = Field(default_factory=list)


class MarketAnalysis(BaseModel):
    """Market analysis for a product."""

    target_market: str = Field(..., description="Description of target market")
    market_size: Optional[str] = Field(default=None, description="Estimated market size")
    competitors: List[str] = Field(default_factory=list, description="List of main competitors")
    competitive_advantages: List[str] = Field(default_factory=list)
    market_trends: List[str] = Field(default_factory=list)
    pricing_strategy: Optional[str] = Field(default=None)


class DesignSpecification(BaseModel):
    """Design specification for a product."""

    user_personas: List[str] = Field(default_factory=list, description="Target user personas")
    user_journeys: List[str] = Field(default_factory=list, description="Key user journeys")
    design_principles: List[str] = Field(default_factory=list)
    ui_components: List[str] = Field(default_factory=list, description="Key UI components needed")
    accessibility_requirements: List[str] = Field(default_factory=list)
    design_system: Optional[str] = Field(default=None, description="Design system to be used")


class ProductRequirement(BaseModel):
    """Individual product requirement."""

    title: str
    description: str
    priority: Priority = Priority.MEDIUM
    category: str = Field(
        ..., description="Category of requirement (functional, non-functional, etc.)"
    )
    acceptance_criteria: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)


class TechnicalSpecification(BaseModel):
    """Technical specification for a product."""

    architecture_pattern: str = Field(
        ..., description="Overall architecture pattern (MVC, microservices, etc.)"
    )
    frontend: List[str] = Field(default_factory=list)
    backend: List[str] = Field(default_factory=list)
    database: List[str] = Field(default_factory=list)
    infrastructure: List[str] = Field(default_factory=list)
    scalability_requirements: List[str] = Field(default_factory=list)
    security_requirements: List[str] = Field(default_factory=list)
    performance_requirements: List[str] = Field(default_factory=list)
    deployment_strategy: str = Field(..., description="How the product will be deployed")


class SynthesisResult(BaseModel):
    """Synthesis result combining all analysis."""

    synthesis_summary: str = Field(
        ..., description="Executive summary synthesizing all analysis components"
    )
    key_insights: List[str] = Field(
        default_factory=list, description="Key insights from combined analysis"
    )
    strategic_recommendations: List[str] = Field(
        default_factory=list, description="Strategic recommendations"
    )
    implementation_priorities: List[str] = Field(
        default_factory=list, description="Prioritized implementation steps"
    )
    risk_mitigation_strategies: List[str] = Field(
        default_factory=list, description="Risk mitigation approaches"
    )
    success_metrics: List[str] = Field(
        default_factory=list, description="Metrics to measure success"
    )
    resource_requirements: List[str] = Field(
        default_factory=list, description="Required resources by category"
    )
    timeline_estimate: str = Field(..., description="Estimated timeline for implementation")
    confidence_assessment: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence in synthesis"
    )
    document: Optional[str] = Field(
        default=None, description="The synthesized product designdocument"
    )


class ValidationResult(BaseModel):
    """Result of document validation."""

    score: int = Field(..., ge=1, le=10, description="Quality score from 1-10")
    approved: bool = Field(..., description="Whether document meets approval threshold")
    recommendation: str = Field(..., description="APPROVE or REVISE")
    feedback: str = Field(..., description="Detailed feedback and suggestions")
    document: str = Field(default=None, description="The validated document")


# API Models


class SessionCreateRequest(BaseModel):
    project_name: Optional[str] = "Default Project"
    config: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    session_id: str
    project_name: str
    created_at: str
    status: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    additional_materials: Optional[Dict[str, str]] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    timestamp: str


class DocumentGenerationRequest(BaseModel):
    session_id: str
    user_input: str
    additional_materials: Optional[Dict[str, str]] = None


class DocumentGenerationResponse(BaseModel):
    session_id: str
    success: bool
    document: Optional[str] = None
    stage: str
    message: Optional[str] = None
    workflow_plan: Optional[Dict[str, Any]] = None


class SessionSaveRequest(BaseModel):
    session_id: str
    data: Dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

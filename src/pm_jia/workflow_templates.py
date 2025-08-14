"""
Default workflow templates for different types of product design requests.

If the LLM planning fails, the default workflow will be used.
"""

from typing import Any, Dict


def get_default_workflow() -> Dict[str, Any]:
    """Fallback default workflow if LLM planning fails."""
    workflow_steps = []

    # Add analysis steps that can run in parallel
    analysis_steps = [
        {
            "step": "product_analysis",
            "type": "product",
            "step_description": "Define product requirements and user stories",
            "role_description": "Product Analyst",
            "expertise": [
                "Product requirements and user stories",
                "Product specifications and creation",
                "Product attraction and marketing",
                "Product pricing and revenue models",
            ],
            "description": "Define product requirements and user stories",
            "depends_on": None,
        },
        {
            "step": "design_analysis",
            "type": "design",
            "step_description": "Analyze UX/UI requirements and user workflows",
            "role_description": "Product Designer",
            "expertise": [
                "User experience (UX) design and user journey mapping",
                "Product workflow and interaction design",
                "Interface design principles and best practices",
                "User research and persona development",
                "Accessibility and inclusive design",
            ],
            "description": "Analyze UX/UI requirements and user workflows",
            "depends_on": None,
        },
        {
            "step": "tech_analysis",
            "type": "technical",
            "step_description": "Define technical architecture and technology stack",
            "role_description": "Technical Architect",
            "expertise": [
                "Software architecture design and patterns",
                "Technology stack selection and evaluation",
                "Scalability and performance considerations",
                "Security architecture and best practices",
                "DevOps and deployment strategies",
            ],
            "description": "Define technical architecture and technology stack",
            "depends_on": None,
        },
        {
            "step": "market_analysis",
            "type": "market",
            "step_description": "Research market trends and competitive landscape",
            "role_description": "Market Analyst",
            "expertise": [
                "Market trend analysis and insights",
                "Competitive landscape research",
                "User preference and behavior analysis",
                "Technology adoption patterns",
                "Industry best practices research",
            ],
            "description": "Research market trends and competitive landscape",
            "depends_on": None,
        },
    ]

    workflow_steps.extend(analysis_steps)

    # Add synthesis and validation steps
    workflow_steps.extend(
        [
            {
                "step": "synthesis",
                "type": "synthesis",
                "step_description": "Combine all analyses into comprehensive document",
                "role_description": "Product Manager",
                "expertise": [
                    "Integration of multiple analyses into coherent documents",
                    "Structured documentation and presentation",
                    "Ensuring consistency across different inputs",
                    "Creating comprehensive product specifications",
                ],
                "depends_on": [
                    "product_analysis",
                    "design_analysis",
                    "tech_analysis",
                    "market_analysis",
                ],
            },
            {
                "step": "validation",
                "type": "validation",
                "step_description": "Validate document quality and completeness",
                "role_description": "Product Validator",
                "expertise": [
                    "Document quality assessment and validation",
                    "Consistency checking across sections",
                    "Identification of missing critical information",
                    "Professional presentation standards",
                    "Providing specific improvement feedback",
                    "Score documents 1-10",
                ],
                "depends_on": ["synthesis"],
            },
        ]
    )

    return {
        "success": True,
        "workflow": workflow_steps,
        "reasoning": "Default workflow covering all essential aspects of product design",
        "estimated_steps": len(workflow_steps),
    }

"""
Workflow Router

This module provides routing logic for directing workflow execution based on
LLM agent decisions. It uses runnable's Conditional node to route to appropriate
workflow templates and handles parameter passing between components.
"""

from typing import Dict, List, Any, Callable, Optional
from runnable import Pipeline, PythonTask, Conditional, pickled


def create_workflow_router(workflow_branches: Dict[str, Callable]) -> Conditional:
    """
    Create a conditional router that directs execution to appropriate workflow branches.

    Args:
        workflow_branches: Dictionary mapping workflow names to pipeline creation functions

    Returns:
        Configured Conditional node for workflow routing
    """

    def get_workflow_name(workflow_decision: Dict[str, Any]) -> str:
        """Extract workflow name from LLM decision."""
        return workflow_decision.get("selected_workflow", "research_workflow")

    return Conditional(
        name="workflow_router",
        condition=get_workflow_name,
        condition_input=pickled("workflow_decision"),
        branches=workflow_branches
    )


def validate_workflow_parameters(
    workflow_decision: Dict[str, Any],
    required_params: List[str]
) -> Dict[str, Any]:
    """
    Validate and normalize workflow parameters from LLM decision.

    Args:
        workflow_decision: LLM agent's workflow selection and parameters
        required_params: List of required parameter names

    Returns:
        Validated and normalized parameters
    """

    workflow_params = workflow_decision.get("workflow_parameters", {})
    validated_params = {}

    # Ensure all required parameters are present
    for param in required_params:
        if param in workflow_params:
            validated_params[param] = workflow_params[param]
        else:
            # Provide default values for missing parameters
            validated_params[param] = get_default_parameter_value(param, workflow_decision)

    # Add meta-information
    validated_params.update({
        "original_query": workflow_decision.get("original_query", ""),
        "selected_workflow": workflow_decision.get("selected_workflow", ""),
        "llm_reasoning": workflow_decision.get("reasoning", ""),
        "confidence": workflow_decision.get("confidence", 0.5)
    })

    return validated_params


def get_default_parameter_value(param_name: str, context: Dict[str, Any]) -> Any:
    """
    Provide default values for missing workflow parameters.

    Args:
        param_name: Name of the parameter
        context: Context from LLM decision for intelligent defaults

    Returns:
        Default parameter value
    """

    defaults = {
        # Research workflow defaults
        "query": context.get("original_query", ""),
        "search_depth": "medium",
        "source_types": ["web"],
        "max_results": 5,
        "summarize": True,

        # Analysis workflow defaults
        "analysis_type": "descriptive",
        "comparison_mode": False,
        "metrics_focus": ["general"],
        "output_format": "structured_report",

        # Content creation workflow defaults
        "content_type": "general",
        "tone": "neutral",
        "length": "medium",
        "include_research": True,

        # Problem solving workflow defaults
        "problem_domain": "general",
        "solution_approach": "systematic",
        "include_alternatives": True,
        "priority": "medium"
    }

    return defaults.get(param_name, "default")


def create_parameter_validator_task(required_params: List[str]) -> PythonTask:
    """
    Create a task that validates workflow parameters.

    Args:
        required_params: List of required parameter names

    Returns:
        PythonTask for parameter validation
    """

    def validate_params(workflow_decision: Dict[str, Any]) -> Dict[str, Any]:
        return validate_workflow_parameters(workflow_decision, required_params)

    return PythonTask(
        function=validate_params,
        name="parameter_validator",
        kwargs={"workflow_decision": pickled("workflow_decision")},
        returns=[pickled("validated_parameters")]
    )


def extract_workflow_context(workflow_decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and prepare context information for workflow execution.

    Args:
        workflow_decision: LLM agent's decision data

    Returns:
        Context dictionary for workflow execution
    """

    return {
        "execution_context": {
            "workflow_name": workflow_decision.get("selected_workflow"),
            "llm_confidence": workflow_decision.get("confidence", 0.5),
            "reasoning": workflow_decision.get("reasoning", ""),
            "timestamp": workflow_decision.get("analysis_timestamp"),
            "original_query": workflow_decision.get("original_query")
        },
        "workflow_parameters": workflow_decision.get("workflow_parameters", {}),
        "user_context": workflow_decision.get("workflow_parameters", {}).get("user_context", {})
    }


def create_context_extractor_task() -> PythonTask:
    """Create a task that extracts workflow context."""

    return PythonTask(
        function=extract_workflow_context,
        name="context_extractor",
        kwargs={"workflow_decision": pickled("workflow_decision")},
        returns=[pickled("workflow_context")]
    )


def log_workflow_routing(workflow_decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log workflow routing decision for debugging and monitoring.

    Args:
        workflow_decision: LLM agent's decision data

    Returns:
        Logging information
    """

    routing_log = {
        "routing_timestamp": workflow_decision.get("analysis_timestamp"),
        "selected_workflow": workflow_decision.get("selected_workflow"),
        "confidence_score": workflow_decision.get("confidence"),
        "original_query": workflow_decision.get("original_query"),
        "reasoning": workflow_decision.get("reasoning"),
        "parameters_count": len(workflow_decision.get("workflow_parameters", {})),
        "routing_status": "success"
    }

    print(f"🔀 Workflow Router: Directing to '{routing_log['selected_workflow']}' "
          f"(confidence: {routing_log['confidence_score']:.1%})")

    return routing_log


def create_routing_logger_task() -> PythonTask:
    """Create a task that logs routing decisions."""

    return PythonTask(
        function=log_workflow_routing,
        name="routing_logger",
        kwargs={"workflow_decision": pickled("workflow_decision")},
        returns=[pickled("routing_log")]
    )


# Workflow Branch Creation Helpers

def create_research_workflow_branch():
    """
    Create a research workflow branch.
    This would be implemented in research_workflow.py
    """
    from .research_workflow import create_research_pipeline
    return create_research_pipeline()


def create_analysis_workflow_branch():
    """
    Create an analysis workflow branch.
    This would be implemented in analysis_workflow.py
    """
    from .analysis_workflow import create_analysis_pipeline
    return create_analysis_pipeline()


def create_content_workflow_branch():
    """
    Create a content creation workflow branch.
    This would be implemented in content_workflow.py
    """
    from .content_workflow import create_content_pipeline
    return create_content_pipeline()


def create_problem_solving_workflow_branch():
    """
    Create a problem solving workflow branch.
    This could be implemented as additional workflow.
    """
    # For now, fallback to research workflow
    from .research_workflow import create_research_pipeline
    return create_research_pipeline()


def get_standard_workflow_branches() -> Dict[str, Callable]:
    """
    Get the standard set of workflow branches.

    Returns:
        Dictionary mapping workflow names to branch creation functions
    """

    return {
        "research_workflow": create_research_workflow_branch,
        "analysis_workflow": create_analysis_workflow_branch,
        "content_creation_workflow": create_content_workflow_branch,
        "problem_solving_workflow": create_problem_solving_workflow_branch
    }


def create_fallback_workflow() -> Pipeline:
    """
    Create a fallback workflow for unknown or failed routing.

    Returns:
        Simple fallback pipeline
    """

    def fallback_handler(workflow_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fallback case when routing fails."""

        return {
            "status": "fallback_executed",
            "message": f"Executed fallback workflow for query: {workflow_decision.get('original_query', 'unknown')}",
            "fallback_reason": "Unknown or failed workflow routing",
            "original_selection": workflow_decision.get("selected_workflow", "unknown"),
            "timestamp": workflow_decision.get("analysis_timestamp")
        }

    return Pipeline([
        PythonTask(
            function=fallback_handler,
            name="fallback_handler",
            kwargs={"workflow_decision": pickled("workflow_decision")},
            returns=[pickled("fallback_result")]
        )
    ])


# Error Handling

def handle_routing_error(error: Exception, workflow_decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle errors that occur during workflow routing.

    Args:
        error: The exception that occurred
        workflow_decision: Original LLM decision data

    Returns:
        Error handling result
    """

    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "failed_workflow": workflow_decision.get("selected_workflow", "unknown"),
        "original_query": workflow_decision.get("original_query", ""),
        "timestamp": workflow_decision.get("analysis_timestamp"),
        "fallback_applied": True
    }

    print(f"❌ Workflow routing error: {error_info['error_message']}")
    print(f"🔄 Applying fallback workflow for: {error_info['original_query']}")

    return error_info


# Main Router Factory

def create_main_workflow_router(
    workflow_branches: Optional[Dict[str, Callable]] = None,
    include_validation: bool = True,
    include_logging: bool = True
) -> Pipeline:
    """
    Create the main workflow routing pipeline.

    Args:
        workflow_branches: Custom workflow branches (uses standard if None)
        include_validation: Whether to include parameter validation
        include_logging: Whether to include routing logging

    Returns:
        Complete workflow routing pipeline
    """

    if workflow_branches is None:
        workflow_branches = get_standard_workflow_branches()

    # Add fallback workflow
    workflow_branches["fallback"] = create_fallback_workflow

    steps = []

    # Optional: Add routing logger
    if include_logging:
        steps.append(create_routing_logger_task())

    # Optional: Add parameter validation
    if include_validation:
        steps.append(create_parameter_validator_task(["query"]))

    # Add context extractor
    steps.append(create_context_extractor_task())

    # Main workflow router
    router = create_workflow_router(workflow_branches)
    steps.append(router)

    return Pipeline(steps)


# Example usage
if __name__ == "__main__":

    # Example workflow decision from LLM
    example_decision = {
        "selected_workflow": "research_workflow",
        "workflow_parameters": {
            "query": "machine learning tutorials",
            "search_depth": "deep",
            "max_results": 8
        },
        "reasoning": "User needs comprehensive research information",
        "confidence": 0.85,
        "analysis_timestamp": "2024-01-15T10:30:00",
        "original_query": "Find me detailed information about machine learning"
    }

    print("🔀 Workflow Router - Example Routing")
    print("=" * 50)

    # Test parameter validation
    print("\n1. Testing Parameter Validation:")
    validated = validate_workflow_parameters(example_decision, ["query", "search_depth", "max_results"])
    print(f"   Validated parameters: {list(validated.keys())}")
    print(f"   Original query: {validated['original_query']}")

    # Test context extraction
    print("\n2. Testing Context Extraction:")
    context = extract_workflow_context(example_decision)
    print(f"   Workflow: {context['execution_context']['workflow_name']}")
    print(f"   Confidence: {context['execution_context']['llm_confidence']:.1%}")

    # Test routing log
    print("\n3. Testing Routing Logger:")
    log_result = log_workflow_routing(example_decision)
    print(f"   Routing status: {log_result['routing_status']}")

    # Test available workflows
    print("\n4. Available Workflow Branches:")
    branches = get_standard_workflow_branches()
    for workflow_name in branches.keys():
        print(f"   - {workflow_name}")

    print("\n✅ Workflow router components ready for integration")

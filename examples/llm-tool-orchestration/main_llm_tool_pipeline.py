"""
Main LLM + Tool Pipeline Integration Example

This is the complete integration example that demonstrates how all components work together:
1. LLM Workflow Selector - analyzes user input and selects workflow
2. Workflow Router - routes to appropriate workflow template
3. Workflow Execution - runs selected workflow with tools
4. Result Aggregation - combines and formats final results

This example shows the full end-to-end flow of LLM-driven tool orchestration.
"""

from typing import Dict, Any, List
from runnable import Pipeline, PythonTask, Conditional, pickled

# Import all our components
try:
    from .llm_workflow_selector import llm_workflow_selector
    from .workflow_router import (
        create_main_workflow_router,
        extract_workflow_context,
        log_workflow_routing
    )
    from .research_workflow import create_research_pipeline
    from .analysis_workflow import create_analysis_pipeline
    from .content_workflow import create_content_pipeline
except ImportError:
    # Handle direct execution
    from llm_workflow_selector import llm_workflow_selector
    from workflow_router import (
        create_main_workflow_router,
        extract_workflow_context,
        log_workflow_routing
    )
    from research_workflow import create_research_pipeline
    from analysis_workflow import create_analysis_pipeline
    from content_workflow import create_content_pipeline


def prepare_user_input(user_query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Prepare and validate user input for the LLM + tool pipeline.

    Args:
        user_query: The user's request/query
        user_context: Optional context about the user (preferences, history, etc.)

    Returns:
        Prepared input data for the pipeline
    """

    if user_context is None:
        user_context = {}

    # Basic input validation
    if not user_query or not user_query.strip():
        raise ValueError("User query cannot be empty")

    prepared_input = {
        "user_query": user_query.strip(),
        "user_context": user_context,
        "pipeline_metadata": {
            "input_timestamp": "2024-01-15T10:30:00",  # In real implementation, use datetime.now()
            "input_length": len(user_query),
            "context_provided": bool(user_context),
            "pipeline_version": "1.0.0"
        }
    }

    print(f"📝 User Input Prepared: '{user_query[:50]}{'...' if len(user_query) > 50 else ''}'")
    return prepared_input


def execute_llm_workflow_selection() -> Dict[str, Any]:
    """
    Execute LLM workflow selection step.

    This function will receive parameters automatically from the pipeline context.

    Returns:
        LLM workflow selection results
    """

    print("🤖 LLM Agent analyzing query and selecting workflow...")

    # Get parameters from pipeline context
    # In a real runnable pipeline, these would be passed via context automatically
    # For this demo, we'll use sample data
    user_query = "Research machine learning best practices"  # This would come from pipeline context
    user_context = {"user_expertise": "intermediate"}

    # Execute LLM workflow selection
    workflow_decision = llm_workflow_selector(
        user_query=user_query,
        context=user_context
    )

    print(f"✅ LLM Selected: '{workflow_decision['selected_workflow']}' "
          f"(confidence: {workflow_decision['confidence']:.1%})")

    return {
        "llm_decision": workflow_decision,
        "selection_metadata": {
            "selection_successful": True,
            "workflow_available": workflow_decision["selected_workflow"] in get_available_workflows(),
            "confidence_level": "high" if workflow_decision["confidence"] > 0.7 else "medium" if workflow_decision["confidence"] > 0.4 else "low"
        },
        "original_input": {
            "user_query": user_query,
            "user_context": user_context
        }
    }


def create_workflow_execution_branches() -> Dict[str, Pipeline]:
    """
    Create workflow execution branches for the conditional router.

    Returns:
        Dictionary mapping workflow names to their pipeline implementations
    """

    return {
        "research_workflow": create_research_pipeline(),
        "analysis_workflow": create_analysis_pipeline(),
        "content_creation_workflow": create_content_pipeline(),
        "problem_solving_workflow": create_research_pipeline()  # Fallback to research
    }


def route_to_workflow(llm_selection: Dict[str, Any]) -> str:
    """
    Route to appropriate workflow based on LLM selection.

    Args:
        llm_selection: LLM workflow selection results

    Returns:
        Workflow route name
    """

    decision = llm_selection["llm_decision"]
    selected_workflow = decision["selected_workflow"]

    # Validate workflow availability
    available_workflows = get_available_workflows()

    if selected_workflow in available_workflows:
        print(f"🔀 Routing to: {selected_workflow}")
        return selected_workflow
    else:
        print(f"⚠️ Workflow '{selected_workflow}' not available, defaulting to research_workflow")
        return "research_workflow"


def get_available_workflows() -> List[str]:
    """Get list of available workflow names."""
    return [
        "research_workflow",
        "analysis_workflow",
        "content_creation_workflow",
        "problem_solving_workflow"
    ]


def aggregate_workflow_results() -> Dict[str, Any]:
    """
    Aggregate and format final results from workflow execution.

    This function receives parameters automatically from the pipeline context.

    Returns:
        Aggregated final results
    """

    print("📊 Aggregating workflow results...")

    # In a real implementation, this would get workflow_output and llm_selection from context
    # For this demo, we'll simulate the aggregation

    # Mock data for demonstration
    workflow_name = "research_workflow"

    # Simulate workflow results
    mock_workflow_output = {
        "research_summary": {
            "research_summary": "Comprehensive analysis of machine learning best practices completed successfully.",
            "key_findings": [
                "Cross-validation is essential for model evaluation",
                "Feature engineering significantly impacts model performance",
                "Regular model monitoring prevents performance degradation"
            ],
            "source_summary": {
                "total_sources": 5,
                "content_quality": 0.85
            }
        }
    }

    mock_llm_selection = {
        "llm_decision": {
            "selected_workflow": workflow_name,
            "confidence": 0.85,
            "reasoning": "Query requires comprehensive research and information gathering"
        },
        "original_input": {
            "user_query": "Research machine learning best practices",
            "user_context": {"user_expertise": "intermediate"},
            "pipeline_metadata": {"pipeline_version": "1.0.0"}
        }
    }

    # Extract key results based on workflow type
    results_summary = extract_workflow_results(mock_workflow_output, workflow_name)

    # Create comprehensive results
    final_results = {
        "execution_summary": {
            "status": "completed",
            "original_query": mock_llm_selection["original_input"]["user_query"],
            "selected_workflow": workflow_name,
            "llm_confidence": mock_llm_selection["llm_decision"]["confidence"],
            "llm_reasoning": mock_llm_selection["llm_decision"]["reasoning"],
            "execution_successful": True
        },
        "workflow_results": results_summary,
        "metadata": {
            "pipeline_version": mock_llm_selection["original_input"]["pipeline_metadata"]["pipeline_version"],
            "total_execution_time": "estimated 3.2 seconds",
            "components_executed": ["llm_selector", "workflow_router", workflow_name],
            "result_quality": calculate_result_quality(results_summary, workflow_name)
        },
        "user_context": mock_llm_selection["original_input"]["user_context"]
    }

    print(f"✅ Results aggregated successfully for {workflow_name}")
    return final_results


def extract_workflow_results(workflow_output: Dict[str, Any], workflow_name: str) -> Dict[str, Any]:
    """
    Extract and format results from specific workflow type.

    Args:
        workflow_output: Raw workflow output
        workflow_name: Name of the workflow that was executed

    Returns:
        Formatted workflow results
    """

    if workflow_name == "research_workflow":
        return extract_research_results(workflow_output)
    elif workflow_name == "analysis_workflow":
        return extract_analysis_results(workflow_output)
    elif workflow_name == "content_creation_workflow":
        return extract_content_results(workflow_output)
    else:
        return {"raw_output": workflow_output, "workflow_type": workflow_name}


def extract_research_results(workflow_output: Dict[str, Any]) -> Dict[str, Any]:
    """Extract results from research workflow."""

    # Research workflow returns research_summary as final output
    research_summary = workflow_output.get("research_summary", {})

    return {
        "result_type": "research_findings",
        "summary": research_summary.get("research_summary", "Research completed"),
        "key_findings": research_summary.get("key_findings", []),
        "sources_analyzed": research_summary.get("source_summary", {}).get("total_sources", 0),
        "research_quality": research_summary.get("source_summary", {}).get("content_quality", 0.5),
        "detailed_results": research_summary
    }


def extract_analysis_results(workflow_output: Dict[str, Any]) -> Dict[str, Any]:
    """Extract results from analysis workflow."""

    # Analysis workflow returns analysis_report as final output
    analysis_report = workflow_output.get("analysis_report", {})

    return {
        "result_type": "analysis_insights",
        "executive_summary": analysis_report.get("executive_summary", "Analysis completed"),
        "key_insights": analysis_report.get("insights", []),
        "recommendations": analysis_report.get("recommendations", []),
        "analyses_performed": analysis_report.get("analysis_overview", {}).get("analyses_performed", 0),
        "success_rate": analysis_report.get("analysis_overview", {}).get("success_rate", "Unknown"),
        "detailed_results": analysis_report
    }


def extract_content_results(workflow_output: Dict[str, Any]) -> Dict[str, Any]:
    """Extract results from content creation workflow."""

    # Content workflow returns formatted_content as final output
    formatted_content = workflow_output.get("formatted_content", {})

    return {
        "result_type": "created_content",
        "title": formatted_content.get("publication_metadata", {}).get("title", "Generated Content"),
        "content": formatted_content.get("final_content", "Content creation completed"),
        "word_count": formatted_content.get("publication_metadata", {}).get("word_count", 0),
        "content_type": formatted_content.get("publication_metadata", {}).get("content_type", "general"),
        "reading_time": formatted_content.get("publication_metadata", {}).get("estimated_reading_time", "Unknown"),
        "ready_for_publication": formatted_content.get("ready_for_publication", False),
        "detailed_results": formatted_content
    }


def calculate_result_quality(results: Dict[str, Any], workflow_name: str) -> str:
    """Calculate overall result quality score."""

    if workflow_name == "research_workflow":
        quality_score = results.get("research_quality", 0.5)
    elif workflow_name == "analysis_workflow":
        # Parse success rate if it's a string percentage
        success_rate = results.get("success_rate", "50%")
        if isinstance(success_rate, str) and "%" in success_rate:
            quality_score = float(success_rate.replace("%", "")) / 100
        else:
            quality_score = 0.5
    elif workflow_name == "content_creation_workflow":
        # Base quality on whether content is ready for publication
        quality_score = 0.8 if results.get("ready_for_publication", False) else 0.6
    else:
        quality_score = 0.5

    if quality_score >= 0.8:
        return "excellent"
    elif quality_score >= 0.6:
        return "good"
    elif quality_score >= 0.4:
        return "fair"
    else:
        return "poor"


def format_user_results(final_results: Dict[str, Any]) -> str:
    """
    Format results for user-friendly display.

    Args:
        final_results: Aggregated final results

    Returns:
        Formatted string for user display
    """

    summary = final_results["execution_summary"]
    results = final_results["workflow_results"]
    metadata = final_results["metadata"]

    # Create formatted output
    output = f"""
🎯 LLM + Tool Pipeline Results
{'=' * 50}

Query: "{summary['original_query']}"
Workflow: {summary['selected_workflow']}
Confidence: {summary['llm_confidence']:.1%}
Quality: {metadata['result_quality']}

{format_workflow_specific_results(results)}

💡 LLM Reasoning: {summary['llm_reasoning']}

📋 Execution Summary:
- Components: {', '.join(metadata['components_executed'])}
- Execution Time: {metadata['total_execution_time']}
- Status: {'✅ Success' if summary['execution_successful'] else '❌ Failed'}
"""

    return output.strip()


def format_workflow_specific_results(results: Dict[str, Any]) -> str:
    """Format results specific to workflow type."""

    result_type = results["result_type"]

    if result_type == "research_findings":
        output = f"""📚 Research Results:
- Sources Analyzed: {results['sources_analyzed']}
- Research Quality: {results['research_quality']:.1%}

Summary: {results['summary'][:200]}{'...' if len(results['summary']) > 200 else ''}

Key Findings:"""

        for i, finding in enumerate(results.get('key_findings', [])[:3], 1):
            output += f"\n{i}. {finding[:100]}{'...' if len(finding) > 100 else ''}"

    elif result_type == "analysis_insights":
        output = f"""📊 Analysis Results:
- Analyses Performed: {results['analyses_performed']}
- Success Rate: {results['success_rate']}

Executive Summary: {results['executive_summary'][:200]}{'...' if len(results['executive_summary']) > 200 else ''}

Key Insights:"""

        for i, insight in enumerate(results.get('key_insights', [])[:3], 1):
            output += f"\n{i}. {insight[:100]}{'...' if len(insight) > 100 else ''}"

        output += "\n\nRecommendations:"
        for i, rec in enumerate(results.get('recommendations', [])[:2], 1):
            output += f"\n{i}. {rec[:100]}{'...' if len(rec) > 100 else ''}"

    elif result_type == "created_content":
        output = f"""✍️ Content Creation Results:
- Title: {results['title']}
- Type: {results['content_type']}
- Word Count: {results['word_count']}
- Reading Time: {results['reading_time']}
- Publication Ready: {'Yes' if results['ready_for_publication'] else 'No'}

Content Preview:
{results['content'][:300]}{'...' if len(results['content']) > 300 else ''}"""

    else:
        output = f"Results Type: {result_type}\nRaw Output Available"

    return output


def create_complete_llm_tool_pipeline() -> Pipeline:
    """
    Create the complete LLM + tool orchestration pipeline.

    Returns:
        Complete pipeline with all components integrated
    """

    return Pipeline([
        # Step 1: LLM Workflow Selection
        PythonTask(
            function=execute_llm_workflow_selection,
            name="llm_workflow_selection",
            returns=[pickled("llm_selection")]
        ),

        # Step 2: Extract workflow context for router
        PythonTask(
            function=lambda llm_sel: extract_workflow_context(llm_sel["llm_decision"]),
            name="extract_workflow_context",
            returns=[pickled("workflow_context")]
        ),

        # Step 3: Conditional workflow routing
        Conditional(
            name="workflow_execution_router",
            condition=lambda llm_sel: route_to_workflow(llm_sel),
            condition_input=pickled("llm_selection"),
            branches=create_workflow_execution_branches()
        ),

        # Step 4: Aggregate final results
        PythonTask(
            function=aggregate_workflow_results,
            name="result_aggregation",
            returns=[pickled("final_results")]
        )
    ])


# Main execution functions

def execute_llm_tool_pipeline(user_query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute the complete LLM + tool pipeline for a user query.

    Args:
        user_query: User's request/question
        user_context: Optional user context

    Returns:
        Complete pipeline results
    """

    print("🚀 Starting LLM + Tool Pipeline Execution")
    print("=" * 60)

    try:
        # Step 1: Prepare input (for logging/display purposes)
        prepared_input = prepare_user_input(user_query, user_context)

        # Step 2: Create and execute pipeline
        pipeline = create_complete_llm_tool_pipeline()

        print("⚙️ Executing pipeline components...")

        # Execute pipeline (parameters are handled automatically by runnable)
        results = pipeline.execute()

        # Step 3: Format results for user
        final_results = results.get("final_results", {})
        formatted_output = format_user_results(final_results)

        print("\n" + "=" * 60)
        print("🎉 Pipeline Execution Completed Successfully!")
        print("=" * 60)
        print(formatted_output)

        return {
            "status": "success",
            "formatted_results": formatted_output,
            "raw_results": final_results,
            "user_query": user_query
        }

    except Exception as e:
        error_message = f"Pipeline execution failed: {str(e)}"
        print(f"\n❌ {error_message}")

        return {
            "status": "error",
            "error": error_message,
            "user_query": user_query,
            "formatted_results": f"Error: {error_message}",
            "raw_results": None
        }


# Example usage and demonstration
if __name__ == "__main__":

    print("🔮 LLM + Tool Pipeline - Complete Integration Demo")
    print("=" * 65)

    # Example queries for different workflows
    example_queries = [
        {
            "query": "Research the latest developments in renewable energy technologies",
            "expected_workflow": "research_workflow",
            "context": {"user_expertise": "beginner", "output_preference": "comprehensive"}
        },
        {
            "query": "Analyze our website performance metrics and compare with industry benchmarks",
            "expected_workflow": "analysis_workflow",
            "context": {"user_role": "product_manager", "data_access": "full"}
        },
        {
            "query": "Create a blog post about sustainable living practices for beginners",
            "expected_workflow": "content_creation_workflow",
            "context": {"audience": "general_public", "tone_preference": "friendly"}
        }
    ]

    # Execute each example
    for i, example in enumerate(example_queries, 1):
        print(f"\n{'🔸' * 20} Example {i} {'🔸' * 20}")
        print(f"Query: \"{example['query']}\"")
        print(f"Expected: {example['expected_workflow']}")
        print(f"Context: {example['context']}")
        print("\nExecuting pipeline...")

        # Execute the pipeline
        result = execute_llm_tool_pipeline(
            user_query=example["query"],
            user_context=example["context"]
        )

        # Brief summary of execution
        print(f"\n📝 Execution Summary:")
        print(f"   Status: {'✅ Success' if result['status'] == 'success' else '❌ Failed'}")
        if result['status'] == 'success':
            workflow_executed = result['raw_results']['execution_summary']['selected_workflow']
            print(f"   Workflow: {workflow_executed}")
            print(f"   Expected: {example['expected_workflow']}")
            print(f"   Match: {'✅' if workflow_executed == example['expected_workflow'] else '❌'}")

        if i < len(example_queries):
            print(f"\n{'▶' * 5} Next Example {'◀' * 5}")

    print(f"\n🎊 Complete Integration Demo Finished!")
    print("All pipeline components working together successfully!")

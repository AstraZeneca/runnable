"""
Simple LLM + Tool Pipeline Integration Demo

This is a simplified version that demonstrates the complete LLM + tool orchestration
concept without the complex parameter passing that may not work with all runnable versions.

This example shows:
1. LLM workflow selection
2. Tool execution based on selection
3. Result aggregation

All components work together to demonstrate the core concept.
"""

from typing import Dict, Any, List
from runnable import Pipeline, PythonTask, Conditional, pickled

# Import our components
try:
    from .llm_workflow_selector import llm_workflow_selector
    from .api_tools import web_search_tool, data_analyzer_tool, summarizer_tool
except ImportError:
    from llm_workflow_selector import llm_workflow_selector
    from api_tools import web_search_tool, data_analyzer_tool, summarizer_tool


def analyze_user_query() -> Dict[str, Any]:
    """
    Step 1: Analyze user query and select workflow using LLM agent.
    """
    print("🤖 LLM Agent analyzing user query...")

    # Sample user query (in real implementation, this would come from user input)
    user_query = "Research the latest developments in machine learning"

    # Execute LLM workflow selection
    llm_decision = llm_workflow_selector(user_query)

    print(f"✅ LLM Selected: {llm_decision['selected_workflow']} "
          f"(confidence: {llm_decision['confidence']:.1%})")

    return {
        "user_query": user_query,
        "selected_workflow": llm_decision["selected_workflow"],
        "workflow_parameters": llm_decision["workflow_parameters"],
        "llm_confidence": llm_decision["confidence"],
        "llm_reasoning": llm_decision["reasoning"]
    }


def execute_research_tools() -> Dict[str, Any]:
    """
    Execute research-focused tools: web search + content extraction + summarization.
    """
    print("🔍 Executing research workflow tools...")

    # Step 1: Web search
    search_result = web_search_tool(
        query="machine learning latest developments 2024",
        max_results=3
    )

    if search_result["status"] != "success":
        return {
            "workflow_type": "research",
            "status": "failed",
            "error": search_result.get("error", "Search failed")
        }

    # Step 2: Combine content for summarization
    search_data = search_result["data"]
    combined_content = "\n\n".join([
        f"Title: {result['title']}\nSummary: {result['snippet']}"
        for result in search_data["results"]
    ])

    # Step 3: Summarize findings
    summary_result = summarizer_tool(
        content=combined_content,
        length="medium",
        style="neutral"
    )

    if summary_result["status"] != "success":
        return {
            "workflow_type": "research",
            "status": "partial",
            "search_results": search_data,
            "summary_error": summary_result.get("error")
        }

    return {
        "workflow_type": "research",
        "status": "completed",
        "search_results": {
            "query": "machine learning latest developments 2024",
            "results_count": len(search_data["results"]),
            "sources": [r["title"] for r in search_data["results"]]
        },
        "research_summary": summary_result["data"]["summary"],
        "summary_stats": {
            "original_length": summary_result["data"]["original_length"],
            "summary_length": summary_result["data"]["summary_length"],
            "compression_ratio": summary_result["data"]["compression_ratio"]
        }
    }


def execute_analysis_tools() -> Dict[str, Any]:
    """
    Execute analysis-focused tools: data analysis + pattern detection.
    """
    print("📊 Executing analysis workflow tools...")

    # Generate sample data for analysis
    sample_data = {
        "metrics": {
            "performance": [85, 92, 78, 95, 88, 90, 87],
            "efficiency": [78, 85, 82, 88, 90, 87, 89],
            "satisfaction": [4.2, 4.5, 4.1, 4.7, 4.4, 4.6, 4.3]
        },
        "categories": ["Week1", "Week2", "Week3", "Week4", "Week5", "Week6", "Week7"]
    }

    # Execute data analysis
    analysis_result = data_analyzer_tool(
        data=sample_data,
        analysis_type="descriptive"
    )

    if analysis_result["status"] != "success":
        return {
            "workflow_type": "analysis",
            "status": "failed",
            "error": analysis_result.get("error", "Analysis failed")
        }

    analysis_data = analysis_result["data"]

    return {
        "workflow_type": "analysis",
        "status": "completed",
        "analysis_summary": {
            "data_quality": analysis_data["summary_statistics"]["data_quality_score"],
            "records_analyzed": analysis_data["summary_statistics"]["total_records"],
            "insights_generated": len(analysis_data.get("insights", []))
        },
        "key_insights": analysis_data.get("insights", []),
        "statistical_summary": analysis_data["summary_statistics"]
    }


def execute_content_creation_tools() -> Dict[str, Any]:
    """
    Execute content creation tools: research + generation + formatting.
    """
    print("✍️ Executing content creation workflow tools...")

    # Step 1: Research for content
    search_result = web_search_tool(
        query="sustainable living practices beginners guide",
        max_results=2
    )

    # Step 2: Generate content (simulated)
    content_title = "Beginner's Guide to Sustainable Living"
    content_body = """
# Beginner's Guide to Sustainable Living

Living sustainably doesn't have to be overwhelming. Here are simple steps to get started:

## Energy Conservation
- Switch to LED light bulbs
- Unplug devices when not in use
- Use natural lighting when possible

## Waste Reduction
- Practice the 3 R's: Reduce, Reuse, Recycle
- Compost organic waste
- Choose reusable bags and containers

## Water Conservation
- Fix leaky faucets promptly
- Take shorter showers
- Collect rainwater for plants

## Transportation
- Walk or bike for short trips
- Use public transportation
- Consider carpooling

## Conclusion
Small changes in daily habits can make a significant environmental impact.
Start with one or two changes and gradually incorporate more sustainable practices.
"""

    return {
        "workflow_type": "content_creation",
        "status": "completed",
        "content_summary": {
            "title": content_title,
            "word_count": len(content_body.split()),
            "sections": 6,
            "content_type": "guide"
        },
        "generated_content": content_body,
        "research_sources": len(search_result["data"]["results"]) if search_result["status"] == "success" else 0
    }


def route_to_workflow_execution(query_analysis: Dict[str, Any]) -> str:
    """
    Route to appropriate workflow based on LLM selection.
    """
    workflow_map = {
        "research_workflow": "research_execution",
        "analysis_workflow": "analysis_execution",
        "content_creation_workflow": "content_execution"
    }

    selected = query_analysis["selected_workflow"]
    route = workflow_map.get(selected, "research_execution")  # Default to research

    print(f"🔀 Routing to: {route} (from {selected})")
    return route


def aggregate_final_results(execution_results: Dict[str, Any], query_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate results from workflow execution and format for user.
    """
    print("📋 Aggregating final results...")

    workflow_type = execution_results.get("workflow_type", "unknown")
    status = execution_results.get("status", "unknown")

    final_results = {
        "pipeline_summary": {
            "status": "success" if status == "completed" else "partial",
            "original_query": query_analysis["user_query"],
            "selected_workflow": query_analysis["selected_workflow"],
            "llm_confidence": query_analysis["llm_confidence"],
            "execution_successful": status == "completed"
        },
        "workflow_execution": {
            "type": workflow_type,
            "status": status,
            "results": execution_results
        },
        "llm_analysis": {
            "reasoning": query_analysis["llm_reasoning"],
            "confidence": query_analysis["llm_confidence"],
            "parameters": query_analysis["workflow_parameters"]
        }
    }

    return final_results


def create_simple_llm_tool_pipeline() -> Pipeline:
    """
    Create a simplified LLM + tool pipeline for demonstration.
    """
    return Pipeline([
        # Step 1: LLM Query Analysis
        PythonTask(
            function=analyze_user_query,
            name="llm_query_analysis",
            returns=[pickled("query_analysis")]
        ),

        # Step 2: Conditional Workflow Routing
        Conditional(
            name="workflow_router",
            condition=lambda qa: route_to_workflow_execution(qa),
            condition_input=pickled("query_analysis"),
            branches={
                "research_execution": Pipeline([
                    PythonTask(
                        function=execute_research_tools,
                        name="research_tools",
                        returns=[pickled("execution_results")]
                    )
                ]),
                "analysis_execution": Pipeline([
                    PythonTask(
                        function=execute_analysis_tools,
                        name="analysis_tools",
                        returns=[pickled("execution_results")]
                    )
                ]),
                "content_execution": Pipeline([
                    PythonTask(
                        function=execute_content_creation_tools,
                        name="content_tools",
                        returns=[pickled("execution_results")]
                    )
                ])
            }
        ),

        # Step 3: Result Aggregation
        PythonTask(
            function=lambda er, qa: aggregate_final_results(er, qa),
            name="result_aggregation",
            returns=[pickled("final_results")]
        )
    ])


def format_demo_results(results: Dict[str, Any]) -> str:
    """
    Format results for user-friendly display.
    """
    pipeline_summary = results["pipeline_summary"]
    workflow_execution = results["workflow_execution"]
    llm_analysis = results["llm_analysis"]

    output = f"""
🎯 LLM + Tool Pipeline Demo Results
{'=' * 50}

📝 Original Query: "{pipeline_summary['original_query']}"
🤖 LLM Selected: {pipeline_summary['selected_workflow']}
📊 Confidence: {pipeline_summary['llm_confidence']:.1%}
✅ Status: {'SUCCESS' if pipeline_summary['execution_successful'] else 'PARTIAL'}

🔍 LLM Reasoning: {llm_analysis['reasoning']}

🛠️ Workflow Execution ({workflow_execution['type']}):
"""

    execution_results = workflow_execution["results"]

    if workflow_execution["type"] == "research":
        output += f"""
- Search Query: {execution_results.get('search_results', {}).get('query', 'N/A')}
- Sources Found: {execution_results.get('search_results', {}).get('results_count', 0)}
- Summary Generated: {'Yes' if 'research_summary' in execution_results else 'No'}
- Compression Ratio: {execution_results.get('summary_stats', {}).get('compression_ratio', 'N/A')}

📄 Research Summary:
{execution_results.get('research_summary', 'No summary available')[:300]}...
"""

    elif workflow_execution["type"] == "analysis":
        output += f"""
- Records Analyzed: {execution_results.get('analysis_summary', {}).get('records_analyzed', 0)}
- Data Quality: {execution_results.get('analysis_summary', {}).get('data_quality', 'N/A')}
- Insights Generated: {execution_results.get('analysis_summary', {}).get('insights_generated', 0)}

💡 Key Insights:"""

        for i, insight in enumerate(execution_results.get('key_insights', [])[:3], 1):
            output += f"\n{i}. {insight}"

    elif workflow_execution["type"] == "content_creation":
        output += f"""
- Content Type: {execution_results.get('content_summary', {}).get('content_type', 'N/A')}
- Word Count: {execution_results.get('content_summary', {}).get('word_count', 0)}
- Sections: {execution_results.get('content_summary', {}).get('sections', 0)}
- Research Sources: {execution_results.get('research_sources', 0)}

📝 Generated Content Preview:
{execution_results.get('generated_content', 'No content available')[:200]}...
"""

    return output.strip()


def run_simple_demo():
    """
    Run the simplified LLM + tool pipeline demo.
    """
    print("🚀 Simple LLM + Tool Pipeline Demo")
    print("=" * 60)

    try:
        # Create and execute pipeline
        pipeline = create_simple_llm_tool_pipeline()

        print("⚙️ Executing pipeline...")
        results = pipeline.execute()

        # Format and display results
        final_results = results.get("final_results", {})
        formatted_output = format_demo_results(final_results)

        print("\n" + "=" * 60)
        print("🎉 Demo Completed Successfully!")
        print("=" * 60)
        print(formatted_output)

        return {"status": "success", "results": final_results}

    except Exception as e:
        error_message = f"Demo failed: {str(e)}"
        print(f"\n❌ {error_message}")
        return {"status": "error", "error": error_message}


if __name__ == "__main__":
    print("🔮 LLM + Tool Orchestration - Simplified Demo")
    print("Demonstrating core concepts without complex parameter passing")
    print("\n")

    result = run_simple_demo()

    if result["status"] == "success":
        print("\n✨ Demo showcases:")
        print("   - LLM workflow selection")
        print("   - Conditional routing to appropriate tools")
        print("   - Mixed orchestration patterns")
        print("   - Result aggregation and formatting")
        print("\n🎯 All components working together successfully!")

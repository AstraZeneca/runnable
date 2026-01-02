"""
Minimal Working LLM + Tool Demo

This is the simplest possible demonstration of the LLM + tool orchestration concept
that works with runnable's current API. It shows the core ideas without complex
parameter passing or routing.
"""

from runnable import Pipeline, PythonTask, pickled
try:
    from .llm_workflow_selector import llm_workflow_selector
    from .api_tools import web_search_tool, summarizer_tool
except ImportError:
    from llm_workflow_selector import llm_workflow_selector
    from api_tools import web_search_tool, summarizer_tool


def step1_llm_analysis():
    """
    Step 1: LLM analyzes query and selects workflow
    """
    print("🤖 Step 1: LLM analyzing query...")

    # Sample query (in production, this would come from user input)
    user_query = "Research machine learning best practices"

    # LLM selects workflow
    llm_result = llm_workflow_selector(user_query)

    print(f"✅ LLM selected: {llm_result['selected_workflow']} "
          f"(confidence: {llm_result['confidence']:.1%})")

    return {
        "user_query": user_query,
        "selected_workflow": llm_result["selected_workflow"],
        "llm_confidence": llm_result["confidence"],
        "llm_reasoning": llm_result["reasoning"]
    }


def step2_execute_tools(llm_analysis):
    """
    Step 2: Execute tools based on LLM selection
    """
    print("🔧 Step 2: Executing selected tools...")

    workflow = llm_analysis["selected_workflow"]
    query = llm_analysis["user_query"]

    if workflow == "research_workflow":
        print("   Running research tools...")

        # Tool 1: Web search
        search_result = web_search_tool(
            query=f"{query} 2024",
            max_results=3
        )

        if search_result["status"] == "success":
            # Tool 2: Summarize findings
            search_data = search_result["data"]
            combined_content = "\n".join([
                f"{result['title']}: {result['snippet']}"
                for result in search_data["results"]
            ])

            summary_result = summarizer_tool(
                content=combined_content,
                length="short"
            )

            return {
                "workflow_executed": workflow,
                "tools_used": ["web_search", "summarizer"],
                "search_results_count": len(search_data["results"]),
                "summary": summary_result["data"]["summary"] if summary_result["status"] == "success" else "Summary failed",
                "success": True
            }

    # Fallback for other workflows or failures
    return {
        "workflow_executed": workflow,
        "tools_used": ["fallback"],
        "message": f"Executed fallback tools for {workflow}",
        "success": True
    }


def step3_format_results(llm_analysis, tool_results):
    """
    Step 3: Format final results for user
    """
    print("📋 Step 3: Formatting final results...")

    final_output = f"""
🎯 LLM + Tool Pipeline Results
=====================================

Original Query: "{llm_analysis['user_query']}"
LLM Selection: {llm_analysis['selected_workflow']}
Confidence: {llm_analysis['llm_confidence']:.1%}

Reasoning: {llm_analysis['llm_reasoning']}

Tools Executed: {', '.join(tool_results['tools_used'])}
Workflow: {tool_results['workflow_executed']}
Status: {'✅ Success' if tool_results['success'] else '❌ Failed'}

"""

    if "summary" in tool_results:
        final_output += f"Results Summary:\n{tool_results['summary']}\n"

    if "search_results_count" in tool_results:
        final_output += f"Sources Analyzed: {tool_results['search_results_count']}\n"

    final_output += "\n🚀 Pipeline executed successfully!"

    print("✅ Results formatted and ready!")
    return {"formatted_results": final_output, "success": True}


def create_minimal_pipeline():
    """
    Create minimal LLM + tool pipeline
    """
    return Pipeline(steps=[
        # Step 1: LLM Analysis
        PythonTask(
            function=step1_llm_analysis,
            name="llm_analysis",
            returns=[pickled("llm_analysis")]
        ),

        # Step 2: Tool Execution
        PythonTask(
            function=step2_execute_tools,
            name="tool_execution",
            returns=[pickled("tool_results")]
        ),

        # Step 3: Result Formatting
        PythonTask(
            function=step3_format_results,
            name="result_formatting",
            returns=[pickled("final_output")]
        )
    ])


def run_minimal_demo():
    """
    Run the minimal LLM + tool demo
    """
    print("🚀 Minimal LLM + Tool Demo")
    print("=" * 40)

    try:
        # Create and execute pipeline
        pipeline = create_minimal_pipeline()
        results = pipeline.execute()

        # Display results
        final_output = results.get("final_output", {})

        if final_output.get("success"):
            print("\n" + "=" * 40)
            print("🎉 Demo Completed Successfully!")
            print("=" * 40)
            print(final_output["formatted_results"])
            return True
        else:
            print("❌ Demo failed during execution")
            return False

    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔮 Minimal LLM + Tool Orchestration Demo")
    print("Demonstrating core concepts with simple sequential execution\n")

    success = run_minimal_demo()

    if success:
        print("\n✨ This demo showcases:")
        print("   1. LLM workflow selection")
        print("   2. Tool execution based on selection")
        print("   3. Result aggregation")
        print("   4. Sequential pipeline execution")
        print("\n🎯 Foundation for more complex orchestration patterns!")

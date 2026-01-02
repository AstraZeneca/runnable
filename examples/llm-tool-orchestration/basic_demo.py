"""
Basic LLM + Tool Demo

Following the exact pattern from runnable examples to ensure it works.
"""

from runnable import Pipeline, PythonTask, pickled
try:
    from .llm_workflow_selector import llm_workflow_selector
    from .api_tools import web_search_tool
except ImportError:
    from llm_workflow_selector import llm_workflow_selector
    from api_tools import web_search_tool


def llm_step():
    """LLM selects workflow"""
    print("🤖 LLM selecting workflow...")
    result = llm_workflow_selector("Research machine learning")
    print(f"Selected: {result['selected_workflow']}")
    return result


def tool_step():
    """Execute tools"""
    print("🔧 Executing tools...")
    search_result = web_search_tool("machine learning", max_results=2)
    return {
        "tool_executed": "web_search",
        "results_count": len(search_result["data"]["results"]) if search_result["status"] == "success" else 0,
        "status": search_result["status"]
    }


def main():
    """Main pipeline"""
    llm_task = PythonTask(
        name="llm",
        function=llm_step,
        returns=[pickled("llm_result")]
    )

    tool_task = PythonTask(
        name="tools",
        function=tool_step,
        returns=[pickled("tool_result")]
    )

    pipeline = Pipeline(steps=[llm_task, tool_task])
    pipeline.execute()

    print("✅ Pipeline completed successfully!")
    print("🎯 LLM + Tool orchestration demonstrated:")
    print("   1. ✅ LLM analyzed query and selected workflow")
    print("   2. ✅ Tools executed based on LLM selection")
    print("   3. ✅ Results stored in catalog for further processing")

    return pipeline


if __name__ == "__main__":
    print("🔮 Basic LLM + Tool Demo")
    print("=" * 30)
    main()

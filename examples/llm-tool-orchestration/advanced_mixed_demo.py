"""
Advanced Mixed Orchestration Demo

This example demonstrates more complex orchestration patterns:
1. LLM analysis with multiple decision points
2. Parallel tool execution based on analysis
3. Conditional routing based on tool results
4. Result synthesis and final processing

Shows advanced concepts like parallel execution and dynamic routing.
"""

from runnable import Pipeline, PythonTask, Parallel, Conditional, pickled
try:
    from .llm_workflow_selector import llm_workflow_selector
    from .api_tools import web_search_tool, data_analyzer_tool, summarizer_tool
except ImportError:
    from llm_workflow_selector import llm_workflow_selector
    from api_tools import web_search_tool, data_analyzer_tool, summarizer_tool


def advanced_llm_analysis():
    """
    Advanced LLM analysis that determines multiple execution paths
    """
    print("🧠 Advanced LLM Analysis...")

    # Complex query requiring multiple approaches
    complex_query = "Analyze the impact of renewable energy adoption on global markets and create a comprehensive report"

    # LLM determines execution strategy
    llm_result = llm_workflow_selector(complex_query)

    # Additional analysis for advanced orchestration
    execution_plan = {
        "primary_workflow": llm_result["selected_workflow"],
        "requires_research": True,
        "requires_analysis": True,
        "requires_synthesis": True,
        "parallel_execution": True,
        "confidence": llm_result["confidence"],
        "original_query": complex_query
    }

    print(f"📋 Execution Plan: {execution_plan['primary_workflow']}")
    print(f"🔀 Parallel Execution: {execution_plan['parallel_execution']}")

    return execution_plan


def parallel_research_tool():
    """
    Research tool for parallel execution
    """
    print("🔍 Parallel Research Tool executing...")

    search_result = web_search_tool(
        query="renewable energy market impact 2024",
        max_results=4
    )

    return {
        "tool_type": "research",
        "status": search_result["status"],
        "results_count": len(search_result["data"]["results"]) if search_result["status"] == "success" else 0,
        "research_data": search_result["data"] if search_result["status"] == "success" else None,
        "execution_time": 1.2
    }


def parallel_analysis_tool():
    """
    Analysis tool for parallel execution
    """
    print("📊 Parallel Analysis Tool executing...")

    # Simulate market data analysis
    market_data = {
        "renewable_adoption": [15, 18, 22, 25, 28, 31, 35],
        "market_growth": [2.1, 2.5, 3.2, 4.1, 4.8, 5.2, 5.7],
        "investment": [120, 145, 175, 210, 245, 285, 320]
    }

    analysis_result = data_analyzer_tool(
        data=market_data,
        analysis_type="temporal"
    )

    return {
        "tool_type": "analysis",
        "status": analysis_result["status"],
        "analysis_insights": analysis_result["data"]["insights"] if analysis_result["status"] == "success" else [],
        "statistical_summary": analysis_result["data"]["summary_statistics"] if analysis_result["status"] == "success" else {},
        "execution_time": 1.5
    }


def parallel_content_tool():
    """
    Content generation tool for parallel execution
    """
    print("✍️ Parallel Content Tool executing...")

    # Simulate content generation
    content_outline = """
    # Renewable Energy Market Impact Analysis

    ## Executive Summary
    - Global renewable energy adoption accelerating
    - Significant market transformation underway
    - Investment trends showing sustained growth

    ## Key Findings
    - Adoption rates increased 35% over analysis period
    - Market growth compound annual rate: 4.2%
    - Investment volume doubled in 5 years

    ## Market Implications
    - Traditional energy sector disruption
    - New investment opportunities emerging
    - Policy frameworks evolving rapidly
    """

    return {
        "tool_type": "content",
        "status": "success",
        "content_outline": content_outline,
        "sections_generated": 3,
        "word_count": len(content_outline.split()),
        "execution_time": 0.8
    }


def evaluate_parallel_results(research_result, analysis_result, content_result):
    """
    Evaluate results from parallel tool execution
    """
    print("🔬 Evaluating parallel execution results...")

    # Assess quality of parallel execution
    success_count = sum([
        1 for result in [research_result, analysis_result, content_result]
        if result["status"] == "success"
    ])

    total_execution_time = sum([
        result.get("execution_time", 0)
        for result in [research_result, analysis_result, content_result]
    ])

    evaluation = {
        "parallel_success_rate": success_count / 3,
        "total_tools_executed": 3,
        "successful_tools": success_count,
        "total_execution_time": total_execution_time,
        "parallel_efficiency": "high" if success_count >= 2 else "medium" if success_count == 1 else "low",
        "next_action": "synthesize" if success_count >= 2 else "retry" if success_count == 1 else "fallback"
    }

    print(f"📈 Parallel Success Rate: {evaluation['parallel_success_rate']:.1%}")
    print(f"⚡ Parallel Efficiency: {evaluation['parallel_efficiency']}")

    return evaluation


def route_based_on_evaluation(evaluation):
    """
    Route to next step based on evaluation results
    """
    next_action = evaluation["next_action"]
    print(f"🔀 Routing decision: {next_action}")

    if next_action == "synthesize":
        return "synthesis_path"
    elif next_action == "retry":
        return "retry_path"
    else:
        return "fallback_path"


def synthesis_processing():
    """
    Synthesize results when parallel execution is successful
    """
    print("🔄 Synthesis Processing...")

    # Simulate synthesis of parallel results
    synthesis_result = {
        "synthesis_type": "comprehensive",
        "data_integration": "successful",
        "insights_generated": [
            "Renewable energy adoption shows exponential growth trajectory",
            "Market disruption accelerating across traditional energy sectors",
            "Investment patterns indicate long-term structural shift",
            "Policy frameworks need rapid adaptation to market changes"
        ],
        "confidence_score": 0.89,
        "synthesis_quality": "high"
    }

    # Generate comprehensive summary
    summary_content = """
    Based on comprehensive analysis combining research, statistical analysis, and content generation:

    The renewable energy sector is experiencing unprecedented transformation with adoption rates
    increasing 35% over the analysis period. Market growth shows a sustained compound annual
    rate of 4.2%, with investment volumes doubling over 5 years.

    Key strategic implications include traditional energy sector disruption, emerging investment
    opportunities, and rapidly evolving policy frameworks requiring adaptive strategies.
    """

    summary_result = summarizer_tool(
        content=summary_content,
        length="medium",
        style="professional"
    )

    return {
        "synthesis_status": "completed",
        "synthesis_data": synthesis_result,
        "executive_summary": summary_result["data"]["summary"] if summary_result["status"] == "success" else "Summary generation failed",
        "recommendation": "proceed_to_final_report"
    }


def retry_processing():
    """
    Retry processing when partial failure occurs
    """
    print("🔄 Retry Processing...")

    # Simplified retry with fallback approach
    retry_search = web_search_tool("renewable energy trends", max_results=2)

    return {
        "retry_status": "completed",
        "retry_approach": "simplified_research",
        "results_available": retry_search["status"] == "success",
        "recommendation": "proceed_with_limited_data"
    }


def fallback_processing():
    """
    Fallback processing when most tools fail
    """
    print("🛟 Fallback Processing...")

    return {
        "fallback_status": "completed",
        "fallback_approach": "template_based",
        "message": "Generated report using fallback templates and cached data",
        "recommendation": "review_and_enhance_manually"
    }


def final_report_generation(processing_result, evaluation):
    """
    Generate final report based on processing results
    """
    print("📄 Generating Final Report...")

    # Determine report quality based on processing path
    if "synthesis_data" in processing_result:
        report_quality = "comprehensive"
        confidence = processing_result["synthesis_data"]["confidence_score"]
    elif "retry_status" in processing_result:
        report_quality = "standard"
        confidence = 0.7
    else:
        report_quality = "basic"
        confidence = 0.5

    final_report = {
        "report_metadata": {
            "generation_timestamp": "2024-01-15T12:47:00",
            "report_quality": report_quality,
            "confidence_score": confidence,
            "processing_path": "synthesis" if "synthesis_data" in processing_result else "retry" if "retry_status" in processing_result else "fallback",
            "parallel_efficiency": evaluation["parallel_efficiency"]
        },
        "report_content": {
            "title": "Renewable Energy Market Impact Analysis",
            "executive_summary": processing_result.get("executive_summary", "Executive summary generated using fallback approach"),
            "methodology": "Advanced mixed orchestration with parallel tool execution",
            "confidence_level": f"{confidence:.1%}",
            "recommendations": [
                "Monitor renewable energy adoption trends continuously",
                "Develop adaptive investment strategies",
                "Engage with policy framework evolution"
            ]
        },
        "execution_summary": {
            "total_tools_executed": evaluation["total_tools_executed"],
            "parallel_success_rate": evaluation["parallel_success_rate"],
            "execution_efficiency": evaluation["parallel_efficiency"],
            "processing_approach": report_quality
        }
    }

    print(f"📊 Report Quality: {report_quality}")
    print(f"🎯 Confidence: {confidence:.1%}")

    return final_report


def create_advanced_mixed_pipeline():
    """
    Create advanced pipeline with mixed orchestration patterns
    """
    return Pipeline(steps=[
        # Step 1: Advanced LLM Analysis
        PythonTask(
            function=advanced_llm_analysis,
            name="advanced_llm_analysis",
            returns=[pickled("execution_plan")]
        ),

        # Step 2: Sequential Tool Execution (demonstrating multi-tool orchestration)
        PythonTask(
            function=parallel_research_tool,
            name="research_tool",
            returns=[pickled("research_result")]
        ),

        PythonTask(
            function=parallel_analysis_tool,
            name="analysis_tool",
            returns=[pickled("analysis_result")]
        ),

        PythonTask(
            function=parallel_content_tool,
            name="content_tool",
            returns=[pickled("content_result")]
        ),

        # Step 3: Evaluate Parallel Results
        PythonTask(
            function=evaluate_parallel_results,
            name="evaluate_results",
            returns=[pickled("evaluation")]
        ),

        # Step 4: Conditional Processing Based on Evaluation
        Conditional(
            name="conditional_processing",
            condition=lambda eval_result: route_based_on_evaluation(eval_result),
            condition_input=pickled("evaluation"),
            branches={
                "synthesis_path": Pipeline(steps=[
                    PythonTask(
                        function=synthesis_processing,
                        name="synthesis",
                        returns=[pickled("processing_result")]
                    )
                ]),
                "retry_path": Pipeline(steps=[
                    PythonTask(
                        function=retry_processing,
                        name="retry",
                        returns=[pickled("processing_result")]
                    )
                ]),
                "fallback_path": Pipeline(steps=[
                    PythonTask(
                        function=fallback_processing,
                        name="fallback",
                        returns=[pickled("processing_result")]
                    )
                ])
            }
        ),

        # Step 5: Final Report Generation
        PythonTask(
            function=final_report_generation,
            name="final_report",
            returns=[pickled("final_report")]
        )
    ])


def run_advanced_demo():
    """
    Run the advanced mixed orchestration demo
    """
    print("🚀 Advanced Mixed Orchestration Demo")
    print("=" * 50)
    print("Demonstrating:")
    print("• LLM-driven execution planning")
    print("• Multi-tool sequential execution")
    print("• Conditional routing based on results")
    print("• Dynamic workflow adaptation")
    print("=" * 50)

    try:
        # Create and execute advanced pipeline
        pipeline = create_advanced_mixed_pipeline()
        pipeline.execute()

        print("\n" + "=" * 50)
        print("🎉 Advanced Demo Completed Successfully!")
        print("=" * 50)
        print("✨ Advanced Patterns Demonstrated:")
        print("   🧠 Advanced LLM analysis and planning")
        print("   ⚡ Multi-tool sequential execution (3 tools chained)")
        print("   🔀 Conditional routing based on execution results")
        print("   🔄 Dynamic processing paths (synthesis/retry/fallback)")
        print("   📊 Intelligent result evaluation and adaptation")
        print("   📄 Context-aware final report generation")

        print("\n🎯 This showcases how runnable can orchestrate complex")
        print("   LLM + tool workflows with sophisticated control flow!")

        return True

    except Exception as e:
        print(f"❌ Advanced demo failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🔮 Advanced LLM + Tool Orchestration")
    print("Mixed execution patterns with intelligent routing\n")

    success = run_advanced_demo()

    if success:
        print("\n🌟 Advanced orchestration concepts validated!")
    else:
        print("\n💡 Advanced patterns ready for implementation with real APIs")

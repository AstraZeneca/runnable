"""
Analysis Workflow Template

This workflow template implements a comprehensive data analysis pipeline:
1. Data Collection - Gather data from multiple sources
2. Parallel Analysis - Run multiple analysis tools simultaneously
3. Results Synthesis - Combine analysis results into insights
4. Report Generation - Create structured analysis reports

Demonstrates mixed orchestration: sequential setup → parallel analysis → sequential synthesis.
"""

from typing import Dict, List, Any
from runnable import Pipeline, PythonTask, Parallel, pickled
try:
    from .api_tools import web_search_tool, data_analyzer_tool, http_api_tool
except ImportError:
    # Handle direct execution
    from api_tools import web_search_tool, data_analyzer_tool, http_api_tool


def prepare_analysis_parameters(workflow_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare analysis parameters based on workflow context.

    Args:
        workflow_context: Context from workflow router

    Returns:
        Analysis configuration and parameters
    """

    execution_context = workflow_context.get("execution_context", {})
    workflow_params = workflow_context.get("workflow_parameters", {})

    # Extract analysis parameters
    original_query = execution_context.get("original_query", "")
    analysis_type = workflow_params.get("analysis_type", "descriptive")
    comparison_mode = workflow_params.get("comparison_mode", False)
    metrics_focus = workflow_params.get("metrics_focus", ["general"])
    output_format = workflow_params.get("output_format", "structured_report")

    return {
        "analysis_config": {
            "query": original_query,
            "analysis_type": analysis_type,
            "comparison_mode": comparison_mode,
            "metrics_focus": metrics_focus,
            "output_format": output_format,
            "parallel_analysis_enabled": True
        },
        "data_sources": determine_data_sources(original_query, metrics_focus),
        "analysis_tools": select_analysis_tools(analysis_type, comparison_mode),
        "execution_metadata": {
            "workflow_name": execution_context.get("workflow_name"),
            "llm_confidence": execution_context.get("llm_confidence", 0.5)
        }
    }


def determine_data_sources(query: str, metrics_focus: List[str]) -> List[Dict[str, Any]]:
    """
    Determine appropriate data sources based on query and metrics focus.

    Args:
        query: Original analysis query
        metrics_focus: List of metrics to focus on

    Returns:
        List of data source configurations
    """

    data_sources = []

    # Always include web search for contextual information
    data_sources.append({
        "source_type": "web_search",
        "source_name": "contextual_data",
        "config": {
            "query": f"{query} data statistics metrics",
            "max_results": 5,
            "search_type": "web"
        }
    })

    # Add specific data sources based on metrics focus
    for metric in metrics_focus:
        if metric in ["performance", "efficiency", "speed"]:
            data_sources.append({
                "source_type": "performance_data",
                "source_name": f"{metric}_metrics",
                "config": {
                    "metric_type": metric,
                    "time_range": "30_days"
                }
            })
        elif metric in ["revenue", "cost", "financial"]:
            data_sources.append({
                "source_type": "financial_data",
                "source_name": f"{metric}_data",
                "config": {
                    "metric_type": metric,
                    "period": "quarterly"
                }
            })

    return data_sources


def select_analysis_tools(analysis_type: str, comparison_mode: bool) -> List[Dict[str, Any]]:
    """
    Select appropriate analysis tools based on requirements.

    Args:
        analysis_type: Type of analysis to perform
        comparison_mode: Whether comparative analysis is needed

    Returns:
        List of analysis tool configurations
    """

    tools = []

    # Base descriptive analysis
    tools.append({
        "tool_name": "descriptive_analyzer",
        "tool_type": "statistical_analysis",
        "config": {
            "analysis_type": "descriptive",
            "include_distributions": True,
            "confidence_level": 0.95
        }
    })

    # Add specific tools based on analysis type
    if analysis_type == "comparative":
        tools.append({
            "tool_name": "comparative_analyzer",
            "tool_type": "comparative_analysis",
            "config": {
                "comparison_method": "statistical_test",
                "significance_level": 0.05
            }
        })

    if analysis_type == "predictive":
        tools.append({
            "tool_name": "predictive_analyzer",
            "tool_type": "predictive_analysis",
            "config": {
                "forecast_horizon": "3_months",
                "model_type": "time_series"
            }
        })

    if analysis_type == "temporal":
        tools.append({
            "tool_name": "trend_analyzer",
            "tool_type": "temporal_analysis",
            "config": {
                "trend_detection": True,
                "seasonal_analysis": True
            }
        })

    # Always add pattern detection
    tools.append({
        "tool_name": "pattern_detector",
        "tool_type": "pattern_analysis",
        "config": {
            "pattern_types": ["correlation", "clustering"],
            "sensitivity": "medium"
        }
    })

    return tools


def collect_analysis_data(analysis_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect data from all configured sources.

    Args:
        analysis_parameters: Analysis configuration

    Returns:
        Collected data from all sources
    """

    data_sources = analysis_parameters["data_sources"]
    collected_data = {}
    collection_metadata = {
        "sources_attempted": len(data_sources),
        "sources_successful": 0,
        "collection_errors": []
    }

    for source in data_sources:
        source_name = source["source_name"]
        source_type = source["source_type"]
        config = source["config"]

        try:
            if source_type == "web_search":
                # Collect contextual data via web search
                search_result = web_search_tool(
                    query=config["query"],
                    max_results=config["max_results"],
                    search_type=config["search_type"]
                )

                if search_result["status"] == "success":
                    collected_data[source_name] = {
                        "data": search_result["data"],
                        "source_type": source_type,
                        "collection_status": "success"
                    }
                    collection_metadata["sources_successful"] += 1
                else:
                    collected_data[source_name] = {
                        "data": None,
                        "source_type": source_type,
                        "collection_status": "failed",
                        "error": search_result.get("error")
                    }
                    collection_metadata["collection_errors"].append(f"{source_name}: {search_result.get('error', 'Unknown error')}")

            elif source_type in ["performance_data", "financial_data"]:
                # Simulate API data collection
                api_result = http_api_tool(
                    endpoint=f"https://api.example.com/{source_type}",
                    method="GET",
                    params=config
                )

                if api_result["status"] == "success":
                    # Generate mock data for this source type
                    mock_data = generate_mock_data_for_source(source_type, config)
                    collected_data[source_name] = {
                        "data": mock_data,
                        "source_type": source_type,
                        "collection_status": "success"
                    }
                    collection_metadata["sources_successful"] += 1
                else:
                    collected_data[source_name] = {
                        "data": None,
                        "source_type": source_type,
                        "collection_status": "failed",
                        "error": api_result.get("error")
                    }
                    collection_metadata["collection_errors"].append(f"{source_name}: {api_result.get('error', 'API error')}")

        except Exception as e:
            collected_data[source_name] = {
                "data": None,
                "source_type": source_type,
                "collection_status": "failed",
                "error": str(e)
            }
            collection_metadata["collection_errors"].append(f"{source_name}: {str(e)}")

    return {
        "collected_data": collected_data,
        "collection_metadata": collection_metadata,
        "analysis_parameters": analysis_parameters
    }


def generate_mock_data_for_source(source_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate appropriate mock data based on source type."""

    import random

    if source_type == "performance_data":
        return {
            "metrics": {
                "response_time": [random.uniform(100, 500) for _ in range(30)],
                "throughput": [random.uniform(50, 200) for _ in range(30)],
                "error_rate": [random.uniform(0, 5) for _ in range(30)]
            },
            "time_series": [f"2024-01-{i+1:02d}" for i in range(30)],
            "metadata": {
                "metric_type": config.get("metric_type", "performance"),
                "unit": "milliseconds" if "time" in config.get("metric_type", "") else "count"
            }
        }

    elif source_type == "financial_data":
        return {
            "metrics": {
                "revenue": [random.uniform(100000, 500000) for _ in range(12)],
                "costs": [random.uniform(50000, 200000) for _ in range(12)],
                "profit": [random.uniform(10000, 100000) for _ in range(12)]
            },
            "periods": [f"2024-Q{i+1}" for i in range(12)],
            "metadata": {
                "metric_type": config.get("metric_type", "financial"),
                "currency": "USD"
            }
        }

    return {"mock_data": "No specific mock data for this source type"}


# Parallel Analysis Functions

def run_descriptive_analysis(data_collection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run descriptive statistical analysis on collected data.

    This function will be executed in parallel with other analysis tools.
    """

    collected_data = data_collection["collected_data"]

    # Extract numerical data for analysis
    numerical_datasets = []
    for source_name, source_data in collected_data.items():
        if source_data["collection_status"] == "success" and source_data["data"]:
            data = source_data["data"]
            if "metrics" in data:
                for metric_name, values in data["metrics"].items():
                    numerical_datasets.append({
                        "source": source_name,
                        "metric": metric_name,
                        "values": values
                    })

    # Run analysis using data analyzer tool
    analysis_result = data_analyzer_tool(
        data={"datasets": numerical_datasets},
        analysis_type="descriptive",
        metrics=["mean", "std", "distribution"]
    )

    return {
        "analysis_name": "descriptive_analysis",
        "analysis_status": analysis_result["status"],
        "results": analysis_result["data"] if analysis_result["status"] == "success" else None,
        "datasets_analyzed": len(numerical_datasets),
        "error": analysis_result.get("error"),
        "metadata": analysis_result.get("metadata", {})
    }


def run_comparative_analysis(data_collection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comparative analysis on collected data.

    This function will be executed in parallel with other analysis tools.
    """

    collected_data = data_collection["collected_data"]

    # Prepare comparative datasets
    comparative_datasets = []
    for source_name, source_data in collected_data.items():
        if source_data["collection_status"] == "success" and source_data["data"]:
            data = source_data["data"]
            if "metrics" in data and len(data["metrics"]) >= 2:
                # Create comparison between first two metrics
                metrics = list(data["metrics"].items())
                comparative_datasets.append({
                    "source": source_name,
                    "group_a": {"name": metrics[0][0], "values": metrics[0][1]},
                    "group_b": {"name": metrics[1][0], "values": metrics[1][1]}
                })

    # Run comparative analysis
    analysis_result = data_analyzer_tool(
        data={"comparisons": comparative_datasets},
        analysis_type="comparative",
        metrics=["significance", "effect_size"]
    )

    return {
        "analysis_name": "comparative_analysis",
        "analysis_status": analysis_result["status"],
        "results": analysis_result["data"] if analysis_result["status"] == "success" else None,
        "comparisons_analyzed": len(comparative_datasets),
        "error": analysis_result.get("error"),
        "metadata": analysis_result.get("metadata", {})
    }


def run_pattern_detection(data_collection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run pattern detection on collected data.

    This function will be executed in parallel with other analysis tools.
    """

    collected_data = data_collection["collected_data"]

    # Prepare data for pattern detection
    pattern_datasets = []
    for source_name, source_data in collected_data.items():
        if source_data["collection_status"] == "success" and source_data["data"]:
            data = source_data["data"]
            if "metrics" in data:
                pattern_datasets.append({
                    "source": source_name,
                    "data": data["metrics"]
                })

    # Run pattern analysis
    analysis_result = data_analyzer_tool(
        data={"pattern_data": pattern_datasets},
        analysis_type="pattern_detection",
        metrics=["correlation", "clustering"]
    )

    return {
        "analysis_name": "pattern_detection",
        "analysis_status": analysis_result["status"],
        "results": analysis_result["data"] if analysis_result["status"] == "success" else None,
        "datasets_analyzed": len(pattern_datasets),
        "error": analysis_result.get("error"),
        "metadata": analysis_result.get("metadata", {})
    }


def synthesize_analysis_results(analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Synthesize results from all parallel analysis tools.

    Args:
        analysis_results: Results from all parallel analysis functions

    Returns:
        Synthesized analysis insights and conclusions
    """

    successful_analyses = []
    failed_analyses = []

    # Categorize results
    for result in analysis_results:
        if result["analysis_status"] == "success":
            successful_analyses.append(result)
        else:
            failed_analyses.append(result)

    # Extract insights from successful analyses
    insights = []
    recommendations = []
    key_findings = {}

    for analysis in successful_analyses:
        analysis_name = analysis["analysis_name"]
        results = analysis["results"]

        if analysis_name == "descriptive_analysis" and results:
            insights.append("Descriptive statistics reveal data distribution patterns and central tendencies.")
            if "summary_statistics" in results:
                key_findings["descriptive"] = results["summary_statistics"]

        elif analysis_name == "comparative_analysis" and results:
            insights.append("Comparative analysis identifies significant differences between data groups.")
            if "comparison_results" in results:
                key_findings["comparative"] = results["comparison_results"]

        elif analysis_name == "pattern_detection" and results:
            insights.append("Pattern analysis reveals underlying correlations and data structures.")
            key_findings["patterns"] = results.get("pattern_summary", {})

    # Generate recommendations based on findings
    if key_findings:
        recommendations.extend(generate_analysis_recommendations(key_findings))

    return {
        "synthesis_status": "success",
        "analysis_summary": {
            "total_analyses": len(analysis_results),
            "successful_analyses": len(successful_analyses),
            "failed_analyses": len(failed_analyses),
            "success_rate": len(successful_analyses) / len(analysis_results) if analysis_results else 0
        },
        "key_insights": insights,
        "key_findings": key_findings,
        "recommendations": recommendations,
        "failed_analyses": [{"name": a["analysis_name"], "error": a["error"]} for a in failed_analyses]
    }


def generate_analysis_recommendations(key_findings: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on analysis findings."""

    recommendations = []

    if "descriptive" in key_findings:
        desc_stats = key_findings["descriptive"]
        if desc_stats.get("data_quality_score", 1.0) < 0.8:
            recommendations.append("Improve data quality by addressing missing values and outliers")

    if "comparative" in key_findings:
        comp_results = key_findings["comparative"]
        if "statistical_significance" in comp_results:
            recommendations.append("Investigate factors causing significant group differences")

    if "patterns" in key_findings:
        recommendations.append("Leverage identified patterns for predictive modeling")

    # Add general recommendations
    recommendations.append("Continue monitoring key metrics for trend identification")
    recommendations.append("Consider expanding analysis to additional data sources")

    return recommendations


def generate_analysis_report(synthesis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate final analysis report with visualizations and insights.

    Args:
        synthesis_results: Synthesized analysis results

    Returns:
        Structured analysis report
    """

    report = {
        "report_status": "success",
        "report_timestamp": synthesis_results.get("synthesis_timestamp"),
        "executive_summary": create_executive_summary(synthesis_results),
        "analysis_overview": {
            "analyses_performed": synthesis_results["analysis_summary"]["total_analyses"],
            "success_rate": f"{synthesis_results['analysis_summary']['success_rate']:.1%}",
            "data_quality": "Good" if synthesis_results["analysis_summary"]["success_rate"] > 0.7 else "Fair"
        },
        "key_findings": synthesis_results["key_findings"],
        "insights": synthesis_results["key_insights"],
        "recommendations": synthesis_results["recommendations"],
        "methodology": "Mixed-method analysis combining descriptive, comparative, and pattern detection techniques",
        "limitations": identify_analysis_limitations(synthesis_results),
        "next_steps": synthesis_results["recommendations"][:3] if synthesis_results["recommendations"] else []
    }

    return report


def create_executive_summary(synthesis_results: Dict[str, Any]) -> str:
    """Create executive summary from synthesis results."""

    summary = synthesis_results["analysis_summary"]
    insights_count = len(synthesis_results["key_insights"])
    recommendations_count = len(synthesis_results["recommendations"])

    exec_summary = f"""Analysis completed with {summary['successful_analyses']} out of {summary['total_analyses']}
analyses successful ({summary['success_rate']:.1%} success rate). Generated {insights_count} key insights
and {recommendations_count} actionable recommendations. """

    if synthesis_results["key_insights"]:
        exec_summary += f"Primary insight: {synthesis_results['key_insights'][0]}"

    return exec_summary.strip()


def identify_analysis_limitations(synthesis_results: Dict[str, Any]) -> List[str]:
    """Identify limitations in the analysis."""

    limitations = []

    if synthesis_results["analysis_summary"]["success_rate"] < 1.0:
        limitations.append("Some analysis components failed to complete successfully")

    if len(synthesis_results["key_findings"]) < 2:
        limitations.append("Limited analysis depth due to insufficient successful analyses")

    limitations.append("Analysis based on simulated data - real data validation needed")
    limitations.append("Pattern detection limited to basic correlation and clustering methods")

    return limitations


# Main Pipeline Creation Function

def create_analysis_pipeline() -> Pipeline:
    """
    Create the complete analysis workflow pipeline.

    Returns:
        Configured analysis pipeline with mixed orchestration
    """

    return Pipeline([
        # Step 1: Prepare analysis parameters
        PythonTask(
            function=prepare_analysis_parameters,
            name="analysis_preparation",
            kwargs={"workflow_context": pickled("workflow_context")},
            returns=[pickled("analysis_parameters")]
        ),

        # Step 2: Collect data from sources
        PythonTask(
            function=collect_analysis_data,
            name="data_collection",
            kwargs={"analysis_parameters": pickled("analysis_parameters")},
            returns=[pickled("data_collection")]
        ),

        # Step 3: Run parallel analysis tools
        Parallel(
            name="parallel_analysis",
            steps=[
                PythonTask(
                    function=run_descriptive_analysis,
                    name="descriptive_analysis",
                    kwargs={"data_collection": pickled("data_collection")},
                    returns=[pickled("descriptive_result")]
                ),
                PythonTask(
                    function=run_comparative_analysis,
                    name="comparative_analysis",
                    kwargs={"data_collection": pickled("data_collection")},
                    returns=[pickled("comparative_result")]
                ),
                PythonTask(
                    function=run_pattern_detection,
                    name="pattern_detection",
                    kwargs={"data_collection": pickled("data_collection")},
                    returns=[pickled("pattern_result")]
                )
            ],
            returns=[pickled("analysis_results")]
        ),

        # Step 4: Synthesize analysis results
        PythonTask(
            function=synthesize_analysis_results,
            name="results_synthesis",
            kwargs={"analysis_results": pickled("analysis_results")},
            returns=[pickled("synthesis_results")]
        ),

        # Step 5: Generate final report
        PythonTask(
            function=generate_analysis_report,
            name="report_generation",
            kwargs={"synthesis_results": pickled("synthesis_results")},
            returns=[pickled("analysis_report")]
        )
    ])


# Example usage and testing
if __name__ == "__main__":

    # Example workflow context
    example_context = {
        "execution_context": {
            "workflow_name": "analysis_workflow",
            "original_query": "analyze our website performance metrics",
            "llm_confidence": 0.78
        },
        "workflow_parameters": {
            "analysis_type": "comparative",
            "comparison_mode": True,
            "metrics_focus": ["performance", "efficiency"],
            "output_format": "structured_report"
        }
    }

    print("📊 Analysis Workflow - Test Components")
    print("=" * 50)

    # Test analysis preparation
    print("\n1. Testing Analysis Preparation:")
    analysis_params = prepare_analysis_parameters(example_context)
    print(f"   Analysis Type: {analysis_params['analysis_config']['analysis_type']}")
    print(f"   Data Sources: {len(analysis_params['data_sources'])}")
    print(f"   Analysis Tools: {len(analysis_params['analysis_tools'])}")

    # Test data collection
    print("\n2. Testing Data Collection:")
    data_collection = collect_analysis_data(analysis_params)
    print(f"   Sources Attempted: {data_collection['collection_metadata']['sources_attempted']}")
    print(f"   Sources Successful: {data_collection['collection_metadata']['sources_successful']}")

    # Test individual analysis function
    print("\n3. Testing Descriptive Analysis:")
    desc_result = run_descriptive_analysis(data_collection)
    print(f"   Analysis Status: {desc_result['analysis_status']}")
    print(f"   Datasets Analyzed: {desc_result['datasets_analyzed']}")

    print("\n✅ Analysis workflow components ready for pipeline execution")

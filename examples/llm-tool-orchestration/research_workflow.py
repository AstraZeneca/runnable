"""
Research Workflow Template

This workflow template implements a comprehensive research pipeline:
1. Web Search - Find relevant sources using search APIs
2. Content Extraction - Extract content from found sources
3. Content Summarization - Create structured summaries
4. Research Synthesis - Combine and synthesize findings

Demonstrates sequential execution with Map nodes for parallel processing
of multiple search results.
"""

from typing import Dict, List, Any
from runnable import Pipeline, PythonTask, Map, pickled
try:
    from .api_tools import web_search_tool, content_extractor_tool, summarizer_tool
except ImportError:
    # Handle direct execution
    from api_tools import web_search_tool, content_extractor_tool, summarizer_tool


def prepare_research_query(workflow_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare and optimize the research query based on workflow parameters.

    Args:
        workflow_context: Context from workflow router

    Returns:
        Optimized search parameters
    """

    execution_context = workflow_context.get("execution_context", {})
    workflow_params = workflow_context.get("workflow_parameters", {})

    # Extract key parameters
    original_query = execution_context.get("original_query", "")
    search_depth = workflow_params.get("search_depth", "medium")
    source_types = workflow_params.get("source_types", ["web"])
    max_results = workflow_params.get("max_results", 5)

    # Optimize query based on search depth
    optimized_query = optimize_search_query(original_query, search_depth)

    return {
        "search_query": optimized_query,
        "original_query": original_query,
        "search_depth": search_depth,
        "source_types": source_types,
        "max_results": max_results,
        "search_params": {
            "query": optimized_query,
            "max_results": max_results,
            "search_type": source_types[0] if source_types else "web"
        }
    }


def optimize_search_query(query: str, depth: str) -> str:
    """
    Optimize search query based on desired depth.

    Args:
        query: Original search query
        depth: Search depth (shallow, medium, deep)

    Returns:
        Optimized search query
    """

    query = query.strip()

    if depth == "deep":
        # Add terms for comprehensive results
        if "tutorial" not in query.lower():
            query += " tutorial guide comprehensive"
        if "example" not in query.lower():
            query += " examples"
    elif depth == "shallow":
        # Add terms for quick overviews
        query += " overview summary"
    else:  # medium
        # Add basic enhancement terms
        query += " guide"

    return query


def execute_web_search(search_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute web search using the prepared parameters.

    Args:
        search_parameters: Search configuration from query preparation

    Returns:
        Search results with metadata
    """

    search_params = search_parameters["search_params"]

    # Execute search using API tool
    search_result = web_search_tool(
        query=search_params["query"],
        max_results=search_params["max_results"],
        search_type=search_params["search_type"]
    )

    if search_result["status"] == "success":
        search_data = search_result["data"]

        return {
            "search_status": "success",
            "search_query": search_params["query"],
            "results_found": search_data["total_results"],
            "search_results": search_data["results"],
            "search_metadata": search_result["metadata"],
            "original_parameters": search_parameters
        }
    else:
        return {
            "search_status": "failed",
            "search_query": search_params["query"],
            "error": search_result.get("error", "Unknown search error"),
            "results_found": 0,
            "search_results": [],
            "original_parameters": search_parameters
        }


def extract_content_from_results(search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract content from search results. This prepares individual extraction tasks.

    Args:
        search_results: Results from web search step

    Returns:
        List of extraction tasks (for Map node processing)
    """

    extraction_tasks = []

    if search_results["search_status"] == "success":
        for idx, result in enumerate(search_results["search_results"]):
            task = {
                "task_id": f"extract_{idx}",
                "url": result["url"],
                "title": result["title"],
                "snippet": result["snippet"],
                "relevance_score": result.get("relevance_score", 0.5),
                "extraction_type": "structured"  # Extract structured content
            }
            extraction_tasks.append(task)

    return extraction_tasks


def process_single_extraction(extraction_task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single content extraction task.

    This function will be used within a Map node to process multiple extractions in parallel.

    Args:
        extraction_task: Single extraction task definition

    Returns:
        Extraction result with content and metadata
    """

    # Execute content extraction
    extraction_result = content_extractor_tool(
        url=extraction_task["url"],
        extract_type=extraction_task["extraction_type"]
    )

    if extraction_result["status"] == "success":
        extracted_data = extraction_result["data"]

        return {
            "extraction_status": "success",
            "task_id": extraction_task["task_id"],
            "url": extraction_task["url"],
            "title": extraction_task["title"],
            "extracted_content": extracted_data["extracted_content"],
            "word_count": extracted_data["word_count"],
            "relevance_score": extraction_task["relevance_score"],
            "extraction_metadata": extraction_result["metadata"]
        }
    else:
        return {
            "extraction_status": "failed",
            "task_id": extraction_task["task_id"],
            "url": extraction_task["url"],
            "title": extraction_task["title"],
            "error": extraction_result.get("error", "Extraction failed"),
            "extracted_content": None,
            "word_count": 0,
            "relevance_score": extraction_task["relevance_score"]
        }


def aggregate_extracted_content(extraction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate all extraction results into a structured dataset.

    Args:
        extraction_results: List of extraction results from Map processing

    Returns:
        Aggregated content ready for summarization
    """

    successful_extractions = []
    failed_extractions = []
    total_words = 0

    for result in extraction_results:
        if result["extraction_status"] == "success":
            successful_extractions.append(result)
            total_words += result["word_count"]
        else:
            failed_extractions.append(result)

    # Prepare aggregated content
    aggregated_content = {
        "total_sources": len(extraction_results),
        "successful_extractions": len(successful_extractions),
        "failed_extractions": len(failed_extractions),
        "total_word_count": total_words,
        "average_relevance": sum(r["relevance_score"] for r in successful_extractions) / len(successful_extractions) if successful_extractions else 0,
        "extracted_sources": successful_extractions,
        "failed_sources": failed_extractions,
        "aggregation_metadata": {
            "success_rate": len(successful_extractions) / len(extraction_results) if extraction_results else 0,
            "quality_score": calculate_content_quality(successful_extractions)
        }
    }

    return aggregated_content


def calculate_content_quality(extractions: List[Dict[str, Any]]) -> float:
    """Calculate overall content quality score."""

    if not extractions:
        return 0.0

    # Simple quality scoring based on word count and relevance
    quality_factors = []

    for extraction in extractions:
        word_count_factor = min(extraction["word_count"] / 200.0, 1.0)  # Normalize to 200 words
        relevance_factor = extraction["relevance_score"]

        quality_factors.append((word_count_factor + relevance_factor) / 2)

    return sum(quality_factors) / len(quality_factors)


def create_comprehensive_summary(aggregated_content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comprehensive summary from all extracted content.

    Args:
        aggregated_content: Aggregated extraction results

    Returns:
        Final research summary and synthesis
    """

    if not aggregated_content["extracted_sources"]:
        return {
            "summary_status": "failed",
            "reason": "No content available for summarization",
            "research_summary": "",
            "key_findings": [],
            "source_summary": "No sources processed successfully"
        }

    # Combine all extracted text for summarization
    combined_text = ""
    source_info = []

    for source in aggregated_content["extracted_sources"]:
        content = source["extracted_content"]
        if isinstance(content, dict) and "text" in content:
            combined_text += f"\n\nSource: {source['title']}\n{content['text']}"
            source_info.append({
                "title": source["title"],
                "url": source["url"],
                "word_count": source["word_count"],
                "relevance": source["relevance_score"]
            })

    # Generate summary using summarizer tool
    summary_result = summarizer_tool(
        content=combined_text,
        length="long",  # Comprehensive summary
        style="neutral"
    )

    if summary_result["status"] == "success":
        summary_data = summary_result["data"]

        return {
            "summary_status": "success",
            "research_summary": summary_data["summary"],
            "summary_statistics": {
                "original_word_count": summary_data["original_length"],
                "summary_word_count": summary_data["summary_length"],
                "compression_ratio": summary_data["compression_ratio"],
                "sources_analyzed": len(source_info)
            },
            "key_findings": extract_key_findings(summary_data["summary"]),
            "source_summary": {
                "total_sources": len(source_info),
                "sources": source_info,
                "average_relevance": aggregated_content["average_relevance"],
                "content_quality": aggregated_content["aggregation_metadata"]["quality_score"]
            },
            "research_metadata": {
                "extraction_success_rate": aggregated_content["aggregation_metadata"]["success_rate"],
                "total_word_count": aggregated_content["total_word_count"],
                "summary_confidence": 0.8  # Mock confidence score
            }
        }
    else:
        return {
            "summary_status": "failed",
            "reason": summary_result.get("error", "Summarization failed"),
            "research_summary": "",
            "source_summary": {
                "total_sources": len(source_info),
                "sources": source_info
            }
        }


def extract_key_findings(summary_text: str) -> List[str]:
    """
    Extract key findings from the summary text.

    Args:
        summary_text: Generated summary text

    Returns:
        List of key findings
    """

    # Simple key finding extraction (in real implementation, this might use NLP)
    sentences = summary_text.split('. ')

    # Select important sentences (simplified logic)
    key_findings = []

    for sentence in sentences[:5]:  # Take first 5 sentences as key findings
        if len(sentence.strip()) > 20:  # Ignore very short sentences
            key_findings.append(sentence.strip())

    return key_findings


# Main Pipeline Creation Function

def create_research_pipeline() -> Pipeline:
    """
    Create the complete research workflow pipeline.

    Returns:
        Configured research pipeline with sequential and parallel processing
    """

    return Pipeline([
        # Step 1: Prepare research query
        PythonTask(
            function=prepare_research_query,
            name="research_query_preparation",
            kwargs={"workflow_context": pickled("workflow_context")},
            returns=[pickled("search_parameters")]
        ),

        # Step 2: Execute web search
        PythonTask(
            function=execute_web_search,
            name="web_search_execution",
            kwargs={"search_parameters": pickled("search_parameters")},
            returns=[pickled("search_results")]
        ),

        # Step 3: Prepare extraction tasks
        PythonTask(
            function=extract_content_from_results,
            name="extraction_task_preparation",
            kwargs={"search_results": pickled("search_results")},
            returns=[pickled("extraction_tasks")]
        ),

        # Step 4: Process content extractions in parallel (Map node)
        Map(
            name="parallel_content_extraction",
            iterate_on=pickled("extraction_tasks"),
            iterate_as="extraction_task",
            branch=Pipeline([
                PythonTask(
                    function=process_single_extraction,
                    name="single_extraction",
                    kwargs={"extraction_task": pickled("extraction_task")},
                    returns=[pickled("extraction_result")]
                )
            ]),
            returns=[pickled("extraction_results")]
        ),

        # Step 5: Aggregate extraction results
        PythonTask(
            function=aggregate_extracted_content,
            name="content_aggregation",
            kwargs={"extraction_results": pickled("extraction_results")},
            returns=[pickled("aggregated_content")]
        ),

        # Step 6: Create comprehensive summary
        PythonTask(
            function=create_comprehensive_summary,
            name="research_summarization",
            kwargs={"aggregated_content": pickled("aggregated_content")},
            returns=[pickled("research_summary")]
        )
    ])


# Example usage and testing
if __name__ == "__main__":

    # Example workflow context (would come from workflow router)
    example_context = {
        "execution_context": {
            "workflow_name": "research_workflow",
            "original_query": "machine learning best practices",
            "llm_confidence": 0.85
        },
        "workflow_parameters": {
            "query": "machine learning best practices",
            "search_depth": "deep",
            "source_types": ["web"],
            "max_results": 4
        }
    }

    print("🔬 Research Workflow - Test Components")
    print("=" * 50)

    # Test query preparation
    print("\n1. Testing Query Preparation:")
    search_params = prepare_research_query(example_context)
    print(f"   Optimized Query: '{search_params['search_query']}'")
    print(f"   Max Results: {search_params['max_results']}")
    print(f"   Search Depth: {search_params['search_depth']}")

    # Test web search
    print("\n2. Testing Web Search:")
    search_results = execute_web_search(search_params)
    print(f"   Search Status: {search_results['search_status']}")
    print(f"   Results Found: {search_results['results_found']}")

    # Test extraction task preparation
    print("\n3. Testing Extraction Task Preparation:")
    extraction_tasks = extract_content_from_results(search_results)
    print(f"   Extraction Tasks: {len(extraction_tasks)}")
    if extraction_tasks:
        print(f"   First Task URL: {extraction_tasks[0]['url']}")

    # Test single extraction
    if extraction_tasks:
        print("\n4. Testing Single Extraction:")
        extraction_result = process_single_extraction(extraction_tasks[0])
        print(f"   Extraction Status: {extraction_result['extraction_status']}")
        print(f"   Word Count: {extraction_result['word_count']}")

    print("\n✅ Research workflow components ready for pipeline execution")

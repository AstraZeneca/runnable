"""
API Tools Library

This module provides generic wrappers for API-based tools that can be used
within workflow templates. Each tool follows a consistent interface pattern
for easy integration with runnable's PythonTask system.

All tools return standardized response formats for consistent data flow
between pipeline stages.
"""

import json
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import random
from urllib.parse import quote_plus


# Standardized response format for all API tools
def create_tool_response(
    status: str,
    data: Any,
    tool_name: str,
    execution_time: float,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized tool response format.

    Args:
        status: "success", "error", or "partial"
        data: Tool-specific result data
        tool_name: Name of the tool that generated this response
        execution_time: Time taken to execute the tool (seconds)
        error: Error message if status is "error"
        metadata: Additional tool-specific metadata

    Returns:
        Standardized response dictionary
    """
    return {
        "status": status,
        "data": data,
        "metadata": {
            "tool_name": tool_name,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "api_cost": metadata.get("api_cost", 0.0) if metadata else 0.0,
            **(metadata or {})
        },
        "error": error
    }


# Web Search Tool
def web_search_tool(
    query: str,
    max_results: int = 5,
    search_type: str = "web"
) -> Dict[str, Any]:
    """
    Simulate web search API calls.

    In a real implementation, this would call actual search APIs like:
    - Google Custom Search API
    - Bing Web Search API
    - DuckDuckGo API
    - etc.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        search_type: Type of search ("web", "academic", "news", "images")

    Returns:
        Standardized tool response with search results
    """
    start_time = time.time()

    try:
        # Simulate API call delay
        time.sleep(0.5)

        # Mock search results
        mock_results = generate_mock_search_results(query, max_results, search_type)

        execution_time = time.time() - start_time

        return create_tool_response(
            status="success",
            data={
                "query": query,
                "results": mock_results,
                "total_results": len(mock_results),
                "search_type": search_type
            },
            tool_name="web_search",
            execution_time=execution_time,
            metadata={
                "api_cost": 0.001 * max_results,
                "search_engine": "mock_search_api",
                "results_filtered": False
            }
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return create_tool_response(
            status="error",
            data=None,
            tool_name="web_search",
            execution_time=execution_time,
            error=f"Search API error: {str(e)}"
        )


def generate_mock_search_results(query: str, max_results: int, search_type: str) -> List[Dict[str, Any]]:
    """Generate mock search results for demonstration."""

    # Base result templates
    base_results = [
        {
            "title": f"Complete Guide to {query.title()}",
            "url": f"https://example.com/guide/{quote_plus(query)}",
            "snippet": f"Comprehensive overview of {query} including best practices, examples, and detailed explanations.",
            "relevance_score": 0.95
        },
        {
            "title": f"{query.title()} - Wikipedia",
            "url": f"https://en.wikipedia.org/wiki/{quote_plus(query)}",
            "snippet": f"Wikipedia article providing encyclopedic information about {query} with references and citations.",
            "relevance_score": 0.88
        },
        {
            "title": f"Latest {query.title()} News and Updates",
            "url": f"https://news.example.com/{quote_plus(query)}",
            "snippet": f"Recent developments and news related to {query} from trusted sources.",
            "relevance_score": 0.82
        },
        {
            "title": f"How to Get Started with {query.title()}",
            "url": f"https://tutorial.example.com/{quote_plus(query)}",
            "snippet": f"Beginner-friendly tutorial for learning {query} step by step.",
            "relevance_score": 0.79
        },
        {
            "title": f"{query.title()} Best Practices and Tips",
            "url": f"https://tips.example.com/{quote_plus(query)}",
            "snippet": f"Expert tips and best practices for working with {query} effectively.",
            "relevance_score": 0.75
        }
    ]

    # Adjust based on search type
    if search_type == "academic":
        for result in base_results:
            result["url"] = result["url"].replace("example.com", "academic.edu")
            result["snippet"] += " Peer-reviewed research and academic analysis."

    elif search_type == "news":
        for result in base_results:
            result["url"] = result["url"].replace("example.com", "news-source.com")
            result["snippet"] = f"Breaking: Recent developments in {query}. " + result["snippet"]

    # Return requested number of results
    return base_results[:max_results]


# Content Extraction Tool
def content_extractor_tool(url: str, extract_type: str = "text") -> Dict[str, Any]:
    """
    Extract content from web pages.

    In a real implementation, this would use libraries like:
    - BeautifulSoup for HTML parsing
    - newspaper3k for article extraction
    - PyPDF2 for PDF content
    - etc.

    Args:
        url: URL to extract content from
        extract_type: Type of content to extract ("text", "structured", "metadata")

    Returns:
        Standardized tool response with extracted content
    """
    start_time = time.time()

    try:
        # Simulate extraction delay
        time.sleep(0.8)

        # Mock content extraction
        mock_content = generate_mock_extracted_content(url, extract_type)

        execution_time = time.time() - start_time

        return create_tool_response(
            status="success",
            data={
                "url": url,
                "extracted_content": mock_content,
                "extraction_type": extract_type,
                "word_count": len(mock_content.get("text", "").split()) if mock_content.get("text") else 0
            },
            tool_name="content_extractor",
            execution_time=execution_time,
            metadata={
                "api_cost": 0.005,
                "extraction_method": "mock_extractor",
                "content_language": "en"
            }
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return create_tool_response(
            status="error",
            data=None,
            tool_name="content_extractor",
            execution_time=execution_time,
            error=f"Content extraction error: {str(e)}"
        )


def generate_mock_extracted_content(url: str, extract_type: str) -> Dict[str, Any]:
    """Generate mock extracted content."""

    base_content = {
        "text": f"""
This is extracted content from {url}. The article provides comprehensive information
about the topic, including detailed explanations, examples, and practical applications.

Key points covered in this content:
- Overview and introduction to the subject
- Technical details and implementation aspects
- Best practices and recommendations
- Real-world examples and case studies
- Future trends and developments

The content is well-structured and provides valuable insights for readers
interested in learning more about this topic.
        """.strip(),

        "title": f"Article Title from {url}",
        "author": "Content Author",
        "publish_date": "2024-01-15",
        "keywords": ["example", "content", "extraction", "article"]
    }

    if extract_type == "structured":
        base_content.update({
            "headings": [
                "Introduction",
                "Technical Details",
                "Best Practices",
                "Examples",
                "Conclusion"
            ],
            "paragraphs": 12,
            "links": 8,
            "images": 3
        })

    elif extract_type == "metadata":
        base_content = {
            "title": base_content["title"],
            "author": base_content["author"],
            "publish_date": base_content["publish_date"],
            "description": "Meta description of the article content",
            "tags": base_content["keywords"],
            "word_count": 245,
            "reading_time": "2 minutes",
            "content_type": "article"
        }

    return base_content


# Text Summarization Tool
def summarizer_tool(
    content: str,
    length: str = "medium",
    style: str = "neutral"
) -> Dict[str, Any]:
    """
    Summarize text content.

    In a real implementation, this would call:
    - OpenAI GPT API for summarization
    - Anthropic Claude API
    - Hugging Face Transformers
    - Custom summarization models

    Args:
        content: Text content to summarize
        length: Summary length ("short", "medium", "long")
        style: Summary style ("neutral", "bullet_points", "technical")

    Returns:
        Standardized tool response with summary
    """
    start_time = time.time()

    try:
        # Simulate summarization processing
        time.sleep(1.0)

        # Mock summarization
        mock_summary = generate_mock_summary(content, length, style)

        execution_time = time.time() - start_time

        return create_tool_response(
            status="success",
            data={
                "original_length": len(content),
                "summary": mock_summary,
                "summary_length": len(mock_summary),
                "compression_ratio": len(mock_summary) / len(content) if content else 0,
                "style": style
            },
            tool_name="summarizer",
            execution_time=execution_time,
            metadata={
                "api_cost": 0.01,
                "model": "mock_summarizer_v1",
                "language": "en"
            }
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return create_tool_response(
            status="error",
            data=None,
            tool_name="summarizer",
            execution_time=execution_time,
            error=f"Summarization error: {str(e)}"
        )


def generate_mock_summary(content: str, length: str, style: str) -> str:
    """Generate mock summary based on parameters."""

    # Extract key concepts from content for more realistic summaries
    words = content.lower().split()
    key_concepts = [word for word in words if len(word) > 5][:5]
    concepts_str = ", ".join(key_concepts[:3]) if key_concepts else "key topics"

    if style == "bullet_points":
        if length == "short":
            return f"• Main focus on {concepts_str}\n• Provides practical insights\n• Includes relevant examples"
        elif length == "long":
            return f"""• Comprehensive coverage of {concepts_str}
• Detailed technical explanations and methodology
• Real-world applications and case studies
• Best practices and implementation guidelines
• Future trends and development directions
• Expert recommendations and insights"""
        else:  # medium
            return f"""• Primary focus on {concepts_str}
• Includes technical details and practical applications
• Provides examples and implementation guidance
• Covers best practices and recommendations"""

    elif style == "technical":
        base = f"Technical analysis of {concepts_str} reveals comprehensive coverage of methodologies and implementations."
        if length == "short":
            return base
        elif length == "long":
            return f"{base} The content provides detailed technical specifications, implementation frameworks, performance metrics, and scalability considerations. Advanced topics include optimization strategies, integration patterns, and future technological developments."
        else:  # medium
            return f"{base} Content includes implementation details, performance considerations, and practical applications with supporting examples."

    else:  # neutral style
        base = f"The content provides an overview of {concepts_str} with practical insights."
        if length == "short":
            return base
        elif length == "long":
            return f"{base} It covers fundamental concepts, implementation approaches, real-world applications, and best practices. The material includes detailed explanations, supporting examples, case studies, and expert recommendations for effective implementation."
        else:  # medium
            return f"{base} It includes detailed explanations, practical examples, and implementation guidance for effective application."


# Data Analysis Tool
def data_analyzer_tool(
    data: Dict[str, Any],
    analysis_type: str = "descriptive",
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze structured data.

    In a real implementation, this would use:
    - Pandas for data analysis
    - NumPy for numerical computations
    - Scikit-learn for ML analysis
    - Plotly/Matplotlib for visualizations

    Args:
        data: Data to analyze (dict, list, or structured format)
        analysis_type: Type of analysis ("descriptive", "comparative", "predictive")
        metrics: Specific metrics to calculate

    Returns:
        Standardized tool response with analysis results
    """
    start_time = time.time()

    try:
        # Simulate analysis processing
        time.sleep(1.2)

        # Mock data analysis
        mock_analysis = generate_mock_analysis(data, analysis_type, metrics or [])

        execution_time = time.time() - start_time

        return create_tool_response(
            status="success",
            data={
                "analysis_type": analysis_type,
                "input_data_size": len(str(data)),
                "results": mock_analysis,
                "metrics_calculated": metrics or ["standard"],
                "insights": generate_mock_insights(mock_analysis)
            },
            tool_name="data_analyzer",
            execution_time=execution_time,
            metadata={
                "api_cost": 0.02,
                "analysis_engine": "mock_analytics_v1",
                "confidence_score": 0.85
            }
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return create_tool_response(
            status="error",
            data=None,
            tool_name="data_analyzer",
            execution_time=execution_time,
            error=f"Data analysis error: {str(e)}"
        )


def generate_mock_analysis(data: Dict[str, Any], analysis_type: str, metrics: List[str]) -> Dict[str, Any]:
    """Generate mock analysis results."""

    base_results = {
        "summary_statistics": {
            "total_records": random.randint(100, 1000),
            "data_quality_score": round(random.uniform(0.7, 0.95), 2),
            "completeness": round(random.uniform(0.8, 1.0), 2)
        }
    }

    if analysis_type == "descriptive":
        base_results.update({
            "descriptive_stats": {
                "mean": round(random.uniform(50, 150), 2),
                "median": round(random.uniform(45, 155), 2),
                "std_deviation": round(random.uniform(10, 30), 2),
                "min_value": round(random.uniform(0, 20), 2),
                "max_value": round(random.uniform(180, 250), 2)
            },
            "distributions": {
                "normal_distribution_fit": round(random.uniform(0.6, 0.9), 2),
                "skewness": round(random.uniform(-1, 1), 2)
            }
        })

    elif analysis_type == "comparative":
        base_results.update({
            "comparison_results": {
                "group_a_performance": round(random.uniform(70, 90), 2),
                "group_b_performance": round(random.uniform(65, 95), 2),
                "statistical_significance": "p < 0.05",
                "effect_size": round(random.uniform(0.2, 0.8), 2)
            }
        })

    elif analysis_type == "predictive":
        base_results.update({
            "predictions": {
                "forecast_accuracy": round(random.uniform(0.75, 0.95), 2),
                "confidence_intervals": "95%",
                "trend_direction": random.choice(["increasing", "decreasing", "stable"]),
                "seasonal_patterns": random.choice([True, False])
            }
        })

    return base_results


def generate_mock_insights(analysis_results: Dict[str, Any]) -> List[str]:
    """Generate mock insights from analysis results."""

    insights = [
        "Data shows consistent patterns across time periods",
        "Quality metrics indicate reliable data sources",
        "Statistical significance suggests meaningful relationships",
        "Trend analysis reveals potential optimization opportunities"
    ]

    # Add specific insights based on results
    if "descriptive_stats" in analysis_results:
        if analysis_results["descriptive_stats"]["std_deviation"] > 20:
            insights.append("High variability detected - recommend further segmentation")

    if "comparison_results" in analysis_results:
        perf_diff = abs(analysis_results["comparison_results"]["group_a_performance"] -
                       analysis_results["comparison_results"]["group_b_performance"])
        if perf_diff > 10:
            insights.append("Significant performance difference between groups identified")

    return insights[:4]  # Return top 4 insights


# Generic HTTP API Tool
def http_api_tool(
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Generic HTTP API call tool.

    This provides a flexible wrapper for calling any HTTP API.
    In production, this would include proper authentication, rate limiting,
    and error handling for various API patterns.

    Args:
        endpoint: API endpoint URL
        method: HTTP method (GET, POST, PUT, DELETE)
        params: Request parameters or body data
        headers: HTTP headers
        timeout: Request timeout in seconds

    Returns:
        Standardized tool response with API results
    """
    start_time = time.time()

    try:
        # Simulate API call
        time.sleep(0.6)

        # Mock API response
        mock_response = {
            "status_code": 200,
            "response_data": {
                "message": "API call successful",
                "endpoint": endpoint,
                "method": method,
                "timestamp": datetime.now().isoformat()
            },
            "headers": {
                "content-type": "application/json",
                "x-rate-limit-remaining": "99"
            }
        }

        execution_time = time.time() - start_time

        return create_tool_response(
            status="success",
            data=mock_response,
            tool_name="http_api",
            execution_time=execution_time,
            metadata={
                "api_cost": 0.003,
                "endpoint": endpoint,
                "method": method,
                "status_code": 200
            }
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return create_tool_response(
            status="error",
            data=None,
            tool_name="http_api",
            execution_time=execution_time,
            error=f"HTTP API error: {str(e)}"
        )


# Tool registry for easy access
AVAILABLE_TOOLS = {
    "web_search": web_search_tool,
    "content_extractor": content_extractor_tool,
    "summarizer": summarizer_tool,
    "data_analyzer": data_analyzer_tool,
    "http_api": http_api_tool
}


def get_tool(tool_name: str):
    """Get a tool function by name."""
    return AVAILABLE_TOOLS.get(tool_name)


def list_available_tools() -> List[str]:
    """List all available tool names."""
    return list(AVAILABLE_TOOLS.keys())


# Example usage and testing
if __name__ == "__main__":

    print("🔧 API Tools Library - Test Results")
    print("=" * 60)

    # Test web search
    print("\n1. Testing Web Search Tool:")
    result = web_search_tool("machine learning tutorials", max_results=3)
    print(f"   Status: {result['status']}")
    print(f"   Results: {len(result['data']['results'])} items")
    print(f"   Execution Time: {result['metadata']['execution_time']:.2f}s")

    # Test content extraction
    print("\n2. Testing Content Extractor Tool:")
    result = content_extractor_tool("https://example.com/article", extract_type="structured")
    print(f"   Status: {result['status']}")
    print(f"   Word Count: {result['data']['word_count']}")
    print(f"   Execution Time: {result['metadata']['execution_time']:.2f}s")

    # Test summarization
    print("\n3. Testing Summarizer Tool:")
    sample_text = "This is a long article about machine learning and artificial intelligence with many technical details and examples."
    result = summarizer_tool(sample_text, length="short", style="bullet_points")
    print(f"   Status: {result['status']}")
    print(f"   Compression Ratio: {result['data']['compression_ratio']:.2f}")
    print(f"   Summary Preview: {result['data']['summary'][:100]}...")

    # Test data analyzer
    print("\n4. Testing Data Analyzer Tool:")
    sample_data = {"values": [1, 2, 3, 4, 5], "categories": ["A", "B", "C"]}
    result = data_analyzer_tool(sample_data, analysis_type="descriptive")
    print(f"   Status: {result['status']}")
    print(f"   Insights: {len(result['data']['insights'])} generated")
    print(f"   Execution Time: {result['metadata']['execution_time']:.2f}s")

    print(f"\n📋 Available Tools: {', '.join(list_available_tools())}")

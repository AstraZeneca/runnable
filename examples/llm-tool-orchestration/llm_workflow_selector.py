"""
LLM Workflow Selector

This module contains the LLM agent logic for analyzing user input and selecting
the most appropriate workflow template from available options.

The LLM agent makes a single upfront decision about which workflow to execute,
then returns the workflow name and parameters for downstream processing.
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime


def llm_workflow_selector(user_query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    LLM agent that analyzes user input and selects appropriate workflow template.

    In a real implementation, this would call an actual LLM API (OpenAI, Anthropic, etc.).
    For demonstration purposes, this uses rule-based logic to simulate LLM decision making.

    Args:
        user_query: The user's input/request that needs processing
        context: Optional context information (user preferences, history, etc.)

    Returns:
        Dict containing:
        - selected_workflow: Name of the workflow template to execute
        - workflow_parameters: Parameters to pass to the selected workflow
        - reasoning: Explanation of why this workflow was selected
        - confidence: Confidence score (0.0 to 1.0) in the selection
    """

    if context is None:
        context = {}

    # Normalize the query for analysis
    query_lower = user_query.lower().strip()

    # Define workflow selection logic (simulates LLM reasoning)
    workflow_selection = analyze_user_intent(query_lower, context)

    # Add metadata and return structured response
    result = {
        "selected_workflow": workflow_selection["workflow"],
        "workflow_parameters": workflow_selection["parameters"],
        "reasoning": workflow_selection["reasoning"],
        "confidence": workflow_selection["confidence"],
        "analysis_timestamp": datetime.now().isoformat(),
        "original_query": user_query,
        "processed_query": workflow_selection["parameters"].get("query", user_query)
    }

    return result


def analyze_user_intent(query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze user intent and map to appropriate workflow.

    This simulates LLM analysis by using pattern matching and keyword detection.
    In a real implementation, this would be replaced with actual LLM API calls.
    """

    # Research workflow indicators
    research_patterns = [
        r"research|find|search|learn about|tell me about|explain|what is|how does",
        r"information|details|background|overview|summary",
        r"papers|articles|studies|sources|references"
    ]

    # Analysis workflow indicators
    analysis_patterns = [
        r"analyze|compare|evaluate|assess|examine",
        r"data|statistics|metrics|trends|patterns",
        r"performance|effectiveness|results|outcomes"
    ]

    # Content creation workflow indicators
    content_patterns = [
        r"create|write|generate|produce|draft|compose",
        r"article|blog|report|document|content|story",
        r"format|structure|outline|template"
    ]

    # Problem solving workflow indicators
    problem_patterns = [
        r"solve|fix|resolve|help with|problem|issue|challenge",
        r"solution|approach|strategy|method|way to",
        r"troubleshoot|debug|optimize|improve"
    ]

    # Check patterns and calculate match scores
    workflows = {
        "research_workflow": calculate_pattern_score(query, research_patterns),
        "analysis_workflow": calculate_pattern_score(query, analysis_patterns),
        "content_creation_workflow": calculate_pattern_score(query, content_patterns),
        "problem_solving_workflow": calculate_pattern_score(query, problem_patterns)
    }

    # Select workflow with highest score
    selected_workflow = max(workflows, key=workflows.get)
    confidence = workflows[selected_workflow]

    # If confidence is too low, default to research workflow
    if confidence < 0.3:
        selected_workflow = "research_workflow"
        confidence = 0.5

    # Generate workflow-specific parameters
    parameters = generate_workflow_parameters(selected_workflow, query, context)
    reasoning = generate_reasoning(selected_workflow, query, confidence, workflows)

    return {
        "workflow": selected_workflow,
        "parameters": parameters,
        "reasoning": reasoning,
        "confidence": min(confidence, 1.0)
    }


def calculate_pattern_score(query: str, patterns: List[str]) -> float:
    """Calculate how well the query matches a set of patterns."""
    total_score = 0.0

    for pattern in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            # Weight score based on pattern strength and query length
            match_strength = len(re.findall(pattern, query, re.IGNORECASE))
            total_score += match_strength * 0.3

    # Normalize by number of patterns to get average score
    return min(total_score / len(patterns) if patterns else 0.0, 1.0)


def generate_workflow_parameters(workflow: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate workflow-specific parameters based on the query and context."""

    base_params = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "user_context": context
    }

    if workflow == "research_workflow":
        return {
            **base_params,
            "search_depth": determine_search_depth(query),
            "source_types": ["web", "academic"] if "academic" in query else ["web"],
            "max_results": 10,
            "summarize": True
        }

    elif workflow == "analysis_workflow":
        return {
            **base_params,
            "analysis_type": determine_analysis_type(query),
            "comparison_mode": "compare" in query.lower(),
            "metrics_focus": extract_metrics_keywords(query),
            "output_format": "structured_report"
        }

    elif workflow == "content_creation_workflow":
        return {
            **base_params,
            "content_type": determine_content_type(query),
            "tone": determine_tone(query),
            "length": determine_content_length(query),
            "include_research": "research" in query.lower()
        }

    elif workflow == "problem_solving_workflow":
        return {
            **base_params,
            "problem_domain": extract_domain(query),
            "solution_approach": "systematic",
            "include_alternatives": True,
            "priority": determine_priority(query)
        }

    else:
        return base_params


def determine_search_depth(query: str) -> str:
    """Determine appropriate search depth based on query complexity."""
    if any(word in query.lower() for word in ["comprehensive", "detailed", "thorough", "complete"]):
        return "deep"
    elif any(word in query.lower() for word in ["quick", "brief", "summary", "overview"]):
        return "shallow"
    else:
        return "medium"


def determine_analysis_type(query: str) -> str:
    """Determine the type of analysis requested."""
    if "compare" in query.lower():
        return "comparative"
    elif any(word in query.lower() for word in ["trend", "time", "temporal"]):
        return "temporal"
    elif any(word in query.lower() for word in ["performance", "metrics"]):
        return "performance"
    else:
        return "exploratory"


def extract_metrics_keywords(query: str) -> List[str]:
    """Extract metric-related keywords from the query."""
    metrics_keywords = []
    metrics_terms = ["revenue", "cost", "efficiency", "performance", "speed", "accuracy", "growth"]

    for term in metrics_terms:
        if term in query.lower():
            metrics_keywords.append(term)

    return metrics_keywords if metrics_keywords else ["general"]


def determine_content_type(query: str) -> str:
    """Determine the type of content to create."""
    content_types = {
        "article": ["article", "blog", "post"],
        "report": ["report", "analysis", "study"],
        "document": ["document", "guide", "manual"],
        "creative": ["story", "creative", "narrative"]
    }

    query_lower = query.lower()
    for content_type, keywords in content_types.items():
        if any(keyword in query_lower for keyword in keywords):
            return content_type

    return "general"


def determine_tone(query: str) -> str:
    """Determine appropriate tone for content."""
    if any(word in query.lower() for word in ["formal", "professional", "business"]):
        return "formal"
    elif any(word in query.lower() for word in ["casual", "friendly", "conversational"]):
        return "casual"
    elif any(word in query.lower() for word in ["technical", "academic", "scientific"]):
        return "technical"
    else:
        return "neutral"


def determine_content_length(query: str) -> str:
    """Determine appropriate content length."""
    if any(word in query.lower() for word in ["long", "detailed", "comprehensive", "thorough"]):
        return "long"
    elif any(word in query.lower() for word in ["short", "brief", "summary", "concise"]):
        return "short"
    else:
        return "medium"


def extract_domain(query: str) -> str:
    """Extract the problem domain from the query."""
    domains = {
        "technical": ["code", "programming", "software", "system", "database", "api"],
        "business": ["business", "strategy", "marketing", "sales", "finance", "management"],
        "personal": ["personal", "life", "career", "health", "relationship"],
        "academic": ["research", "study", "academic", "education", "learning"]
    }

    query_lower = query.lower()
    for domain, keywords in domains.items():
        if any(keyword in query_lower for keyword in keywords):
            return domain

    return "general"


def determine_priority(query: str) -> str:
    """Determine problem priority level."""
    if any(word in query.lower() for word in ["urgent", "critical", "emergency", "asap"]):
        return "high"
    elif any(word in query.lower() for word in ["low", "when possible", "eventually"]):
        return "low"
    else:
        return "medium"


def generate_reasoning(workflow: str, query: str, confidence: float, all_scores: Dict[str, float]) -> str:
    """Generate human-readable reasoning for workflow selection."""

    workflow_descriptions = {
        "research_workflow": "comprehensive research and information gathering",
        "analysis_workflow": "data analysis and comparative evaluation",
        "content_creation_workflow": "content creation and writing",
        "problem_solving_workflow": "systematic problem-solving approach"
    }

    selected_desc = workflow_descriptions.get(workflow, workflow)

    # Find second-best option for context
    sorted_workflows = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    second_best = sorted_workflows[1] if len(sorted_workflows) > 1 else ("none", 0)

    reasoning = f"Selected '{workflow}' for {selected_desc} based on query analysis. "
    reasoning += f"Confidence: {confidence:.1%}. "

    if confidence > 0.7:
        reasoning += "High confidence due to clear intent indicators in the query."
    elif confidence > 0.5:
        reasoning += f"Moderate confidence. Alternative consideration was '{second_best[0]}' (score: {second_best[1]:.1%})."
    else:
        reasoning += "Lower confidence - query could match multiple workflows. Using most general approach."

    return reasoning


# Example usage and testing
if __name__ == "__main__":

    # Test examples
    test_queries = [
        "Research the latest developments in machine learning",
        "Analyze the performance metrics of our Q3 sales",
        "Create a blog post about sustainable energy",
        "Help me solve this database connection problem",
        "Find information about Python web frameworks",
        "Compare the efficiency of different sorting algorithms"
    ]

    print("🤖 LLM Workflow Selector - Test Results")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        result = llm_workflow_selector(query)

        print(f"   Selected: {result['selected_workflow']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Reasoning: {result['reasoning']}")
        print(f"   Key Parameters: {list(result['workflow_parameters'].keys())}")

"""
Content Creation Workflow Template

This workflow template implements a comprehensive content creation pipeline:
1. Research Phase - Gather information and sources
2. Content Generation - Create initial content draft
3. Quality Assessment - Evaluate content quality
4. Content Refinement - Improve content based on quality checks (conditional)
5. Final Formatting - Format content for publication

Demonstrates sequential execution with conditional branching for quality control loops.
"""

from typing import Dict, List, Any
from runnable import Pipeline, PythonTask, Conditional, pickled
try:
    from .api_tools import web_search_tool, content_extractor_tool, summarizer_tool
except ImportError:
    # Handle direct execution
    from api_tools import web_search_tool, content_extractor_tool, summarizer_tool


def prepare_content_parameters(workflow_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare content creation parameters based on workflow context.

    Args:
        workflow_context: Context from workflow router

    Returns:
        Content creation configuration and parameters
    """

    execution_context = workflow_context.get("execution_context", {})
    workflow_params = workflow_context.get("workflow_parameters", {})

    # Extract content parameters
    original_query = execution_context.get("original_query", "")
    content_type = workflow_params.get("content_type", "general")
    tone = workflow_params.get("tone", "neutral")
    length = workflow_params.get("length", "medium")
    include_research = workflow_params.get("include_research", True)

    return {
        "content_config": {
            "topic": original_query,
            "content_type": content_type,
            "tone": tone,
            "target_length": length,
            "include_research": include_research,
            "quality_threshold": 0.7,  # Quality score threshold for approval
            "max_refinement_iterations": 2
        },
        "research_config": {
            "search_query": enhance_search_query(original_query, content_type),
            "max_sources": 5,
            "research_depth": "medium"
        },
        "execution_metadata": {
            "workflow_name": execution_context.get("workflow_name"),
            "llm_confidence": execution_context.get("llm_confidence", 0.5)
        }
    }


def enhance_search_query(topic: str, content_type: str) -> str:
    """
    Enhance search query based on content type for better research results.

    Args:
        topic: Original topic/query
        content_type: Type of content being created

    Returns:
        Enhanced search query
    """

    enhancements = {
        "article": "article guide comprehensive detailed",
        "report": "report analysis findings data",
        "creative": "creative inspiration examples ideas",
        "technical": "technical documentation specifications",
        "general": "information overview guide"
    }

    enhancement = enhancements.get(content_type, enhancements["general"])
    return f"{topic} {enhancement}"


def conduct_content_research(content_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Conduct research to gather information for content creation.

    Args:
        content_parameters: Content creation parameters

    Returns:
        Research results and source materials
    """

    research_config = content_parameters["research_config"]
    content_config = content_parameters["content_config"]

    # Skip research if not required
    if not content_config["include_research"]:
        return {
            "research_status": "skipped",
            "research_summary": "Research phase skipped per configuration",
            "source_materials": [],
            "research_quality": 0.5
        }

    # Execute web search for research
    search_result = web_search_tool(
        query=research_config["search_query"],
        max_results=research_config["max_sources"],
        search_type="web"
    )

    if search_result["status"] != "success":
        return {
            "research_status": "failed",
            "research_summary": "Failed to gather research materials",
            "source_materials": [],
            "research_quality": 0.0,
            "error": search_result.get("error", "Unknown research error")
        }

    # Process search results
    search_data = search_result["data"]
    source_materials = []

    for result in search_data["results"][:3]:  # Process top 3 results
        # Extract content from each source
        extraction_result = content_extractor_tool(
            url=result["url"],
            extract_type="text"
        )

        if extraction_result["status"] == "success":
            extracted_data = extraction_result["data"]
            source_materials.append({
                "title": result["title"],
                "url": result["url"],
                "snippet": result["snippet"],
                "content": extracted_data["extracted_content"].get("text", ""),
                "relevance_score": result.get("relevance_score", 0.5)
            })

    # Summarize research findings
    if source_materials:
        combined_content = "\n\n".join([
            f"Source: {source['title']}\n{source['content'][:500]}..."
            for source in source_materials
        ])

        summary_result = summarizer_tool(
            content=combined_content,
            length="medium",
            style="neutral"
        )

        research_summary = summary_result["data"]["summary"] if summary_result["status"] == "success" else "Research summary unavailable"
    else:
        research_summary = "No usable research materials found"

    return {
        "research_status": "completed",
        "research_summary": research_summary,
        "source_materials": source_materials,
        "research_quality": calculate_research_quality(source_materials),
        "search_metadata": search_result["metadata"]
    }


def calculate_research_quality(source_materials: List[Dict[str, Any]]) -> float:
    """Calculate research quality score based on sources."""

    if not source_materials:
        return 0.0

    quality_factors = []
    for source in source_materials:
        content_length_factor = min(len(source["content"]) / 1000, 1.0)  # Normalize to 1000 chars
        relevance_factor = source.get("relevance_score", 0.5)
        quality_factors.append((content_length_factor + relevance_factor) / 2)

    return sum(quality_factors) / len(quality_factors)


def generate_initial_content(research_results: Dict[str, Any], content_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate initial content draft based on research and parameters.

    Args:
        research_results: Results from research phase
        content_parameters: Content creation configuration

    Returns:
        Initial content draft with metadata
    """

    content_config = content_parameters["content_config"]

    # Create content based on research and configuration
    content_draft = create_content_draft(
        topic=content_config["topic"],
        content_type=content_config["content_type"],
        tone=content_config["tone"],
        target_length=content_config["target_length"],
        research_summary=research_results.get("research_summary", ""),
        source_materials=research_results.get("source_materials", [])
    )

    # Calculate initial content metrics
    content_metrics = {
        "word_count": len(content_draft.split()),
        "paragraph_count": len([p for p in content_draft.split("\n\n") if p.strip()]),
        "estimated_reading_time": len(content_draft.split()) // 200,  # ~200 words per minute
        "has_introduction": "introduction" in content_draft.lower() or content_draft.startswith("This"),
        "has_conclusion": "conclusion" in content_draft.lower() or "summary" in content_draft.lower()
    }

    return {
        "generation_status": "completed",
        "content_draft": content_draft,
        "content_metrics": content_metrics,
        "generation_metadata": {
            "content_type": content_config["content_type"],
            "tone": content_config["tone"],
            "research_used": research_results["research_status"] == "completed",
            "sources_referenced": len(research_results.get("source_materials", []))
        }
    }


def create_content_draft(topic: str, content_type: str, tone: str, target_length: str,
                        research_summary: str, source_materials: List[Dict[str, Any]]) -> str:
    """
    Create content draft based on parameters.

    In a real implementation, this would call an LLM API for content generation.
    For demonstration, this creates structured mock content.
    """

    # Length mappings
    length_targets = {
        "short": 300,
        "medium": 800,
        "long": 1500
    }

    target_words = length_targets.get(target_length, 800)

    # Create content structure based on type
    if content_type == "article":
        content = create_article_content(topic, tone, target_words, research_summary, source_materials)
    elif content_type == "report":
        content = create_report_content(topic, tone, target_words, research_summary, source_materials)
    elif content_type == "creative":
        content = create_creative_content(topic, tone, target_words, research_summary)
    else:
        content = create_general_content(topic, tone, target_words, research_summary)

    return content


def create_article_content(topic: str, tone: str, target_words: int, research_summary: str, sources: List[Dict[str, Any]]) -> str:
    """Create article-style content."""

    intro = f"# {topic.title()}\n\n"

    if tone == "formal":
        intro += f"This comprehensive article examines {topic} through detailed analysis and research."
    elif tone == "casual":
        intro += f"Let's dive into {topic} and explore what makes it interesting!"
    else:
        intro += f"This article provides an overview of {topic} based on current information and research."

    body = f"\n\n## Overview\n\n{research_summary[:300] if research_summary else f'The topic of {topic} encompasses various important aspects that are worth understanding.'}\n\n"

    if sources:
        body += "## Key Insights\n\n"
        for i, source in enumerate(sources[:3], 1):
            snippet = source.get('snippet', f'Important information about {topic}')[:150]
            body += f"{i}. {snippet}...\n\n"

    body += f"## Implementation\n\nWhen working with {topic}, it's important to consider best practices and established methodologies. "
    body += "This ensures optimal results and helps avoid common pitfalls.\n\n"

    conclusion = f"## Conclusion\n\nIn summary, {topic} represents a significant area of focus that benefits from careful consideration and proper implementation. "
    conclusion += "By following established guidelines and staying informed about latest developments, practitioners can achieve successful outcomes."

    full_content = intro + body + conclusion

    # Adjust length to target
    words = full_content.split()
    if len(words) > target_words:
        full_content = " ".join(words[:target_words]) + "..."
    elif len(words) < target_words * 0.8:
        # Add more content if too short
        full_content += f"\n\n## Additional Considerations\n\nFurther exploration of {topic} reveals additional nuances and opportunities for deeper understanding."

    return full_content


def create_report_content(topic: str, tone: str, target_words: int, research_summary: str, sources: List[Dict[str, Any]]) -> str:
    """Create report-style content."""

    report = f"# {topic.title()} - Analysis Report\n\n"
    report += "## Executive Summary\n\n"
    report += f"This report analyzes {topic} based on available data and research findings. "
    report += f"Key insights and recommendations are provided for stakeholder consideration.\n\n"

    if research_summary:
        report += f"## Findings\n\n{research_summary}\n\n"

    report += f"## Analysis\n\nBased on the research conducted, several key factors emerge regarding {topic}:\n\n"
    report += f"- Current state assessment reveals significant opportunities\n"
    report += f"- Market conditions support continued development\n"
    report += f"- Risk factors are manageable with proper planning\n\n"

    report += f"## Recommendations\n\n1. Proceed with implementation following established best practices\n"
    report += f"2. Monitor key performance indicators regularly\n"
    report += f"3. Maintain flexibility to adapt to changing conditions\n\n"

    report += f"## Conclusion\n\nThe analysis supports a positive outlook for {topic} with appropriate risk management strategies in place."

    return report


def create_creative_content(topic: str, tone: str, target_words: int, research_summary: str) -> str:
    """Create creative content."""

    creative = f"# Exploring {topic.title()}\n\n"

    creative += f"Imagine a world where {topic} transforms the way we think and act. "
    creative += f"This isn't just a concept—it's a reality that's unfolding before our eyes.\n\n"

    creative += f"## The Journey Begins\n\n"
    creative += f"Every story about {topic} starts with curiosity. What if we could push beyond current limitations? "
    creative += f"What possibilities would emerge?\n\n"

    if research_summary:
        creative += f"Research shows us glimpses of this potential: {research_summary[:200]}...\n\n"

    creative += f"## Innovation in Motion\n\n"
    creative += f"The true magic of {topic} lies not in what it is, but in what it could become. "
    creative += f"Each development opens new doors, each breakthrough reveals new horizons.\n\n"

    creative += f"## Looking Forward\n\n"
    creative += f"As we continue to explore {topic}, we're not just observers—we're participants in shaping its future. "
    creative += f"The story continues, and we all have a role to play."

    return creative


def create_general_content(topic: str, tone: str, target_words: int, research_summary: str) -> str:
    """Create general-purpose content."""

    content = f"# Understanding {topic.title()}\n\n"

    content += f"## Introduction\n\n{topic.title()} is an important subject that deserves careful consideration and understanding. "
    content += f"This guide provides essential information to help you navigate this topic effectively.\n\n"

    if research_summary:
        content += f"## Background Information\n\n{research_summary}\n\n"

    content += f"## Key Concepts\n\nWhen exploring {topic}, several fundamental concepts emerge:\n\n"
    content += f"- Core principles that guide implementation\n"
    content += f"- Best practices developed through experience\n"
    content += f"- Common challenges and their solutions\n\n"

    content += f"## Practical Applications\n\n{topic.title()} has practical applications across various domains. "
    content += f"Understanding these applications helps in making informed decisions about implementation and use.\n\n"

    content += f"## Conclusion\n\nMastering {topic} requires both theoretical understanding and practical experience. "
    content += f"With proper preparation and continued learning, success is achievable."

    return content


def assess_content_quality(content_generation: Dict[str, Any], content_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess the quality of generated content against defined criteria.

    Args:
        content_generation: Generated content data
        content_parameters: Content configuration parameters

    Returns:
        Quality assessment results with pass/fail decision
    """

    content_draft = content_generation["content_draft"]
    content_metrics = content_generation["content_metrics"]
    content_config = content_parameters["content_config"]

    # Quality assessment criteria
    quality_scores = {}

    # 1. Length appropriateness
    target_length = content_config["target_length"]
    word_count = content_metrics["word_count"]

    length_targets = {"short": 300, "medium": 800, "long": 1500}
    target_words = length_targets.get(target_length, 800)

    length_ratio = word_count / target_words
    quality_scores["length_score"] = min(1.0, max(0.0, 1.0 - abs(length_ratio - 1.0)))

    # 2. Structure quality
    has_intro = content_metrics.get("has_introduction", False)
    has_conclusion = content_metrics.get("has_conclusion", False)
    paragraph_count = content_metrics.get("paragraph_count", 0)

    structure_score = 0.0
    if has_intro:
        structure_score += 0.4
    if has_conclusion:
        structure_score += 0.4
    if paragraph_count >= 3:
        structure_score += 0.2

    quality_scores["structure_score"] = structure_score

    # 3. Content completeness
    topic_mentions = content_draft.lower().count(content_config["topic"].lower())
    completeness_score = min(1.0, topic_mentions / 3.0)  # Expect at least 3 mentions
    quality_scores["completeness_score"] = completeness_score

    # 4. Readability (basic)
    avg_sentence_length = word_count / max(content_draft.count('.'), 1)
    readability_score = 1.0 if 10 <= avg_sentence_length <= 20 else 0.7
    quality_scores["readability_score"] = readability_score

    # Calculate overall quality score
    overall_quality = sum(quality_scores.values()) / len(quality_scores)

    # Determine if quality meets threshold
    quality_threshold = content_config["quality_threshold"]
    passes_quality_check = overall_quality >= quality_threshold

    return {
        "quality_assessment": "passed" if passes_quality_check else "failed",
        "overall_quality_score": overall_quality,
        "quality_threshold": quality_threshold,
        "individual_scores": quality_scores,
        "improvement_suggestions": generate_improvement_suggestions(quality_scores, content_metrics) if not passes_quality_check else [],
        "quality_metadata": {
            "assessment_criteria": list(quality_scores.keys()),
            "word_count": word_count,
            "target_length": target_length
        }
    }


def generate_improvement_suggestions(quality_scores: Dict[str, float], content_metrics: Dict[str, Any]) -> List[str]:
    """Generate specific improvement suggestions based on quality scores."""

    suggestions = []

    if quality_scores.get("length_score", 1.0) < 0.7:
        suggestions.append("Adjust content length to better match target requirements")

    if quality_scores.get("structure_score", 1.0) < 0.7:
        if not content_metrics.get("has_introduction"):
            suggestions.append("Add a clear introduction section")
        if not content_metrics.get("has_conclusion"):
            suggestions.append("Add a comprehensive conclusion")
        if content_metrics.get("paragraph_count", 0) < 3:
            suggestions.append("Expand content with additional paragraphs and sections")

    if quality_scores.get("completeness_score", 1.0) < 0.7:
        suggestions.append("Include more references to the main topic throughout the content")

    if quality_scores.get("readability_score", 1.0) < 0.8:
        suggestions.append("Improve readability by adjusting sentence length and complexity")

    return suggestions


def refine_content(content_generation: Dict[str, Any], quality_assessment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Refine content based on quality assessment feedback.

    Args:
        content_generation: Original content generation data
        quality_assessment: Quality assessment results

    Returns:
        Refined content
    """

    original_draft = content_generation["content_draft"]
    improvement_suggestions = quality_assessment["improvement_suggestions"]

    # Apply improvements based on suggestions
    refined_content = original_draft

    # Simple refinement logic (in real implementation, this would use LLM for refinement)
    for suggestion in improvement_suggestions:
        if "introduction" in suggestion.lower():
            if not refined_content.startswith("# ") and "# " not in refined_content[:100]:
                topic = content_generation.get("generation_metadata", {}).get("content_type", "topic")
                refined_content = f"# Introduction to {topic.title()}\n\n" + refined_content

        elif "conclusion" in suggestion.lower():
            if "conclusion" not in refined_content.lower():
                refined_content += "\n\n## Conclusion\n\nIn summary, this content provides valuable insights and information that contribute to a comprehensive understanding of the topic."

        elif "length" in suggestion.lower():
            words = refined_content.split()
            if len(words) < 400:  # If too short, add content
                refined_content += "\n\n## Additional Information\n\nFurther exploration of this topic reveals additional nuances and considerations that are important for a complete understanding."

        elif "readability" in suggestion.lower():
            # Simple sentence splitting for better readability
            refined_content = refined_content.replace(". ", ".\n\n").replace("\n\n\n", "\n\n")

    # Update metrics for refined content
    refined_metrics = {
        "word_count": len(refined_content.split()),
        "paragraph_count": len([p for p in refined_content.split("\n\n") if p.strip()]),
        "estimated_reading_time": len(refined_content.split()) // 200,
        "has_introduction": refined_content.startswith("# ") or "# " in refined_content[:200],
        "has_conclusion": "conclusion" in refined_content.lower()
    }

    return {
        "refinement_status": "completed",
        "refined_content": refined_content,
        "refined_metrics": refined_metrics,
        "improvements_applied": improvement_suggestions,
        "refinement_metadata": {
            "original_word_count": content_generation["content_metrics"]["word_count"],
            "refined_word_count": refined_metrics["word_count"],
            "improvement_count": len(improvement_suggestions)
        }
    }


def format_final_content(final_content_data: Dict[str, Any], content_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply final formatting to the content for publication.

    Args:
        final_content_data: Final content (either refined or original)
        content_parameters: Content configuration parameters

    Returns:
        Formatted content ready for publication
    """

    # Determine which content to use (refined or original)
    if "refined_content" in final_content_data:
        content = final_content_data["refined_content"]
        content_metrics = final_content_data["refined_metrics"]
        source = "refined"
    else:
        content = final_content_data["content_draft"]
        content_metrics = final_content_data["content_metrics"]
        source = "original"

    content_config = content_parameters["content_config"]

    # Apply formatting based on content type
    formatted_content = apply_content_formatting(content, content_config["content_type"])

    # Add metadata and publication info
    publication_metadata = {
        "title": extract_title_from_content(content),
        "content_type": content_config["content_type"],
        "tone": content_config["tone"],
        "word_count": content_metrics["word_count"],
        "estimated_reading_time": f"{content_metrics['estimated_reading_time']} minutes",
        "publication_ready": True,
        "formatting_applied": True,
        "content_source": source
    }

    return {
        "formatting_status": "completed",
        "final_content": formatted_content,
        "publication_metadata": publication_metadata,
        "content_statistics": content_metrics,
        "ready_for_publication": True
    }


def apply_content_formatting(content: str, content_type: str) -> str:
    """Apply type-specific formatting to content."""

    # Ensure proper markdown formatting
    formatted = content

    # Add proper line spacing
    formatted = formatted.replace("\n\n\n", "\n\n")

    # Ensure headers have proper spacing
    lines = formatted.split("\n")
    formatted_lines = []

    for i, line in enumerate(lines):
        formatted_lines.append(line)

        # Add spacing after headers
        if line.startswith("#") and i + 1 < len(lines) and lines[i + 1].strip():
            if not lines[i + 1].startswith("#"):
                formatted_lines.append("")

    formatted = "\n".join(formatted_lines)

    # Content type specific formatting
    if content_type == "report":
        # Add page breaks for reports
        formatted = formatted.replace("## ", "\n---\n\n## ")

    elif content_type == "article":
        # Ensure proper article structure
        if not formatted.startswith("# "):
            formatted = "# Article Title\n\n" + formatted

    return formatted.strip()


def extract_title_from_content(content: str) -> str:
    """Extract title from content."""

    lines = content.split("\n")
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()

    # Fallback: use first line or generate generic title
    first_line = lines[0].strip() if lines else "Content"
    return first_line[:50] + "..." if len(first_line) > 50 else first_line


# Conditional Logic for Quality Control

def should_refine_content(quality_assessment: Dict[str, Any], content_parameters: Dict[str, Any]) -> str:
    """
    Determine if content should be refined based on quality assessment.

    This function is used by the Conditional node to route the workflow.

    Args:
        quality_assessment: Quality assessment results
        content_parameters: Content configuration

    Returns:
        Route name ("refine_content" or "format_content")
    """

    if quality_assessment["quality_assessment"] == "failed":
        # Check if we haven't exceeded max iterations
        # (In a real implementation, this would track iteration count)
        return "refine_content"
    else:
        return "format_content"


# Main Pipeline Creation Function

def create_content_pipeline() -> Pipeline:
    """
    Create the complete content creation workflow pipeline.

    Returns:
        Configured content pipeline with conditional quality control
    """

    return Pipeline([
        # Step 1: Prepare content parameters
        PythonTask(
            function=prepare_content_parameters,
            name="content_preparation",
            kwargs={"workflow_context": pickled("workflow_context")},
            returns=[pickled("content_parameters")]
        ),

        # Step 2: Conduct research
        PythonTask(
            function=conduct_content_research,
            name="content_research",
            kwargs={"content_parameters": pickled("content_parameters")},
            returns=[pickled("research_results")]
        ),

        # Step 3: Generate initial content
        PythonTask(
            function=generate_initial_content,
            name="content_generation",
            kwargs={
                "research_results": pickled("research_results"),
                "content_parameters": pickled("content_parameters")
            },
            returns=[pickled("content_generation")]
        ),

        # Step 4: Assess content quality
        PythonTask(
            function=assess_content_quality,
            name="quality_assessment",
            kwargs={
                "content_generation": pickled("content_generation"),
                "content_parameters": pickled("content_parameters")
            },
            returns=[pickled("quality_assessment")]
        ),

        # Step 5: Conditional routing based on quality
        Conditional(
            name="quality_control_router",
            condition=lambda qa, cp: should_refine_content(qa, cp),
            condition_input=[pickled("quality_assessment"), pickled("content_parameters")],
            branches={
                "refine_content": Pipeline([
                    PythonTask(
                        function=refine_content,
                        name="content_refinement",
                        kwargs={
                            "content_generation": pickled("content_generation"),
                            "quality_assessment": pickled("quality_assessment")
                        },
                        returns=[pickled("refined_content")]
                    ),
                    PythonTask(
                        function=format_final_content,
                        name="final_formatting",
                        kwargs={
                            "final_content_data": pickled("refined_content"),
                            "content_parameters": pickled("content_parameters")
                        },
                        returns=[pickled("formatted_content")]
                    )
                ]),
                "format_content": Pipeline([
                    PythonTask(
                        function=format_final_content,
                        name="direct_formatting",
                        kwargs={
                            "final_content_data": pickled("content_generation"),
                            "content_parameters": pickled("content_parameters")
                        },
                        returns=[pickled("formatted_content")]
                    )
                ])
            }
        )
    ])


# Example usage and testing
if __name__ == "__main__":

    # Example workflow context
    example_context = {
        "execution_context": {
            "workflow_name": "content_creation_workflow",
            "original_query": "sustainable energy solutions",
            "llm_confidence": 0.82
        },
        "workflow_parameters": {
            "content_type": "article",
            "tone": "neutral",
            "length": "medium",
            "include_research": True
        }
    }

    print("✍️ Content Creation Workflow - Test Components")
    print("=" * 55)

    # Test content preparation
    print("\n1. Testing Content Preparation:")
    content_params = prepare_content_parameters(example_context)
    print(f"   Content Type: {content_params['content_config']['content_type']}")
    print(f"   Target Length: {content_params['content_config']['target_length']}")
    print(f"   Include Research: {content_params['content_config']['include_research']}")

    # Test research phase
    print("\n2. Testing Research Phase:")
    research_results = conduct_content_research(content_params)
    print(f"   Research Status: {research_results['research_status']}")
    print(f"   Source Materials: {len(research_results.get('source_materials', []))}")
    print(f"   Research Quality: {research_results.get('research_quality', 0):.2f}")

    # Test content generation
    print("\n3. Testing Content Generation:")
    content_gen = generate_initial_content(research_results, content_params)
    print(f"   Generation Status: {content_gen['generation_status']}")
    print(f"   Word Count: {content_gen['content_metrics']['word_count']}")
    print(f"   Has Introduction: {content_gen['content_metrics']['has_introduction']}")

    # Test quality assessment
    print("\n4. Testing Quality Assessment:")
    quality_assessment = assess_content_quality(content_gen, content_params)
    print(f"   Quality Assessment: {quality_assessment['quality_assessment']}")
    print(f"   Overall Score: {quality_assessment['overall_quality_score']:.2f}")
    print(f"   Improvement Suggestions: {len(quality_assessment['improvement_suggestions'])}")

    # Test conditional routing
    print("\n5. Testing Conditional Routing:")
    route = should_refine_content(quality_assessment, content_params)
    print(f"   Routing Decision: {route}")

    print("\n✅ Content creation workflow components ready for pipeline execution")

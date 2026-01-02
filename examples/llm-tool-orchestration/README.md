# LLM + Tool Orchestration with Runnable

This directory demonstrates advanced orchestration patterns that combine LLM (Large Language Model) agents with tool execution using the runnable framework. The examples show how an LLM can analyze user queries, select appropriate workflows, and orchestrate tool execution through sequential, parallel, and conditional patterns.

## Overview

The LLM + Tool orchestration system provides:
- **Intelligent Workflow Selection**: LLM agents analyze user input and select optimal tool workflows
- **Dynamic Tool Orchestration**: Execute tools based on LLM decisions with mixed orchestration patterns
- **Modular Architecture**: Reusable components for LLMs, tools, and workflow templates
- **Error Handling**: Robust error recovery and fallback mechanisms
- **Extensible Design**: Easy to add new workflows, tools, and LLM models

## Architecture

```
User Query → LLM Agent → Workflow Router → [Selected Workflow] → Tool Execution → Results
                ↓                          ↓
         Workflow Decision           Mixed Orchestration:
         {workflow, params}          - Sequential tools
                                    - Parallel tools
                                    - Conditional branches
```

## Core Components

### 1. LLM Workflow Selector (`llm_workflow_selector.py`)
The intelligent agent that analyzes user queries and selects appropriate workflows.

```python
from llm_workflow_selector import llm_workflow_selector

# Analyze query and select workflow
result = llm_workflow_selector(
    user_query="Research machine learning best practices",
    context={"user_expertise": "intermediate"}
)

# Returns structured decision
{
    "selected_workflow": "research_workflow",
    "workflow_parameters": {"depth": "detailed", "format": "summary"},
    "confidence": 0.85,
    "reasoning": "Query requires comprehensive research and analysis"
}
```

### 2. API Tool Library (`api_tools.py`)
Generic wrappers for external APIs and tools with standardized response format.

```python
from api_tools import web_search_tool, summarizer_tool, data_analyzer_tool

# Web search with standardized response
search_result = web_search_tool("machine learning", max_results=3)
# Returns: {"status": "success", "data": {...}, "metadata": {...}}

# Content summarization
summary = summarizer_tool(content="...", length="medium", style="neutral")

# Data analysis
analysis = data_analyzer_tool(data={...}, analysis_type="descriptive")
```

### 3. Workflow Router (`workflow_router.py`)
Conditional routing logic that directs execution to appropriate workflow templates.

```python
from workflow_router import create_main_workflow_router

# Create router with workflow branches
router = create_main_workflow_router({
    "research_workflow": research_pipeline,
    "analysis_workflow": analysis_pipeline,
    "content_workflow": content_pipeline
})
```

### 4. Workflow Templates
Pre-built workflow templates for common orchestration patterns:

- **Research Workflow** (`research_workflow.py`): Search → Extract → Summarize
- **Analysis Workflow** (`analysis_workflow.py`): Data Collection → Parallel Analysis → Synthesis
- **Content Workflow** (`content_workflow.py`): Research → Generate → Quality Check → Format

## Examples

### Basic Demo (`basic_demo.py`) - **Start Here**
The simplest working example demonstrating core LLM + tool orchestration:

```bash
cd examples/llm-tool-orchestration
uv run basic_demo.py
```

**Output:**
```
🤖 LLM selecting workflow...
Selected: research_workflow
🔧 Executing tools...
✅ Pipeline completed successfully!
🎯 LLM + Tool orchestration demonstrated:
   1. ✅ LLM analyzed query and selected workflow
   2. ✅ Tools executed based on LLM selection
   3. ✅ Results stored in catalog for further processing
```

### Minimal Working Demo (`minimal_working_demo.py`)
Clean three-step pipeline showing the complete flow:

```bash
uv run minimal_working_demo.py
```

Features:
- LLM query analysis and workflow selection
- Tool execution based on LLM decision
- Result formatting and aggregation

### Simple Integration Demo (`simple_integration_demo.py`)
Demonstrates conditional routing to different workflow branches:

```bash
uv run simple_integration_demo.py
```

Shows:
- Multiple workflow templates (research, analysis, content creation)
- Conditional routing based on LLM selection
- Mixed orchestration patterns within workflows

### Advanced Mixed Demo (`advanced_mixed_demo.py`)
Complex orchestration with sophisticated control flow:

```bash
uv run advanced_mixed_demo.py
```

Advanced patterns:
- Multi-step LLM analysis and planning
- Sequential tool execution (3 tools chained)
- Conditional processing based on execution results
- Dynamic processing paths (synthesis/retry/fallback)
- Context-aware final report generation

### Complete Integration (`main_llm_tool_pipeline.py`)
Full-featured pipeline integration with all components:

```bash
uv run main_llm_tool_pipeline.py
```

Enterprise features:
- Complete workflow template library
- Advanced result aggregation
- Comprehensive error handling
- Quality assessment and reporting

## Orchestration Patterns

### Sequential Execution
Tools execute one after another with data flow:

```python
# Research Workflow Example
Pipeline([
    PythonTask(function=web_search_tool, name="search"),
    PythonTask(function=content_extractor_tool, name="extract"),
    PythonTask(function=summarizer_tool, name="summarize")
])
```

### Parallel Execution
Independent tools execute simultaneously:

```python
# Analysis Workflow Example
Parallel([
    PythonTask(function=data_analyzer_tool, name="analyze"),
    PythonTask(function=statistical_tool, name="statistics"),
    PythonTask(function=visualization_tool, name="visualize")
])
```

### Conditional Routing
Dynamic workflow selection based on results:

```python
# Content Workflow Example
Conditional(
    condition=lambda result: result["quality_score"] > 0.8,
    branches={
        "high_quality": publish_pipeline,
        "needs_improvement": refinement_pipeline
    }
)
```

## Key Features

### Intelligent Workflow Selection
- **Context-Aware**: LLM considers user expertise, preferences, and query complexity
- **Confidence Scoring**: Provides confidence metrics for workflow decisions
- **Reasoning Transparency**: Explains why specific workflows were selected
- **Fallback Handling**: Graceful degradation when preferred workflows fail

### Tool Integration
- **Standardized Interface**: All tools follow consistent input/output patterns
- **Error Recovery**: Automatic retry with exponential backoff
- **Performance Monitoring**: Execution time and cost tracking
- **Generic Wrappers**: Easy integration of new APIs and tools

### Mixed Orchestration
- **Sequential Chains**: Data flows through ordered tool execution
- **Parallel Processing**: Independent tools execute concurrently for efficiency
- **Conditional Logic**: Dynamic routing based on intermediate results
- **Nested Workflows**: Complex patterns combining multiple orchestration types

### Result Management
- **Structured Aggregation**: Consistent result formatting across workflows
- **Quality Assessment**: Automatic quality scoring and validation
- **Metadata Tracking**: Comprehensive execution metadata and lineage
- **User-Friendly Output**: Formatted results for easy consumption

## Adding New Components

### New Workflow Template
1. Create workflow file (e.g., `my_workflow.py`)
2. Implement `create_my_pipeline()` function returning Pipeline
3. Register in workflow router branches
4. Add LLM selection logic for when to use it

```python
# my_workflow.py
from runnable import Pipeline, PythonTask, pickled

def create_my_pipeline():
    return Pipeline([
        PythonTask(
            function=my_tool_function,
            name="my_step",
            returns=[pickled("my_result")]
        )
    ])
```

### New API Tool
1. Add tool function to `api_tools.py`
2. Follow standardized response format
3. Include error handling and retries
4. Add to workflow templates as needed

```python
# In api_tools.py
def my_new_tool(param1: str, param2: int) -> Dict[str, Any]:
    try:
        # Tool implementation
        result = call_external_api(param1, param2)

        return create_tool_response(
            status="success",
            data=result,
            tool_name="my_new_tool",
            execution_time=1.23
        )
    except Exception as e:
        return create_tool_response(
            status="error",
            error=str(e),
            tool_name="my_new_tool"
        )
```

### New LLM Model
Replace the LLM decision logic in `llm_workflow_selector.py`:

```python
# For real LLM integration
def llm_workflow_selector(user_query: str, context: Dict[str, Any] = None):
    # Replace mock logic with actual LLM API calls
    response = your_llm_api.chat.completions.create(
        model="your-model",
        messages=[{"role": "user", "content": create_selection_prompt(user_query, context)}],
        response_format={"type": "json_object"}
    )

    return parse_llm_response(response)
```

## Configuration

### Tool Settings
Customize tool behavior through parameters:

```python
# Web search configuration
search_result = web_search_tool(
    query="machine learning",
    max_results=5,          # Number of results
    search_type="web",      # web, news, images
    freshness="month"       # day, week, month, year
)

# Summarizer configuration
summary = summarizer_tool(
    content="...",
    length="medium",        # short, medium, long
    style="professional",   # neutral, professional, casual
    focus="key_points"      # overview, key_points, technical
)
```

### Workflow Parameters
Control workflow execution through LLM-generated parameters:

```python
# Research workflow parameters
{
    "depth": "detailed",           # surface, standard, detailed
    "format": "structured_summary", # summary, report, bullet_points
    "sources": 5,                  # Number of sources to analyze
    "focus_areas": ["methodology", "results", "implications"]
}

# Analysis workflow parameters
{
    "analysis_type": "comprehensive",  # basic, standard, comprehensive
    "visualizations": True,            # Include charts and graphs
    "statistical_tests": ["correlation", "regression"],
    "confidence_level": 0.95
}
```

## Error Handling

### Tool Failure Recovery
- **Automatic Retry**: Failed tools retry with exponential backoff
- **Fallback Tools**: Alternative tools when primary tools fail
- **Partial Results**: Graceful handling of incomplete data
- **Error Aggregation**: Comprehensive error reporting

### Workflow Failures
- **Fallback Workflows**: Alternative workflows when selection fails
- **Partial Execution**: Complete successful steps even if others fail
- **Quality Gates**: Conditional routing based on result quality
- **Manual Intervention**: Clear failure points for human review

### Example Error Handling
```python
# Tool with retry logic
def robust_web_search(query: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return web_search_tool(query)
        except Exception as e:
            if attempt == max_retries - 1:
                return create_tool_response("error", error=str(e), tool_name="web_search")
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Performance Optimization

### Parallel Tool Execution
Use runnable's Parallel node for independent tools:

```python
# Execute multiple analyses simultaneously
parallel_analysis = Parallel([
    PythonTask(function=sentiment_analysis, name="sentiment"),
    PythonTask(function=keyword_extraction, name="keywords"),
    PythonTask(function=topic_modeling, name="topics")
])
```

### Caching
Implement result caching for expensive operations:

```python
# Tool with caching
@lru_cache(maxsize=100)
def cached_web_search(query: str, max_results: int):
    return web_search_tool(query, max_results)
```

### Resource Management
Monitor and control resource usage:

```python
# Track execution metrics
{
    "execution_time": 3.45,
    "api_calls": 7,
    "tokens_used": 1250,
    "cost_estimate": 0.023
}
```

## Best Practices

### LLM Integration
- **Clear Prompts**: Use structured prompts for consistent workflow selection
- **Validation**: Validate LLM responses before workflow execution
- **Fallbacks**: Always have fallback workflows for edge cases
- **Context Management**: Maintain conversation context across pipeline steps

### Tool Design
- **Idempotent**: Tools should produce same results given same inputs
- **Stateless**: Avoid tool dependencies on external state
- **Standardized**: Follow consistent input/output patterns
- **Resilient**: Handle network failures and API rate limits gracefully

### Workflow Composition
- **Modular**: Create reusable workflow components
- **Parameterized**: Make workflows configurable through parameters
- **Testable**: Include unit tests for individual workflow steps
- **Documented**: Clear documentation for workflow purpose and usage

### Error Management
- **Fail Fast**: Detect and report errors early in pipeline
- **Graceful Degradation**: Provide partial results when possible
- **User Feedback**: Clear error messages for end users
- **Logging**: Comprehensive logging for debugging and monitoring

## Testing

### Unit Tests
Test individual components in isolation:

```bash
# Test LLM workflow selector
pytest tests/test_llm_workflow_selector.py

# Test API tools
pytest tests/test_api_tools.py

# Test workflow templates
pytest tests/test_workflows.py
```

### Integration Tests
Test complete pipeline execution:

```bash
# Test basic demo end-to-end
pytest tests/test_basic_demo.py

# Test advanced orchestration
pytest tests/test_advanced_demo.py
```

### Mock Testing
Use mock responses for external API testing:

```python
# Mock API responses for testing
@patch('api_tools.requests.get')
def test_web_search_tool(mock_get):
    mock_get.return_value.json.return_value = {"results": [...]}
    result = web_search_tool("test query")
    assert result["status"] == "success"
```

## Monitoring and Observability

### Execution Metrics
Track pipeline performance:
- Workflow selection accuracy
- Tool execution times
- Success/failure rates
- Resource utilization

### Quality Metrics
Monitor result quality:
- LLM confidence scores
- Tool result validation
- User satisfaction feedback
- Error frequency analysis

### Cost Tracking
Monitor resource costs:
- API call costs
- Compute resource usage
- Storage requirements
- Execution time costs

## Deployment Considerations

### Production Readiness
- **API Keys**: Secure management of external API credentials
- **Rate Limiting**: Respect API rate limits and implement backoff
- **Monitoring**: Comprehensive logging and metrics collection
- **Scaling**: Horizontal scaling for high-throughput scenarios

### Security
- **Input Validation**: Sanitize user inputs before processing
- **API Security**: Secure handling of API credentials and responses
- **Data Privacy**: Proper handling of sensitive user data
- **Access Control**: Authentication and authorization for pipeline access

## Troubleshooting

### Common Issues

**Issue: Parameter passing errors**
```
Solution: Ensure proper use of pickled() for parameter passing between tasks:
returns=[pickled("result_name")]
```

**Issue: Conditional routing failures**
```
Solution: Verify condition functions return valid branch names:
condition=lambda result: "success_branch" if result["status"] == "success" else "error_branch"
```

**Issue: Tool execution timeouts**
```
Solution: Implement timeout handling in tool wrappers:
response = requests.get(url, timeout=30)
```

**Issue: LLM selection inconsistency**
```
Solution: Use more structured prompts and validation:
- Add examples in prompts
- Validate response format
- Implement fallback selection logic
```

### Debug Mode
Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run pipeline with debug logging
pipeline.execute()
```

### Performance Issues
- **Slow LLM responses**: Consider using faster models for workflow selection
- **API rate limits**: Implement proper backoff and retry logic
- **Memory usage**: Monitor data passing between pipeline steps
- **Parallel execution**: Use Parallel nodes for independent operations

## Contributing

To contribute new examples or improvements:

1. Follow the existing code patterns and structure
2. Add comprehensive tests for new components
3. Update documentation for new features
4. Ensure examples run successfully with `uv run`
5. Follow runnable's coding standards and patterns

## Further Reading

- [Runnable Documentation](../README.md)
- [Pipeline Design Patterns](../01-tasks/)
- [Advanced Orchestration](../05-nodes/)
- [Error Handling Best Practices](../06-error-handling/)

---

This framework provides a solid foundation for building sophisticated LLM + tool orchestration systems with runnable. Start with the basic demo and gradually explore more advanced patterns as your needs grow.

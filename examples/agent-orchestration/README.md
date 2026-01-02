# Agent Orchestration Examples

This directory contains examples of using runnable as an orchestration engine for different types of agents. Each example demonstrates sequential agent workflows with data passing between agents.

## Examples Overview

### 1. LLM/AI Agent Orchestration (`llm_agents_sequential.py`)
**Use Case**: Document analysis pipeline with specialized AI agents
- **Document Analyzer**: Extracts key information from text
- **Content Summarizer**: Creates structured summaries
- **Sentiment Analyzer**: Analyzes emotional tone
- **Report Generator**: Creates final formatted report

**Best For**: NLP workflows, content processing, AI-driven analysis chains

### 2. Software Agent Orchestration (`software_agents_sequential.py`)
**Use Case**: Data processing pipeline with automated services
- **Data Fetcher Agent**: Retrieves data from external APIs
- **Data Validator Agent**: Validates and cleans data
- **Database Agent**: Stores processed data
- **Notification Agent**: Sends alerts and notifications

**Best For**: ETL workflows, API integrations, automated system operations

### 3. Human Agent Workflow (`human_agents_sequential.py`)
**Use Case**: Content approval workflow requiring human decisions
- **Content Creator**: Human creates initial content
- **Technical Reviewer**: Human reviews technical accuracy
- **Legal Reviewer**: Human reviews for compliance
- **Manager Approver**: Human gives final approval
- **Publisher Agent**: Automated publishing after approvals

**Best For**: Approval workflows, quality assurance, compliance processes

### 4. Mixed Agent Orchestration (`mixed_agents_sequential.py`)
**Use Case**: Customer support incident resolution with all agent types
- **AI Triage Agent**: Initial issue classification
- **System Diagnostics Agent**: Automated system checks
- **Technical Expert**: Human analysis and decision making
- **AI Solution Generator**: Creates implementation plans
- **Implementation Agent**: Executes automated solutions
- **Support Agent**: Human verification and customer communication
- **AI Documentation Agent**: Final documentation and follow-up

**Best For**: Complex workflows requiring AI intelligence, automation, and human oversight

## Running the Examples

Each example is a standalone Python script that can be executed directly:

```bash
# Run individual examples
uv run examples/agent-orchestration/llm_agents_sequential.py
uv run examples/agent-orchestration/software_agents_sequential.py
uv run examples/agent-orchestration/human_agents_sequential.py
uv run examples/agent-orchestration/mixed_agents_sequential.py
```

## Key Patterns Demonstrated

### 1. Sequential Agent Chaining
Each agent's output becomes the next agent's input, creating a processing pipeline:

```python
pipeline = Pipeline([
    PythonTask(function=agent1, returns=[pickled("result1")]),
    PythonTask(function=agent2, kwargs={"input": pickled("result1")}, returns=[pickled("result2")]),
    PythonTask(function=agent3, kwargs={"input": pickled("result2")}, returns=[pickled("final")])
])
```

### 2. Data Flow Between Agents
Using runnable's `pickled()` mechanism to pass complex data structures:

```python
# Agent 1 outputs
return {"analysis": data, "metadata": info}

# Agent 2 receives
def next_agent(previous_result: Dict[str, Any]) -> Dict[str, Any]:
    analysis = previous_result["analysis"]
    # Process and return new data
    return {"processed": analysis, "new_info": additional_data}
```

### 3. Agent Specialization
Each agent has a specific role and expertise:
- **AI agents** excel at analysis, classification, and content generation
- **Software agents** handle system operations, API calls, and automation
- **Human agents** provide judgment, approval, and complex decision-making

### 4. Error Handling and State Management
Agents can handle failures and maintain context across the workflow:

```python
def resilient_agent(input_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = process_data(input_data)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "error": str(e), "fallback": fallback_data}
```

## Extending These Examples

### Adding New Agent Types
1. Create a new function following the same pattern:
```python
def my_custom_agent(input_data: Dict[str, Any]) -> Dict[str, Any]:
    # Your agent logic here
    return {"processed_data": result}
```

2. Add to your pipeline:
```python
PythonTask(function=my_custom_agent, kwargs={"input_data": pickled("previous_result")})
```

### Real-World Integration
To integrate with actual services, replace the mock implementations with real API calls:

```python
# Replace mock AI calls with real ones
import openai
def ai_agent(input_text: str) -> Dict[str, Any]:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": input_text}]
    )
    return {"ai_response": response.choices[0].message.content}

# Replace mock system calls with real ones
import requests
def api_agent(endpoint: str) -> Dict[str, Any]:
    response = requests.get(endpoint)
    return {"api_data": response.json()}
```

### Parallel Agent Execution
For independent agents that can run simultaneously, use runnable's parallel execution:

```python
from runnable import Pipeline, ParallelStep, PythonTask

parallel_agents = ParallelStep(
    name="parallel_analysis",
    steps=[
        PythonTask(function=ai_agent1, name="analyzer1"),
        PythonTask(function=ai_agent2, name="analyzer2"),
        PythonTask(function=system_agent, name="system_check")
    ]
)
```

## Architecture Benefits

Using runnable for agent orchestration provides:

1. **Reproducibility**: All agent interactions are logged and can be replayed
2. **Scalability**: Can run locally, in containers, or on Kubernetes
3. **Monitoring**: Built-in execution tracking and metrics
4. **Error Handling**: Robust error handling and retry mechanisms
5. **Flexibility**: Easy to modify workflows without changing agent implementations
6. **Testing**: Can mock individual agents for testing the overall workflow

## Next Steps

- **Conditional Logic**: Add conditional nodes to route based on agent outputs
- **Parallel Processing**: Use parallel steps for independent agent operations
- **Error Recovery**: Implement retry logic and fallback agents
- **Monitoring**: Add custom metrics and alerts for agent performance
- **Security**: Implement proper authentication and authorization for agents

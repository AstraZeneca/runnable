# Custom Patterns

Create your own pipeline flow control patterns through runnable's extensible node type system.

## The Core Insight

**All node types follow the same pattern**: They create pipeline flow control structures that wrap a `BaseNode` implementation, with runnable providing orchestration, graph traversal, and execution coordination.

Runnable includes several built-in node types for common patterns:

- **Task nodes**: Execute individual functions, notebooks, or shell commands
- **Parallel nodes**: Execute multiple branches simultaneously ([see Parallel Execution](parallel-execution.md))
- **Map nodes**: Iterate over data with pipeline execution per item ([see Map Patterns](map-patterns.md))
- **Conditional nodes**: Execute different branches based on parameters ([see Conditional Workflows](conditional-workflows.md))
- **Stub nodes**: Pass-through nodes for testing and placeholders ([see Mocking and Testing](mocking-testing.md))

## Understanding the Pattern with Parallel Nodes

Let's examine how the `Parallel` node works to understand the extensibility pattern:

```python
from runnable import Pipeline, PythonTask, Parallel

def main():
    parallel_step = Parallel(
        name="process_parallel",
        branches={
            "branch_a": Pipeline(steps=[PythonTask(name="task_a", function=task_a)]),
            "branch_b": Pipeline(steps=[PythonTask(name="task_b", function=task_b)])
        }
    )
    pipeline = Pipeline(steps=[parallel_step])
    pipeline.execute()
    return pipeline
```

**What happens internally:**
1. `Parallel` (SDK class) provides the user API
2. `create_node()` method converts to `ParallelNode` (execution implementation)
3. `ParallelNode` handles the actual parallel branch coordination
4. Entry point registration makes it discoverable

## The Plugin System

**Node types are pluggable** - runnable automatically discovers and loads custom node types via entry points.

### How Pipeline Nodes Work Internally

**Every node type follows the same pattern**:
1. **Node class**: Provides the pipeline API (`Parallel`, `Map`, etc.)
2. **Node implementation**: Handles the actual execution logic (`ParallelNode`, `MapNode`, etc.)
3. **Entry point registration**: Makes it discoverable

```python
# Built-in node types are registered like this:
[project.entry-points.'nodes']
"task" = "extensions.nodes.task:TaskNode"
"parallel" = "extensions.nodes.parallel:ParallelNode"
"map" = "extensions.nodes.map:MapNode"
"conditional" = "extensions.nodes.conditional:ConditionalNode"
"stub" = "extensions.nodes.stub:StubNode"
"success" = "extensions.nodes.success:SuccessNode"
"fail" = "extensions.nodes.fail:FailNode"
```

## Building Custom Node Types

Create new node types for your specific pipeline flow control needs:

### 1. Implement the Node Implementation
```python
# my_package/nodes.py
from runnable.nodes import BaseNode
from runnable.datastore import StepAttempt

class RetryNode(BaseNode):
    """Execute a pipeline with retry logic"""

    def __init__(self, max_attempts: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_attempts = max_attempts

    def execute(self, **kwargs) -> StepAttempt:
        # Your retry execution logic
        for attempt in range(self.max_attempts):
            try:
                # Execute the branch pipeline
                result = self._execute_branch()
                if result.status == "SUCCESS":
                    return result
            except Exception as e:
                if attempt == self.max_attempts - 1:
                    return StepAttempt(status="FAIL", message=str(e))
                # Log and continue to next attempt

        return StepAttempt(status="FAIL")
```

### 2. Create the Pipeline Node Wrapper
```python
# my_package/nodes.py
from runnable.sdk import BaseTraversal

class Retry(BaseTraversal):
    """Retry node for pipeline execution with failure recovery"""
    branch: "Pipeline"
    max_attempts: int = Field(default=3)

    def create_node(self) -> RetryNode:
        return RetryNode(
            name=self.name,
            branch=self.branch._dag.model_copy(),
            max_attempts=self.max_attempts,
            **self.model_dump(exclude_defaults=True)
        )
```

### 3. Register the Node Type
```toml
# pyproject.toml
[project.entry-points.'nodes']
"retry" = "my_package.nodes:RetryNode"
```

### 4. Use Your Custom Node in Pipelines
```python
from my_package.nodes import Retry
from runnable import Pipeline, PythonTask

def main():
    pipeline = Pipeline(steps=[
        Retry(
            name="resilient_process",
            max_attempts=5,
            branch=Pipeline(steps=[
                PythonTask(name="flaky_task", function=potentially_failing_task)
            ])
        )
    ])
    pipeline.execute()
    return pipeline
```

## Integration Advantage

**🔑 Key Benefit**: Custom node types live entirely in **your codebase**, enabling domain-specific pipeline flow control.

### Complete Control & Customization

```python
# In your private repository
# company-workflows/nodes/business_nodes.py

class ApprovalGate(BaseTraversal):
    """Execute pipeline branch only after approval workflow"""
    approval_channel: str = Field(...)
    timeout_hours: int = Field(default=24)

    def create_node(self) -> ApprovalGateNode:
        # Your proprietary approval system integration
        pass
```

**Integration benefits:**

- **🔒 Business Logic**: Implement organization-specific workflow patterns and approvals
- **🏢 Domain-Specific**: Create flow control for your specific business processes
- **💼 Governance**: Built-in compliance gates, approval workflows, audit trails
- **🔧 Standardization**: Reusable flow control patterns across teams and projects

### Reusable Node Libraries

```python
# Internal package: company-runnable-nodes
from company_runnable_nodes import (
    ApprovalGate,           # Business approval workflows
    DataQualityGate,        # Quality control checkpoints
    ComplianceCheck,        # Regulatory compliance gates
    ResourceThrottle,       # Cost and resource management
)

# Teams build standardized workflows
pipeline = Pipeline(steps=[
    PythonTask(name="prep", function=prepare_data),
    DataQualityGate(name="quality_check", thresholds=quality_config),
    ApprovalGate(name="manager_approval", channel="#ml-approvals"),
    ComplianceCheck(name="sox_compliance", requirements=["data_retention", "audit_trail"]),
    PythonTask(name="deploy", function=deploy_model)
])
```

**This makes runnable a platform for building your organization's custom workflow patterns** - standardized flow control, governance, and business logic.

## Need Help?

**Custom node types involve understanding both pipeline flow control and your specific orchestration requirements.**

!!! question "Get Support"

    **We're here to help you succeed!** Building custom node types involves:

    - Understanding runnable's graph execution engine and node lifecycle
    - Implementing proper flow control and state management
    - Integration with external systems for approvals, gates, or custom logic
    - Plugin registration and pipeline composition

    **Don't hesitate to reach out:**

    - 📧 **Contact the team** for architecture guidance and integration support
    - 🤝 **Collaboration opportunities** - we're interested in supporting workflow pattern innovations
    - 📖 **Documentation feedback** - help us improve these guides based on your implementation experience

Your success with custom node types helps the entire runnable community!

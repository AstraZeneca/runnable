# SageMaker Pipeline Executor Design

**Date**: February 20, 2026
**Status**: Design Approved
**Author**: Brainstorming Session

## Overview

Add SageMaker Pipelines support as a new pipeline executor for Runnable, enabling AWS-native ML pipeline execution with cost optimization and enterprise integration capabilities.

## Goals

- **Meta Executor Addition**: Add SageMaker as another transpiler option alongside `local`, `argo`, `local-container`
- **AWS Native Integration**: Leverage SageMaker's infrastructure, IAM, and cost optimization features
- **Consistent Architecture**: Follow Runnable's established patterns for executor implementation
- **Clear Limitations**: Provide transparent boundaries for what SageMaker can and cannot handle

## Architecture

### Execution Pattern
The SageMaker executor follows Runnable's **DAG Transpilation** pattern (Pattern 2), similar to the Argo executor:

1. **Full Transpilation**: Convert entire Runnable DAG to SageMaker Pipeline definition using AWS SDK
2. **Container Execution**: All tasks run as SageMaker ProcessingSteps with containerized execution
3. **Runnable Data Flow**: Preserve existing catalog and run log mechanisms (no native SageMaker data passing)
4. **Dependency Management**: Use SageMaker's `depends_on` to recreate Runnable's DAG dependencies

### Core Implementation

```python
class SageMakerExecutor(GenericPipelineExecutor):
    service_name: str = "sagemaker"
    _should_setup_run_log_at_traversal: bool = PrivateAttr(default=False)

    def execute_graph(self, dag: Graph, map_variable=None):
        """Main transpilation entry point - follows Argo pattern"""
        # Check for unsupported node types
        unsupported_nodes = self._check_for_unsupported_nodes(dag)
        if unsupported_nodes:
            self._raise_unsupported_error(unsupported_nodes)

        # Convert DAG to SageMaker Pipeline
        steps = []
        step_map = {}

        for node in self._traverse_dag_in_order(dag):
            # Handle different node types
            match node.node_type:
                case "task" | "stub":
                    # Simple task - create single ProcessingStep
                    step = self._create_task_step(node, step_map)
                    steps.append(step)
                    step_map[node.internal_name] = step

                case "parallel":
                    # Parallel node - create fan-out/fan-in pattern
                    parallel_steps, final_step = self._handle_parallel_node(node, step_map)
                    steps.extend(parallel_steps)
                    step_map[node.internal_name] = final_step

                case "conditional":
                    # Conditional node - evaluate condition, route to matching branch
                    conditional_steps, final_step = self._handle_conditional_node(node, step_map)
                    steps.extend(conditional_steps)
                    step_map[node.internal_name] = final_step

                case "success" | "fail":
                    # Terminal nodes - create simple step
                    step = self._create_terminal_step(node, step_map)
                    steps.append(step)
                    step_map[node.internal_name] = step

                case _:
                    raise exceptions.ExecutorNotSupported(
                        f"Node type '{node.node_type}' not supported by SageMaker executor"
                    )

        # Build SageMaker Pipeline
        sg_pipeline = Pipeline(
            name=self._generate_pipeline_name(),
            steps=steps,
            sagemaker_session=self._get_session()
        )

        sg_pipeline.upsert(role_arn=self.role_arn)
        execution = sg_pipeline.start(
            parameters={
                "run_id": self._context.run_id,  # Pass run context
            }
        )

        # Optional monitoring
        if self.wait_for_completion:
            self._monitor_execution(execution)

    def trigger_node_execution(self, node, map_variable=None):
        """Runs inside SageMaker Processing job container"""
        # Storage already accessible via IAM role
        # Just execute the node using base class
        self._execute_node(node=node, map_variable=map_variable)

    def _create_task_step(self, node, step_map):
        """Create ProcessingStep for a task node with proper dependencies."""
        depends_on_steps = []
        for parent in self._get_immediate_parents(node):
            if parent.internal_name in step_map:
                depends_on_steps.append(step_map[parent.internal_name])

        processor = self._create_processor(node)
        return ProcessingStep(
            name=self._sanitize_name(node.internal_name),
            processor=processor,
            depends_on=depends_on_steps
        )

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for SageMaker (alphanumeric and hyphens, max 256 chars)."""
        sanitized = name.replace(".", "-").replace("_", "-")
        return sanitized[:256]
```

**Note:** The method is `execute_graph` (matching Argo pattern), not `execute_from_graph`. The implementation must handle different node types via match/case rather than assuming all nodes are simple tasks.

## Configuration

### Basic Configuration
```yaml
pipeline-executor:
  type: sagemaker
  config:
    # Required: AWS Infrastructure
    role_arn: "arn:aws:iam::123456789:role/SageMakerExecutionRole"
    region: "us-east-1"

    # Required: Container
    image: "my-pipeline:latest"

    # Optional: Compute Defaults
    instance_type: "ml.m5.large"
    instance_count: 1
    volume_size_gb: 30
    max_runtime_seconds: 3600

    # Optional: Execution Control
    wait_for_completion: false  # Default: don't block
```

### Advanced Configuration with Overrides
```yaml
pipeline-executor:
  type: sagemaker
  config:
    role_arn: "arn:aws:iam::123456789:role/SageMakerExecutionRole"
    image: "my-pipeline:latest"

    # Pipeline-level defaults
    instance_type: "ml.m5.large"
    volume_size_gb: 30
    max_runtime_seconds: 3600

    # Task-specific overrides (matches Argo pattern)
    overrides:
      gpu_training:
        instance_type: "ml.p3.2xlarge"
        instance_count: 1
        volume_size_gb: 100
        max_runtime_seconds: 7200

      lightweight_tasks:
        instance_type: "ml.t3.medium"
        volume_size_gb: 10
        max_runtime_seconds: 600

# Storage configuration stays in proper layers
catalog:
  type: s3
  config:
    bucket: "my-pipeline-bucket"
    prefix: "catalog/"

run_log_store:
  type: s3
  config:
    bucket: "my-pipeline-bucket"
    prefix: "logs/"
```

### Configuration Responsibilities
- **SageMaker Executor**: Compute configuration, IAM roles, instance types
- **Catalog/Run Log Store**: Storage configuration, S3 buckets, access patterns
- **IAM Role**: Must have permissions for both SageMaker execution AND storage access

## Complex Node Handling

### Parallel Node Support
SageMaker supports parallel execution using the fan-out/fan-in pattern with dummy success nodes:

```python
def _handle_parallel_node(self, parallel_node):
    """Convert ParallelNode using fan_out/fan_in pattern"""

    # 1. Fan-out step
    fan_out_step = ProcessingStep(
        name=f"{parallel_node.name}_fan_out",
        processor=self._create_processor_for_fan_out(parallel_node)
    )

    # 2. Process each branch + add success node
    branch_success_steps = []
    all_steps = [fan_out_step]

    for branch_name, branch_pipeline in parallel_node.branches.items():
        prev_step = fan_out_step

        # Process actual branch steps
        for task in branch_pipeline.tasks:
            step = ProcessingStep(
                name=f"{branch_name}_{task.name}",
                processor=self._create_processor(task),
                depends_on=[prev_step]
            )
            all_steps.append(step)
            prev_step = step

        # Add success node with proper internal name
        success_node_internal_name = f"{parallel_node.internal_name}.{branch_name}.success"
        branch_success_step = ProcessingStep(
            name=f"{branch_name}_success",
            processor=self._create_success_processor(success_node_internal_name),
            depends_on=[prev_step]
        )
        all_steps.append(branch_success_step)
        branch_success_steps.append(branch_success_step)

    # 3. Fan-in depends on ALL branch success nodes
    fan_in_step = ProcessingStep(
        name=f"{parallel_node.name}_fan_in",
        processor=self._create_processor_for_fan_in(parallel_node),
        depends_on=branch_success_steps  # Multiple dependencies supported
    )
    all_steps.append(fan_in_step)

    return all_steps, fan_in_step
```

### Conditional Node Support

SageMaker supports simple conditional execution using the same fan-out/fan-in pattern as parallel nodes. The key difference is that the fan-out step evaluates a condition to determine which branch is active, and the fan-in step uses Runnable's existing `ConditionalNode.fan_in()` to resolve status based on the selected branch.

**How it works (following the Argo pattern):**

1. **Fan-out step**: Runs `runnable fan <mode> <node>` which calls `ConditionalNode.fan_out()`. This evaluates the condition and writes the selected branch name to the run log.
2. **All branches execute**: Unlike Argo (which uses `when` parameters to skip branches), SageMaker runs all branches as ProcessingSteps. Each branch step runs `runnable execute-single-node` which checks the run log to determine if it is the active branch.
3. **Fan-in step**: Runs `ConditionalNode.fan_in()` which reads the selected branch from the run log and consolidates status from only that branch.

```python
def _handle_conditional_node(self, conditional_node, step_map):
    """Convert ConditionalNode using fan-out/fan-in pattern.

    Structurally identical to parallel handling. The condition evaluation
    and branch selection are handled by Runnable's ConditionalNode.fan_out()
    and fan_in() methods running inside the containers, not by SageMaker's
    native ConditionStep.
    """

    # 1. Fan-out step: evaluates the condition, writes selected branch to run log
    fan_out_step = ProcessingStep(
        name=f"{conditional_node.name}_fan_out",
        processor=self._create_processor_for_fan_out(conditional_node)
    )

    # 2. Process each branch + add success node
    # All branches run; fan_in resolves which was active
    branch_success_steps = []
    all_steps = [fan_out_step]

    for branch_name, branch_pipeline in conditional_node.branches.items():
        prev_step = fan_out_step

        for task in branch_pipeline.tasks:
            step = ProcessingStep(
                name=f"{branch_name}_{task.name}",
                processor=self._create_processor(task),
                depends_on=[prev_step]
            )
            all_steps.append(step)
            prev_step = step

        success_node_internal_name = f"{conditional_node.internal_name}.{branch_name}.success"
        branch_success_step = ProcessingStep(
            name=f"{branch_name}_success",
            processor=self._create_success_processor(success_node_internal_name),
            depends_on=[prev_step]
        )
        all_steps.append(branch_success_step)
        branch_success_steps.append(branch_success_step)

    # 3. Fan-in depends on ALL branch success nodes
    fan_in_step = ProcessingStep(
        name=f"{conditional_node.name}_fan_in",
        processor=self._create_processor_for_fan_in(conditional_node),
        depends_on=branch_success_steps
    )
    all_steps.append(fan_in_step)

    return all_steps, fan_in_step
```

**Trade-off: All branches execute.** SageMaker's native `ConditionStep` could skip non-selected branches, but it only supports simple predicate comparisons (`ConditionEquals`, `ConditionGreaterThan`, etc.) and cannot evaluate arbitrary Python logic. Since Runnable's `ConditionalNode` evaluates conditions via user-defined Python code, we run all branches and let Runnable's fan-in logic determine which one counts. This wastes compute on inactive branches but preserves semantic correctness and consistency with Runnable's execution model.

**Future optimization:** If a conditional has only static/simple conditions, a future phase could map them to SageMaker's native `ConditionStep` with `if_steps`/`else_steps` to avoid running inactive branches.

### Unsupported Node Types

SageMaker has significant limitations compared to Argo. The executor will reject unsupported patterns with clear error messages:

```python
def _check_for_unsupported_nodes(self, dag):
    """Reject nodes that SageMaker cannot handle"""
    unsupported = []

    for node in dag.nodes:
        # Always unsupported: dynamic execution patterns
        if node.node_type in ["map", "loop"]:
            unsupported.append(node)

        # Nested parallel detection: count dots in internal name
        elif node.internal_name.count('.') > 2:
            unsupported.append(node)

    return unsupported
```

**Supported Patterns:**
- ✅ Simple task chains: `task1 → task2 → task3`
- ✅ Basic parallel: parallel branches with fixed dependencies
- ✅ Simple conditionals: if/else with static conditions

**Unsupported Patterns:**
- ❌ Map nodes: dynamic iteration over datasets
- ❌ Loop nodes: while/for loop constructs
- ❌ Nested parallel: `step.branch.nested.nested_branch.task` (>2 dots)
- ❌ Complex conditionals: dynamic condition evaluation

**Error Handling:**
```python
if unsupported_nodes:
    nested = [n.internal_name for n in unsupported_nodes if n.internal_name.count('.') > 2]
    other = [f"{n.internal_name} ({n.node_type})" for n in unsupported_nodes if n.internal_name.count('.') <= 2]

    msg = "SageMaker executor limitations:\n"
    if nested:
        msg += f"- Nested structures not supported: {nested}\n"
    if other:
        msg += f"- Unsupported node types: {other}\n"
    msg += "Use 'argo' executor for complex workflows."

    raise exceptions.ExecutorNotSupported(msg)
```

## Data Flow Architecture

### Storage Integration
- **Unified S3 Configuration**: Extends existing S3 catalog configuration to work with SageMaker
- **No Native SageMaker Data Flow**: All parameter and artifact flow handled by Runnable's existing mechanisms
- **Shared Storage**: All SageMaker Processing jobs access same S3 paths for catalog and run logs
- **IAM-based Access**: IAM role provides storage permissions, no explicit input/output definitions

### Container Requirements
- Container must have Runnable installed with same version
- Container must have access to pipeline code and dependencies
- IAM role must have permissions for S3 (catalog/logs) access
- All steps run: `runnable execute-single-node <node_internal_name>`

## Implementation Plan

### Phase 1: Basic Implementation
1. Create `SageMakerExecutor` class extending `GenericPipelineExecutor`
2. Implement basic DAG transpilation for simple task chains
3. Add configuration schema with IAM role and compute settings
4. Implement unsupported node detection and clear error messages

### Phase 2: Parallel Support
1. Implement fan-out/fan-in pattern for `ParallelNode`
2. Add dummy success node creation
3. Test complex dependency management

### Phase 3: Advanced Features
1. Add resource override system (matching Argo pattern)
2. Implement optional monitoring and wait functionality
3. Add cost optimization features (spot instances, instance selection)

## Testing Strategy

1. **Unit Tests**: Configuration validation, node type detection, dependency building
2. **Integration Tests**: End-to-end pipeline execution with mock SageMaker
3. **Limitation Tests**: Verify proper rejection of unsupported patterns
4. **Compatibility Tests**: Same pipeline works across local/argo/sagemaker executors

## Risks and Mitigations

### Risk: Complex Dependency Management
- **Mitigation**: Follow proven Argo patterns, leverage SageMaker's multi-step `depends_on`

### Risk: Storage Access Coordination
- **Mitigation**: Use IAM roles for permission-based access, avoid complex volume mounting

### Risk: User Confusion About Limitations
- **Mitigation**: Provide clear error messages with specific guidance to use Argo for complex patterns

## TODO: Design Decisions to Resolve

### SageMaker Native Features Integration
**Issue**: SageMaker has native caching and retry capabilities that may conflict with Runnable's systems.

**SageMaker Native Capabilities:**
- **Step Caching**: Input-based caching with S3 storage for step result reuse
- **Retry Policies**: Built-in job-level and pipeline-level retry with backoff

**Decision Needed**: How should these integrate with Runnable's memoization and retry systems?

**Options:**
1. **Runnable-managed** (Recommended): Disable SageMaker caching/retries, use Runnable's systems for consistency across executors
2. **SageMaker-native**: Use SageMaker's systems, disable Runnable's for SageMaker executor
3. **Hybrid**: Allow configuration to choose which system handles caching/retries

**Implications:**
- **Consistency**: Runnable-managed ensures same behavior across local/Argo/SageMaker
- **Performance**: SageMaker-native may be more efficient for AWS-only workflows
- **Complexity**: Hybrid approach increases configuration complexity

**Action Required**: Decide on approach before implementation and document in configuration section.

### Testing Strategy for AWS Integration
**Issue**: SageMaker executor requires AWS resources for complete validation, but testing without AWS credentials is challenging.

**Testing Approach Options:**

1. **Mock-Based Testing** (Current plan):
   - Pros: Fast, no AWS needed, can test error conditions
   - Cons: Doesn't validate actual SageMaker integration, may miss SDK changes

2. **SageMaker Local Mode**:
   - Uses `sagemaker.local.LocalSession()` to run processing jobs locally
   - Pros: Real SageMaker SDK usage, no AWS credentials needed
   - Cons: May not support all Pipeline features, doesn't test AWS-specific issues

3. **AWS Integration Tests**:
   - Real AWS resources with GitHub Actions/CI credentials
   - Pros: Full end-to-end validation, tests IAM roles and S3 access
   - Cons: Requires AWS account setup, costs money, slower execution

**Recommended Strategy**: Layered approach:
- Unit Tests: Mock-based for fast feedback
- Integration Tests: SageMaker Local Mode for better coverage
- E2E Tests: Real AWS for full validation (optional)

**Action Required**: Determine which testing levels are needed and update implementation plan with appropriate test infrastructure setup.

### Output File Generation and Validation
**Issue**: Argo generates YAML files that can be inspected, linted, and validated before submission. Should SageMaker executor support similar output file generation?

**Questions to Resolve:**
1. Does AWS publish a JSON Schema for SageMaker Pipeline definitions?
2. Is there an AWS CLI command for validating pipeline definitions (like `aws sagemaker validate-pipeline`)?
3. Does the SageMaker SDK have a local validation or dry-run mode?
4. If no validation exists, is output file generation still valuable for debugging/inspection?

**Options:**
1. **SDK-only submission**: Skip output file generation, always submit directly via SDK
2. **Output file with inspection**: Generate JSON for debugging but no validation
3. **Output file with validation**: If validation mechanism exists, support full lint workflow

**Action Required**: Research SageMaker Pipeline validation capabilities before implementing output file generation.

### Run Context and Configuration Propagation
**Issue**: Each SageMaker ProcessingJob needs access to Runnable's execution context, but the design doesn't specify how this is propagated.

**Required Context:**
- `run_id`: Unique identifier for the pipeline execution
- Configuration file path or serialized configuration
- Pipeline parameters and initial context

**Argo's Approach:**
```python
# Argo uses workflow parameters
_run_id_as_parameter: str = PrivateAttr(default="{{workflow.parameters.run_id}}")
```

**Possible SageMaker Approaches:**
1. **Environment variables**: Pass run_id and config via container environment
2. **S3 context file**: Write context to S3, containers read at startup
3. **SageMaker Pipeline parameters**: Use native parameter passing for run_id

**Action Required**: Design and document the context propagation mechanism.

### SDK Dependency Strategy
**Issue**: Adding `sagemaker>=2.120.0` as a core dependency significantly bloats the base package. The SageMaker SDK has many transitive dependencies.

**Recommendation**: Make SageMaker an optional dependency like Kubernetes:
```toml
[project.optional-dependencies]
sagemaker = ["sagemaker>=2.120.0"]
```

**Usage:**
```bash
pip install runnable[sagemaker]
```

**Action Required**: Decide on dependency strategy and update pyproject.toml accordingly.

### VPC and Network Configuration
**Issue**: SageMaker ProcessingJobs often need VPC configuration to access private resources (databases, internal APIs, etc.).

**Required Configuration:**
```yaml
pipeline-executor:
  type: sagemaker
  config:
    # ... existing config ...
    vpc_config:
      subnets:
        - "subnet-xxx"
        - "subnet-yyy"
      security_group_ids:
        - "sg-xxx"
```

**Action Required**: Add VPC configuration to schema and implementation.

### AWS Secrets Manager Integration
**Issue**: Argo has `secret_from_k8s` for Kubernetes secrets. SageMaker should support AWS Secrets Manager for secure credential handling.

**Proposed Configuration:**
```yaml
pipeline-executor:
  type: sagemaker
  config:
    secrets_from_aws:
      - environment_variable: DB_PASSWORD
        secret_arn: "arn:aws:secretsmanager:us-east-1:xxx:secret:db-password"
        secret_key: "password"
```

**Action Required**: Design secrets integration pattern.

### Resource Tagging
**Issue**: AWS resources should be tagged for cost tracking and organization. No tagging configuration in current design.

**Proposed Configuration:**
```yaml
pipeline-executor:
  type: sagemaker
  config:
    tags:
      project: "my-ml-project"
      team: "data-science"
      environment: "production"
```

**Action Required**: Add tags to all SageMaker resources created.

## Success Metrics

1. **Drop-in Compatibility**: Existing simple pipelines work with just executor configuration change
2. **Clear Boundaries**: Users understand when to use SageMaker vs Argo
3. **AWS Integration**: Seamless integration with AWS IAM, S3, and cost management
4. **Performance**: Efficient execution of ML workloads on appropriate SageMaker instance types

## Related Work

- **Argo Executor**: Reference implementation for DAG transpilation patterns
- **Local Container Executor**: Reference for container-based execution patterns
- **S3 Catalog**: Existing S3 integration patterns for storage access

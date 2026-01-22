"""
Airflow DAG Factory for Runnable pipelines.

This module provides AirflowDagFactory which converts Runnable pipelines to Airflow DAGs
at import time, following the dag-factory pattern.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field, PrivateAttr

from extensions.pipeline_executor import GenericPipelineExecutor
from runnable import defaults, exceptions
from runnable.defaults import IterableParameterModel
from runnable.graph import search_node_by_internal_name
from runnable.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)

# Guard imports for optional Airflow dependency
try:
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator
    from airflow.providers.docker.operators.docker import DockerOperator
    from airflow.utils.task_group import TaskGroup
    from airflow.utils.trigger_rule import TriggerRule

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    if TYPE_CHECKING:
        from airflow import DAG
        from airflow.operators.empty import EmptyOperator
        from airflow.operators.python import BranchPythonOperator
        from airflow.providers.docker.operators.docker import DockerOperator
        from airflow.utils.task_group import TaskGroup
        from airflow.utils.trigger_rule import TriggerRule


def _check_airflow_available():
    """Raise ImportError if Airflow is not available."""
    if not AIRFLOW_AVAILABLE:
        raise ImportError(
            "Airflow is not installed. Install it with: "
            "pip install 'runnable[airflow]' or "
            "pip install apache-airflow apache-airflow-providers-docker"
        )


class AirflowExecutor(GenericPipelineExecutor):
    """
    Pipeline executor for Airflow DAG generation and container execution.

    This executor serves two purposes:
    1. DAG Generation: Override execute_graph to build Airflow DAG structures
    2. Container Execution: Handle run log setup when running inside DockerOperator

    Configuration:

    ```yaml
    pipeline-executor:
      type: airflow
      config:
        image: my-runnable-image:latest
        docker_url: unix://var/run/docker.sock
        network_mode: bridge
        auto_remove: success
        mount_tmp_dir: false
        parameters_file: /path/to/parameters.yaml
        config_file: /path/to/config.yaml
        volumes:
          - /host/run_logs:/tmp/run_logs
          - /host/catalog:/tmp/catalog
        environment:
          KEY: value
    ```
    """

    service_name: str = "airflow"

    # Docker operator configuration
    image: str = ""
    docker_url: str = Field(default="unix://var/run/docker.sock")
    network_mode: str = Field(default="bridge")
    auto_remove: str = Field(default="success")
    mount_tmp_dir: bool = Field(default=False)
    volumes: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)

    # Runnable configuration - explicit paths instead of auto-discovery
    parameters_file: Optional[str] = Field(default=None)
    config_file: Optional[str] = Field(default=None)

    _should_setup_run_log_at_traversal: bool = PrivateAttr(default=False)

    # Runtime state for DAG generation
    _dag: Any = PrivateAttr(default=None)
    _pipeline_file: str = PrivateAttr(default="")
    _effective_image: str = PrivateAttr(default="")

    # ═══════════════════════════════════════════════════════════════
    # Container Execution Methods - used when running inside DockerOperator
    # ═══════════════════════════════════════════════════════════════

    def execute_node(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        Execute a node inside an Airflow DockerOperator container.

        This method:
        1. Sets up the run log (creates if first step, reuses if exists)
        2. Creates or retrieves step log
        3. Executes the node
        4. Handles failure scenarios
        """
        self._set_up_run_log()

        try:
            # This should only happen during a retry
            step_log = self._context.run_log_store.get_step_log(
                node._get_step_log_name(iter_variable), self._context.run_id
            )
            assert self._context.is_retry
        except exceptions.StepLogNotFoundError:
            step_log = self._context.run_log_store.create_step_log(
                node.name, node._get_step_log_name(iter_variable)
            )

            step_log.step_type = node.node_type
            step_log.status = defaults.PROCESSING
            self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        self._execute_node(node=node, iter_variable=iter_variable)

        # Raise exception if the step failed
        step_log = self._context.run_log_store.get_step_log(
            node._get_step_log_name(iter_variable), self._context.run_id
        )
        if step_log.status == defaults.FAIL:
            run_log = self._context.run_log_store.get_run_log_by_id(
                self._context.run_id
            )
            run_log.status = defaults.FAIL
            self._context.run_log_store.put_run_log(run_log)
            raise Exception(f"Step {node.name} failed")

        # This makes the fail node execute if we are heading that way.
        self._implicitly_fail(node, iter_variable)

    def fan_out(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        Handle fan-out for composite nodes (parallel, map, conditional).

        This can be the first step of the graph if the pipeline starts with
        a composite node, so it must set up the run log.
        """
        self._set_up_run_log()
        super().fan_out(node, iter_variable)

    def fan_in(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """Handle fan-in for composite nodes."""
        super().fan_in(node, iter_variable)

    def _implicitly_fail(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        Execute fail node if the next node is a fail node.

        This is needed for orchestrators where the fail node execution
        must be triggered explicitly after a step fails.
        """
        assert self._context.dag
        _, current_branch = search_node_by_internal_name(
            dag=self._context.dag, internal_name=node.internal_name
        )
        _, next_node_name = self._get_status_and_next_node_name(
            node, current_branch, iter_variable=iter_variable
        )
        if next_node_name:
            # Terminal nodes do not have next node name
            next_node = current_branch.get_node_by_name(next_node_name)

            if next_node.node_type == defaults.FAIL:
                self.execute_node(next_node, iter_variable=iter_variable)

    # ═══════════════════════════════════════════════════════════════
    # DAG Generation Methods - used by AirflowDagFactory
    # ═══════════════════════════════════════════════════════════════

    def execute_graph(
        self,
        dag: Any,
        iter_variable: Optional[IterableParameterModel] = None,
    ) -> tuple[Any, Any]:
        """
        Build Airflow tasks from a Runnable graph.

        This overrides the base execute_graph to generate Airflow DAG structures
        instead of executing nodes directly.

        Args:
            dag: The Runnable graph to convert
            iter_variable: Optional iteration variable for map nodes

        Returns:
            Tuple of (first_task, last_task) for dependency chaining
        """
        _check_airflow_available()
        from extensions.nodes.conditional import ConditionalNode
        from extensions.nodes.map import MapNode
        from extensions.nodes.parallel import ParallelNode

        current_node_name = dag.start_at
        first_task = None
        previous_task = None
        last_task = None
        is_first_node = True

        while current_node_name:
            node = dag.get_node_by_name(current_node_name)
            task_id = self._sanitize_task_id(node.internal_name)

            match node.node_type:
                case "task" | "stub":
                    task = self._create_docker_task(
                        node=node,
                        task_id=task_id,
                        iter_variable=iter_variable,
                        is_first_node=is_first_node,
                    )

                case "success":
                    task = EmptyOperator(
                        task_id=task_id,
                        trigger_rule=TriggerRule.ALL_SUCCESS,
                    )

                case "fail":
                    task = EmptyOperator(
                        task_id=task_id,
                        trigger_rule=TriggerRule.ONE_FAILED,
                    )

                case "parallel":
                    assert isinstance(node, ParallelNode)
                    task = self._create_parallel_task_group(
                        node=node,
                        task_id=task_id,
                        iter_variable=iter_variable,
                        is_first_node=is_first_node,
                    )

                case "map":
                    assert isinstance(node, MapNode)
                    task = self._create_map_task_group(
                        node=node,
                        task_id=task_id,
                        iter_variable=iter_variable,
                        is_first_node=is_first_node,
                    )

                case "conditional":
                    assert isinstance(node, ConditionalNode)
                    task = self._create_conditional_task_group(
                        node=node,
                        task_id=task_id,
                        iter_variable=iter_variable,
                        is_first_node=is_first_node,
                    )

                case _:
                    raise ValueError(f"Unsupported node type: {node.node_type}")

            # Track first task
            if first_task is None:
                first_task = task

            # Chain dependencies
            if previous_task is not None:
                previous_task >> task

            previous_task = task
            last_task = task
            is_first_node = False

            # Move to next node (terminal nodes have no next)
            if node.node_type in ["success", "fail"]:
                break

            try:
                current_node_name = node._get_next_node()
            except Exception:
                break

        return first_task, last_task

    def _create_docker_task(
        self,
        node: Any,
        task_id: str,
        iter_variable: Optional[IterableParameterModel] = None,
        is_first_node: bool = False,
    ) -> "DockerOperator":
        """Create a DockerOperator for a task node."""
        command = self._build_execute_command(
            node=node,
            iter_variable=iter_variable,
            init_run_log=is_first_node,
        )

        return DockerOperator(
            task_id=task_id,
            image=self._effective_image,
            command=command,
            docker_url=self.docker_url,
            network_mode=self.network_mode,
            auto_remove=self.auto_remove,
            mount_tmp_dir=self.mount_tmp_dir,
            mounts=self._get_volume_mounts(),
            environment=self.environment,
        )

    def _build_execute_command(
        self,
        node: Any,
        iter_variable: Optional[IterableParameterModel] = None,
        init_run_log: bool = False,
    ) -> list[str]:
        """Build runnable execute-single-node command."""
        command = [
            "runnable",
            "execute-single-node",
            "{{ run_id }}",
            self._pipeline_file,
            node._command_friendly_name(),
            "--mode",
            "python",
        ]

        if self.config_file:
            command.extend(["--config-file", self.config_file])

        if init_run_log:
            command.append("--init-run-log")
            if self.parameters_file:
                command.extend(["--parameters-file", self.parameters_file])

        if iter_variable:
            iter_json = iter_variable.model_dump_json()
            command.extend(["--iter-variable", iter_json])

        return command

    def _create_parallel_task_group(
        self,
        node: Any,
        task_id: str,
        iter_variable: Optional[IterableParameterModel] = None,
        is_first_node: bool = False,
    ) -> "TaskGroup":
        """Create a TaskGroup for a parallel node."""
        with TaskGroup(group_id=task_id) as parallel_group:
            fan_out = self._create_fan_task(
                node=node,
                task_id=f"{task_id}_fan_out",
                mode="out",
                iter_variable=iter_variable,
                init_run_log=is_first_node,
            )

            branch_ends = []
            for branch_name, branch_graph in node.branches.items():
                branch_task_id = self._sanitize_task_id(branch_name)
                with TaskGroup(group_id=branch_task_id):
                    first, last = self.execute_graph(
                        dag=branch_graph,
                        iter_variable=iter_variable,
                    )
                    fan_out >> first
                    branch_ends.append(last)

            fan_in = self._create_fan_task(
                node=node,
                task_id=f"{task_id}_fan_in",
                mode="in",
                iter_variable=iter_variable,
                trigger_rule=TriggerRule.ALL_DONE,
            )

            for end in branch_ends:
                end >> fan_in

        return parallel_group

    def _create_map_task_group(
        self,
        node: Any,
        task_id: str,
        iter_variable: Optional[IterableParameterModel] = None,
        is_first_node: bool = False,
    ) -> "TaskGroup":
        """Create a TaskGroup for a map node with dynamic task mapping."""
        iterate_as = node.iterate_as

        with TaskGroup(group_id=task_id) as map_group:
            fan_out = self._create_fan_task(
                node=node,
                task_id=f"{task_id}_fan_out",
                mode="out",
                iter_variable=iter_variable,
                init_run_log=is_first_node,
                do_xcom_push=True,
            )

            # Build branch command with Jinja template for iteration value
            branch_command = self._build_branch_command_for_map(
                node=node,
                iterate_as=iterate_as,
                task_id=task_id,
                iter_variable=iter_variable,
            )

            branch_task = DockerOperator.partial(
                task_id=f"{task_id}_branch",
                image=self._effective_image,
                docker_url=self.docker_url,
                network_mode=self.network_mode,
                auto_remove=self.auto_remove,
                mount_tmp_dir=self.mount_tmp_dir,
                mounts=self._get_volume_mounts(),
                environment=self.environment,
            ).expand(command=[branch_command])

            fan_in = self._create_fan_task(
                node=node,
                task_id=f"{task_id}_fan_in",
                mode="in",
                iter_variable=iter_variable,
                trigger_rule=TriggerRule.ALL_DONE,
            )

            fan_out >> branch_task >> fan_in

        return map_group

    def _build_branch_command_for_map(
        self,
        node: Any,
        iterate_as: str,
        task_id: str,
        iter_variable: Optional[IterableParameterModel] = None,
    ) -> str:
        """Build command for map branch execution with Jinja templating."""
        branch_graph = node.branch
        branch_start = branch_graph.start_at
        branch_node = branch_graph.get_node_by_name(branch_start)

        # Build iter_variable with Jinja template
        map_var = {}
        if iter_variable and iter_variable.map_variable:
            for key, value in iter_variable.map_variable.items():
                map_var[key] = {"value": value.value}

        map_var[iterate_as] = {
            "value": f"{{{{ ti.xcom_pull(task_ids='{task_id}_fan_out')[ti.map_index] }}}}"
        }

        iter_variable_json = json.dumps({"map_variable": map_var})

        command = (
            f"runnable execute-single-node {{{{ run_id }}}} "
            f"{self._pipeline_file} {branch_node._command_friendly_name()} "
            f"--mode python "
            f"--iter-variable '{iter_variable_json}'"
        )

        if self.config_file:
            command += f" --config-file {self.config_file}"

        return command

    def _create_conditional_task_group(
        self,
        node: Any,
        task_id: str,
        iter_variable: Optional[IterableParameterModel] = None,
        is_first_node: bool = False,
    ) -> "TaskGroup":
        """Create a TaskGroup for a conditional node."""
        with TaskGroup(group_id=task_id) as conditional_group:
            fan_out = self._create_fan_task(
                node=node,
                task_id=f"{task_id}_fan_out",
                mode="out",
                iter_variable=iter_variable,
                init_run_log=is_first_node,
                do_xcom_push=True,
            )

            def branch_selector(**context):
                ti = context["ti"]
                selected_branch = ti.xcom_pull(task_ids=f"{task_id}_fan_out")
                return f"{task_id}.{selected_branch}"

            branch_op = BranchPythonOperator(
                task_id=f"{task_id}_branch_selector",
                python_callable=branch_selector,
            )

            fan_out >> branch_op

            branch_ends = []
            for branch_name, branch_graph in node.branches.items():
                branch_task_id = self._sanitize_task_id(branch_name)
                with TaskGroup(group_id=branch_task_id):
                    first, last = self.execute_graph(
                        dag=branch_graph,
                        iter_variable=iter_variable,
                    )
                    branch_op >> first
                    branch_ends.append(last)

            fan_in = self._create_fan_task(
                node=node,
                task_id=f"{task_id}_fan_in",
                mode="in",
                iter_variable=iter_variable,
                trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
            )

            for end in branch_ends:
                end >> fan_in

        return conditional_group

    def _create_fan_task(
        self,
        node: Any,
        task_id: str,
        mode: str,
        iter_variable: Optional[IterableParameterModel] = None,
        init_run_log: bool = False,
        trigger_rule: "TriggerRule" = None,
        do_xcom_push: bool = False,
    ) -> "DockerOperator":
        """Create a fan-out or fan-in DockerOperator task."""
        command = self._build_fan_command(
            node=node,
            mode=mode,
            iter_variable=iter_variable,
            init_run_log=init_run_log,
        )

        kwargs = {
            "task_id": task_id,
            "image": self._effective_image,
            "command": command,
            "docker_url": self.docker_url,
            "network_mode": self.network_mode,
            "auto_remove": self.auto_remove,
            "mount_tmp_dir": self.mount_tmp_dir,
            "mounts": self._get_volume_mounts(),
            "environment": self.environment,
        }

        if trigger_rule:
            kwargs["trigger_rule"] = trigger_rule

        if do_xcom_push:
            kwargs["do_xcom_push"] = True

        return DockerOperator(**kwargs)

    def _build_fan_command(
        self,
        node: Any,
        mode: str,
        iter_variable: Optional[IterableParameterModel] = None,
        init_run_log: bool = False,
    ) -> list[str]:
        """Build runnable fan command."""
        command = [
            "runnable",
            "fan",
            "{{ run_id }}",
            node._command_friendly_name(),
            self._pipeline_file,
            mode,
            "--mode",
            "python",
        ]

        if self.config_file:
            command.extend(["--config-file", self.config_file])

        if init_run_log:
            command.append("--init-run-log")
            if self.parameters_file:
                command.extend(["--parameters-file", self.parameters_file])

        if iter_variable:
            iter_json = iter_variable.model_dump_json()
            command.extend(["--iter-variable", iter_json])

        return command

    def _get_volume_mounts(self) -> list:
        """Convert volume strings to Docker mount objects."""
        if not AIRFLOW_AVAILABLE:
            return []

        from docker.types import Mount

        mounts = []
        for volume in self.volumes:
            parts = volume.split(":")
            if len(parts) >= 2:
                source = parts[0]
                target = parts[1]
                read_only = len(parts) > 2 and parts[2] == "ro"
                mounts.append(
                    Mount(
                        target=target,
                        source=source,
                        type="bind",
                        read_only=read_only,
                    )
                )
        return mounts

    def _sanitize_task_id(self, name: str) -> str:
        """Sanitize a name to be a valid Airflow task ID."""
        sanitized = name.replace(" ", "_").replace(".", "_")
        sanitized = "".join(c for c in sanitized if c.isalnum() or c in "_-")
        return sanitized


class AirflowDagFactory(BaseModel):
    """
    Factory for creating Airflow DAGs from Runnable pipelines.

    This factory loads Runnable pipeline definitions and converts them to
    native Airflow DAG objects using AirflowExecutor for task generation.

    Example usage:
        ```python
        # airflow/dags/my_pipelines.py
        from runnable.airflow import AirflowDagFactory

        factory = AirflowDagFactory(
            image="my-runnable-image:latest",
        )

        traversal_dag = factory.create_dag(
            "examples/02-sequential/traversal.py",
            dag_id="traversal-pipeline",
        )
        ```

    Configuration:
        image: Docker image containing the Runnable pipeline code
        docker_url: Docker daemon URL (default: unix://var/run/docker.sock)
        network_mode: Docker network mode (default: bridge)
        auto_remove: Auto-remove container policy (default: success)
        mount_tmp_dir: Whether to mount tmp directory (default: False)
        volumes: Volume mounts for run logs and catalog
        environment: Environment variables for containers
        default_args: Default arguments for Airflow DAG
        schedule: DAG schedule interval
        catchup: Whether to backfill (default: False)
        tags: DAG tags
        config_file: Config file for runnable services (run-log-store, catalog, etc.)
        parameters_file: Parameters file for pipeline execution
    """

    # Docker operator defaults (passed to AirflowExecutor)
    image: str
    docker_url: str = Field(default="unix://var/run/docker.sock")
    network_mode: str = Field(default="bridge")
    auto_remove: str = Field(default="success")
    mount_tmp_dir: bool = Field(default=False)
    volumes: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)

    # Airflow DAG defaults
    default_args: dict[str, Any] = Field(default_factory=dict)
    schedule: Optional[str] = Field(default=None)
    catchup: bool = Field(default=False)
    tags: list[str] = Field(default_factory=list)

    # Runnable configuration files
    config_file: Optional[str] = Field(default=None)
    parameters_file: Optional[str] = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        _check_airflow_available()
        super().__init__(**data)

    def create_dag(
        self,
        pipeline_file: str,
        dag_id: str,
        image: Optional[str] = None,
        schedule: Optional[str] = None,
        **dag_kwargs,
    ) -> "DAG":
        """
        Create an Airflow DAG from a Runnable pipeline file.

        Args:
            pipeline_file: Path to the Runnable pipeline Python file
            dag_id: Unique identifier for the DAG
            image: Override the default Docker image for this DAG
            schedule: Override the default schedule for this DAG
            **dag_kwargs: Additional arguments passed to DAG constructor

        Returns:
            An Airflow DAG object
        """
        import runpy

        from runnable import context
        from runnable.sdk import AsyncPipeline, Pipeline

        # Load the pipeline by executing the Python file
        # Use a different run_name to avoid triggering if __name__ == "__main__" block
        result = runpy.run_path(pipeline_file, run_name="__airflow_dag_builder__")

        # Find the Pipeline object in the result
        # The main() function typically returns the pipeline
        pipeline = None

        # Set up a minimal context so that pipeline.execute() returns early
        # (it checks if context is set and returns without executing)
        # We use a simple object as a marker - any non-None value works
        context.set_run_context(object())  # type: ignore

        try:
            # First, try calling main() if it exists
            if "main" in result and callable(result["main"]):
                pipeline = result["main"]()
            else:
                # Otherwise, look for a Pipeline instance in the module namespace
                for value in result.values():
                    if isinstance(value, (Pipeline, AsyncPipeline)):
                        pipeline = value
                        break
        finally:
            # Clear the context after loading
            context.set_run_context(None)

        if pipeline is None:
            raise ValueError(
                f"Could not find Pipeline in {pipeline_file}. "
                "Ensure the file defines a main() function that returns a Pipeline, "
                "or has a Pipeline instance in module scope."
            )

        runnable_graph = pipeline.return_dag()

        # Resolve effective configuration
        effective_image = image or self.image
        effective_schedule = schedule if schedule is not None else self.schedule

        # Merge DAG kwargs with defaults
        dag_config = {
            "dag_id": dag_id,
            "default_args": {**self.default_args, **dag_kwargs.pop("default_args", {})},
            "schedule": effective_schedule,
            "catchup": dag_kwargs.pop("catchup", self.catchup),
            "tags": dag_kwargs.pop("tags", self.tags),
            **dag_kwargs,
        }

        # Create the Airflow DAG
        airflow_dag = DAG(**dag_config)

        # Create executor with Docker configuration
        executor = AirflowExecutor(
            image=effective_image,
            docker_url=self.docker_url,
            network_mode=self.network_mode,
            auto_remove=self.auto_remove,
            mount_tmp_dir=self.mount_tmp_dir,
            volumes=self.volumes,
            environment=self.environment,
            config_file=self.config_file,
            parameters_file=self.parameters_file,
        )

        # Set runtime state for DAG generation
        executor._pipeline_file = pipeline_file
        executor._effective_image = effective_image

        # Build DAG tasks from the pipeline graph using executor
        with airflow_dag:
            executor.execute_graph(dag=runnable_graph)

        return airflow_dag

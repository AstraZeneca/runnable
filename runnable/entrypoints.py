import json
import logging
from typing import Optional

import runnable.context as context
from runnable import console, defaults, graph, nodes, utils

logger = logging.getLogger(defaults.LOGGER_NAME)


def execute_pipeline_yaml_spec(
    pipeline_file: str,
    configuration_file: str = "",
    tag: str = "",
    run_id: str = "",
    parameters_file: str = "",
):
    # pylint: disable=R0914,R0913
    """
    The entry point to runnable execution for any YAML based spec.
    The result could:
        - Execution of the pipeline if its local executor
        - Rendering of the spec in the case of non local executor
    """

    service_configurations = context.ServiceConfigurations(
        configuration_file=configuration_file,
        execution_context=context.ExecutionContext.PIPELINE,
    )
    configurations = {
        "pipeline_definition_file": pipeline_file,
        "parameters_file": parameters_file,
        "tag": tag,
        "run_id": run_id,
        "execution_mode": context.ExecutionMode.YAML,
        "configuration_file": configuration_file,
        **service_configurations.services,
    }

    logger.info("Resolved configurations:")
    logger.info(json.dumps(configurations, indent=4))

    run_context = context.PipelineContext.model_validate(configurations)

    run_context.execute()

    run_context.pipeline_executor.send_return_code()


def execute_single_node(
    configuration_file: str,
    pipeline_file: str,
    step_name: str,
    map_variable: str,
    mode: str,
    run_id: str,
    tag: str = "",
    parameters_file: str = "",
):
    """
    This entry point is triggered during the execution of the pipeline
        - non local execution environments

    The mode defines how the pipeline spec is provided to the runnable
        - yaml
        - python
    """

    service_configurations = context.ServiceConfigurations(
        configuration_file=configuration_file,
        execution_context=context.ExecutionContext.PIPELINE,
    )
    configurations = {
        "pipeline_definition_file": pipeline_file,
        "parameters_file": parameters_file,
        "tag": tag,
        "run_id": run_id,
        "execution_mode": mode,
        "configuration_file": configuration_file,
        **service_configurations.services,
    }

    logger.info("Resolved configurations:")
    logger.info(json.dumps(configurations, indent=4))

    run_context = context.PipelineContext.model_validate(configurations)
    assert run_context.dag

    map_variable_dict = utils.json_to_ordered_dict(map_variable)

    step_internal_name = nodes.BaseNode._get_internal_name_from_command_name(step_name)
    node_to_execute, _ = graph.search_node_by_internal_name(
        run_context.dag, step_internal_name
    )

    logger.info("Executing the single node of : %s", node_to_execute)

    run_context.pipeline_executor.execute_node(
        node=node_to_execute, map_variable=map_variable_dict
    )

    # run_context.pipeline_executor.send_return_code()


def execute_single_branch(
    branch_name: str,
    branch: graph.Graph,
    run_context: context.PipelineContext,
    map_variable: Optional[dict] = None,
):
    """
    Execute a single branch in a separate process for parallel execution.

    This function is designed to be called by multiprocessing to execute
    individual branches of parallel and map nodes.

    Args:
        branch_name (str): The name/identifier of the branch
        branch (Graph): The graph object representing the branch to execute
        run_context (PipelineContext): The pipeline execution context
        map_variable (dict, optional): Map variables for the execution
    """
    # Set up branch-specific logging
    _setup_branch_logging(branch_name)

    logger.info(f"Executing single branch: {branch_name}")

    try:
        # Set the context in the global context module to ensure all components have access
        import runnable.context as context_module

        context_module.run_context = run_context

        # Execute the branch using the pipeline executor
        run_context.pipeline_executor.execute_graph(branch, map_variable=map_variable)
        logger.info(f"Branch {branch_name} completed successfully")
        return True
    except Exception as e:
        logger.error(f"Branch {branch_name} failed with error: {e}")
        return False


def _setup_branch_logging(branch_name: str):
    """
    Set up branch-specific logging with prefixes to organize parallel execution logs.

    Args:
        branch_name (str): The name of the branch to use as a prefix
    """
    import logging
    import sys

    # Create a custom formatter that includes the branch name
    class BranchFormatter(logging.Formatter):
        def __init__(self, branch_name: str):
            self.branch_name = branch_name
            # Extract just the meaningful part of the branch name for cleaner display
            self.display_name = self._get_display_name(branch_name)
            super().__init__()

        def _get_display_name(self, branch_name: str) -> str:
            """Extract a clean display name from the full branch name."""
            # For parallel branches like 'parallel_step.branch1', use 'branch1'
            # For map branches like 'map_state.1', use 'iter:1'
            if "." in branch_name:
                parts = branch_name.split(".")
                if len(parts) >= 2:
                    last_part = parts[-1]
                    # Check if it looks like a map iteration (numeric)
                    if last_part.isdigit():
                        return f"iter:{last_part}"
                    else:
                        return last_part
            return branch_name

        def format(self, record):
            # Add branch prefix to the message
            original_msg = record.getMessage()
            record.msg = f"[{self.display_name}] {original_msg}"
            record.args = ()

            # Use a simple format for clarity
            return f"{record.levelname}:{record.msg}"

    # Get the root logger and add our custom formatter
    root_logger = logging.getLogger()

    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        if hasattr(handler, "_branch_handler"):
            root_logger.removeHandler(handler)

    # Create a new handler with branch-specific formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(BranchFormatter(branch_name))
    handler._branch_handler = True  # type: ignore  # Mark it as our custom handler
    handler.setLevel(logging.INFO)

    # Add the handler to the root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def execute_job_non_local(
    job_definition_file: str,
    configuration_file: str = "",
    tag: str = "",
    run_id: str = "",
    parameters_file: str = "",
):
    service_configurations = context.ServiceConfigurations(
        configuration_file=configuration_file,
        execution_context=context.ExecutionContext.JOB,
    )
    configurations = {
        "job_definition_file": job_definition_file,
        "parameters_file": parameters_file,
        "tag": tag,
        "run_id": run_id,
        "configuration_file": configuration_file,
        **service_configurations.services,
    }

    logger.info("Resolved configurations:")
    logger.info(json.dumps(configurations, indent=4))

    run_context = context.JobContext.model_validate(configurations)
    assert run_context.job

    logger.info("Executing the job in non-local mode")
    logger.info("Job to execute: %s", run_context.job)

    try:
        run_context.job_executor.execute_job(
            run_context.job,
            catalog_settings=run_context.catalog_settings,
        )
    finally:
        console.print("Job execution completed. Sending return code...")

    run_context.job_executor.send_return_code()


def fan(
    configuration_file: str,
    pipeline_file: str,
    step_name: str,
    mode: str,
    in_or_out: str,
    map_variable: str,
    run_id: str,
    tag: str = "",
    parameters_file: str = "",
):
    """
    The entry point to either fan in or out for a composite node. Only 3rd party orchestrators should use this.

    It should have similar set up of configurations to execute because orchestrator modes can initiate the execution.

    Args:
        configuration_file (str): The configuration file.
        mode: in or out
        step_name : The name of the step to execute in dot path convention
        pipeline_file (str): The config/dag file
        run_id (str): The run id of the run.
        tag (str): If a tag is provided at the run time
        parameters_file (str): The parameters being sent in to the application

    """
    service_configurations = context.ServiceConfigurations(
        configuration_file=configuration_file,
        execution_context=context.ExecutionContext.PIPELINE,
    )
    configurations = {
        "pipeline_definition_file": pipeline_file,
        "parameters_file": parameters_file,
        "tag": tag,
        "run_id": run_id,
        "execution_mode": mode,
        "configuration_file": configuration_file,
        **service_configurations.services,
    }

    logger.info("Resolved configurations:")
    logger.info(json.dumps(configurations, indent=4))

    run_context = context.PipelineContext.model_validate(configurations)
    assert run_context.dag

    step_internal_name = nodes.BaseNode._get_internal_name_from_command_name(step_name)
    node_to_execute, _ = graph.search_node_by_internal_name(
        run_context.dag, step_internal_name
    )

    map_variable_dict = utils.json_to_ordered_dict(map_variable)

    if in_or_out == "in":
        logger.info("Fanning in for : %s", node_to_execute)
        run_context.pipeline_executor.fan_in(
            node=node_to_execute, map_variable=map_variable_dict
        )
    elif in_or_out == "out":
        logger.info("Fanning out for : %s", node_to_execute)
        run_context.pipeline_executor.fan_out(
            node=node_to_execute, map_variable=map_variable_dict
        )
    else:
        raise ValueError(f"Invalid mode {mode}")


if __name__ == "__main__":
    # This is only for perf testing purposes.
    # execute_single_branch()  # Missing required arguments
    pass

# if __name__ == "__main__":
#     # This is only for perf testing purposes.
#     prepare_configurations(run_id="abc", pipeline_file="examples/mocking.yaml")

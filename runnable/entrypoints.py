import json
import logging

import runnable.context as context
from runnable import defaults, graph, nodes, utils

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

    run_context.pipeline_executor.send_return_code()


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
        run_context.job_executor.add_task_log_to_catalog("job")

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


# if __name__ == "__main__":
#     # This is only for perf testing purposes.
#     prepare_configurations(run_id="abc", pipeline_file="examples/mocking.yaml")

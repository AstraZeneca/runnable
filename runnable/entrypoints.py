import importlib
import json
import logging
import os
import sys
from typing import Optional, cast

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Column

import runnable.context as context
from runnable import console, defaults, graph, task_console, tasks, utils
from runnable.defaults import RunnableConfig, ServiceConfig
from runnable.executor import BaseJobExecutor, BasePipelineExecutor

logger = logging.getLogger(defaults.LOGGER_NAME)


def get_default_configs() -> RunnableConfig:
    """
    User can provide extensions as part of their code base, runnable-config.yaml provides the place to put them.
    """
    user_configs: RunnableConfig = {}
    if utils.does_file_exist(defaults.USER_CONFIG_FILE):
        user_configs = cast(RunnableConfig, utils.load_yaml(defaults.USER_CONFIG_FILE))

    return user_configs


def prepare_configurations(
    run_id: str,
    configuration_file: str = "",
    tag: str = "",
    parameters_file: str = "",
    is_job: bool = False,
) -> context.Context:
    """
    Sets up everything needed
    Replace the placeholders in the dag/config against the variables file.

    Attach the secrets_handler, run_log_store, catalog_handler to the executor and return it.

    Args:
        variables_file (str): The variables file, if used or None
        run_id (str): The run id of the run.
        tag (str): If a tag is provided at the run time

    Returns:
        executor.BaseExecutor : A prepared executor as per the dag/config
    """
    runnable_defaults = get_default_configs()

    variables = utils.gather_variables()

    templated_configuration = {}
    configuration_file = os.environ.get(
        "RUNNABLE_CONFIGURATION_FILE", configuration_file
    )

    if configuration_file:
        templated_configuration = utils.load_yaml(configuration_file)

    # apply variables
    configuration = cast(
        RunnableConfig, utils.apply_variables(templated_configuration, variables)
    )

    # Since all the services (run_log_store, catalog, secrets, executor) are
    # dynamically loaded via stevedore, we cannot validate the configuration
    # before they are passed to the service.

    logger.info(f"Resolved configurations: {configuration}")

    # Run log settings, configuration over-rides everything
    # The user config has run-log-store while internally we use run_log_store
    run_log_config: Optional[ServiceConfig] = configuration.get("run-log-store", None)  # type: ignore
    if not run_log_config:
        run_log_config = cast(
            ServiceConfig,
            runnable_defaults.get("run-log-store", defaults.DEFAULT_RUN_LOG_STORE),
        )
    run_log_store = utils.get_provider_by_name_and_type("run_log_store", run_log_config)

    # Catalog handler settings, configuration over-rides everything
    catalog_config: Optional[ServiceConfig] = configuration.get("catalog", None)
    if not catalog_config:
        catalog_config = cast(
            ServiceConfig, runnable_defaults.get("catalog", defaults.DEFAULT_CATALOG)
        )
    catalog_handler = utils.get_provider_by_name_and_type("catalog", catalog_config)

    # Secret handler settings, configuration over-rides everything
    secrets_config: Optional[ServiceConfig] = configuration.get("secrets", None)
    if not secrets_config:
        secrets_config = cast(
            ServiceConfig, runnable_defaults.get("secrets", defaults.DEFAULT_SECRETS)
        )
    secrets_handler = utils.get_provider_by_name_and_type("secrets", secrets_config)

    # pickler
    pickler_config = cast(
        ServiceConfig, runnable_defaults.get("pickler", defaults.DEFAULT_PICKLER)
    )
    pickler_handler = utils.get_provider_by_name_and_type("pickler", pickler_config)

    if not is_job:
        # executor configurations, configuration over rides everything
        executor_config: Optional[ServiceConfig] = configuration.get(
            "pipeline-executor", None
        )  # type: ignore
        # as pipeline-executor is not a valid key
        if not executor_config:
            executor_config = cast(
                ServiceConfig,
                runnable_defaults.get(
                    "pipeline-executor", defaults.DEFAULT_PIPELINE_EXECUTOR
                ),
            )
        configured_executor = utils.get_provider_by_name_and_type(
            "pipeline_executor", executor_config
        )
    else:
        # executor configurations, configuration over rides everything
        job_executor_config: Optional[ServiceConfig] = configuration.get(
            "job-executor", None
        )  # type: ignore
        if not job_executor_config:
            job_executor_config = cast(
                ServiceConfig,
                runnable_defaults.get("job-executor", defaults.DEFAULT_JOB_EXECUTOR),
            )
        assert job_executor_config, "Job executor is not provided"
        configured_executor = utils.get_provider_by_name_and_type(
            "job_executor", job_executor_config
        )

    # Construct the context
    run_context = context.Context(
        executor=configured_executor,
        run_log_store=run_log_store,
        catalog_handler=catalog_handler,
        secrets_handler=secrets_handler,
        pickler=pickler_handler,
        variables=variables,
        tag=tag,
        run_id=run_id,
        configuration_file=configuration_file,
        parameters_file=parameters_file,
    )

    context.run_context = run_context

    return run_context


def set_pipeline_spec_from_yaml(run_context: context.Context, pipeline_file: str):
    """
    Reads the pipeline file from a YAML file and sets the pipeline spec in the run context
    """
    pipeline_config = utils.load_yaml(pipeline_file)
    logger.info("The input pipeline:")
    logger.info(json.dumps(pipeline_config, indent=4))

    dag_config = pipeline_config["dag"]

    dag_hash = utils.get_dag_hash(dag_config)
    dag = graph.create_graph(dag_config)
    run_context.dag_hash = dag_hash

    run_context.pipeline_file = pipeline_file
    run_context.dag = dag


def set_pipeline_spec_from_python(run_context: context.Context, python_module: str):
    # Call the SDK to get the dag
    # Import the module and call the function to get the dag
    module_file = python_module.strip(".py")
    module, func = utils.get_module_and_attr_names(module_file)
    sys.path.insert(0, os.getcwd())  # Need to add the current directory to path
    imported_module = importlib.import_module(module)

    run_context.from_sdk = True
    dag = getattr(imported_module, func)().return_dag()

    run_context.pipeline_file = python_module
    run_context.dag = dag


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
    run_id = utils.generate_run_id(run_id=run_id)

    run_context = prepare_configurations(
        configuration_file=configuration_file,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
    )

    assert isinstance(run_context.executor, BasePipelineExecutor)

    set_pipeline_spec_from_yaml(run_context, pipeline_file)
    executor = run_context.executor

    utils.set_runnable_environment_variables(
        run_id=run_id, configuration_file=configuration_file, tag=tag
    )

    # Prepare for graph execution
    executor._set_up_run_log(exists_ok=False)

    console.print("Working with context:")
    console.print(run_context)
    console.rule(style="[dark orange]")

    logger.info(f"Executing the graph: {run_context.dag}")
    with Progress(
        TextColumn(
            "[progress.description]{task.description}", table_column=Column(ratio=2)
        ),
        BarColumn(table_column=Column(ratio=1), style="dark_orange"),
        TimeElapsedColumn(table_column=Column(ratio=1)),
        console=console,
        expand=True,
    ) as progress:
        pipeline_execution_task = progress.add_task(
            "[dark_orange] Starting execution .. ", total=1
        )
        try:
            run_context.progress = progress
            executor.execute_graph(dag=run_context.dag)  # type: ignore

            if not executor._is_local:
                # Non local executors only traverse the graph and do not execute the nodes
                executor.send_return_code(stage="traversal")
                return

            run_log = run_context.run_log_store.get_run_log_by_id(
                run_id=run_context.run_id, full=False
            )

            if run_log.status == defaults.SUCCESS:
                progress.update(
                    pipeline_execution_task,
                    description="[green] Success",
                    completed=True,
                )
            else:
                progress.update(
                    pipeline_execution_task, description="[red] Failed", completed=True
                )
        except Exception as e:  # noqa: E722
            console.print(e, style=defaults.error_style)
            progress.update(
                pipeline_execution_task,
                description="[red] Errored execution",
                completed=True,
            )
            run_log = run_context.run_log_store.get_run_log_by_id(
                run_id=run_context.run_id, full=False
            )
            run_log.status = defaults.FAIL
            run_context.run_log_store.add_branch_log(run_log, run_context.run_id)
            raise e

    executor.send_return_code()


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
    from runnable import nodes

    task_console.print(
        f"Executing the single node: {step_name} with map variable: {map_variable}"
    )

    configuration_file = os.environ.get(
        "RUNNABLE_CONFIGURATION_FILE", configuration_file
    )

    run_context = prepare_configurations(
        configuration_file=configuration_file,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
    )
    assert isinstance(run_context.executor, BasePipelineExecutor)

    if mode == "yaml":
        # Load the yaml file
        set_pipeline_spec_from_yaml(run_context, pipeline_file)
    elif mode == "python":
        # Call the SDK to get the dag
        set_pipeline_spec_from_python(run_context, pipeline_file)

    assert run_context.dag

    task_console.print("Working with context:")
    task_console.print(run_context)
    task_console.rule(style="[dark orange]")

    executor = run_context.executor
    utils.set_runnable_environment_variables(
        run_id=run_id, configuration_file=configuration_file, tag=tag
    )

    # TODO:  Is it useful to make it get from an environment variable
    map_variable_dict = utils.json_to_ordered_dict(map_variable)

    step_internal_name = nodes.BaseNode._get_internal_name_from_command_name(step_name)
    node_to_execute, _ = graph.search_node_by_internal_name(
        run_context.dag, step_internal_name
    )

    logger.info("Executing the single node of : %s", node_to_execute)
    ## This step is where we save output of the function/shell command
    try:
        executor.execute_node(node=node_to_execute, map_variable=map_variable_dict)
    finally:
        run_context.executor.add_task_log_to_catalog(
            name=node_to_execute.internal_name, map_variable=map_variable_dict
        )

    executor.send_return_code()


def execute_job_yaml_spec(
    job_definition_file: str,
    configuration_file: str = "",
    tag: str = "",
    run_id: str = "",
    parameters_file: str = "",
):
    # A job and task are internally the same.
    run_id = utils.generate_run_id(run_id=run_id)

    run_context = prepare_configurations(
        configuration_file=configuration_file,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
        is_job=True,
    )

    assert isinstance(run_context.executor, BaseJobExecutor)

    executor = run_context.executor
    utils.set_runnable_environment_variables(
        run_id=run_id, configuration_file=configuration_file, tag=tag
    )

    run_context.job_definition_file = job_definition_file

    job_config = utils.load_yaml(job_definition_file)
    logger.info(
        "Executing the job from the user."
        f"job definition: {job_definition_file}, config: {job_config}"
    )
    assert job_config.get("type"), "Job type is not provided"

    console.print("Working with context:")
    console.print(run_context)
    console.rule(style="[dark orange]")

    # A hack where we create a task node and get our job/catalog settings
    catalog_config: list[str] = job_config.pop("catalog", {})

    # rename the type to command_type of task
    job_config["command_type"] = job_config.pop("type")
    job = tasks.create_task(job_config)

    logger.info(
        "Executing the job from the user. We are still in the caller's compute environment"
    )

    assert isinstance(executor, BaseJobExecutor)
    try:
        executor.submit_job(job, catalog_settings=catalog_config)
    finally:
        run_context.executor.add_task_log_to_catalog("job")

    executor.send_return_code()


def set_job_spec_from_yaml(run_context: context.Context, job_definition_file: str):
    """
    Reads the pipeline file from a YAML file and sets the pipeline spec in the run context
    """
    job_config = utils.load_yaml(job_definition_file)
    logger.info("The input job definition file:")
    logger.info(json.dumps(job_config, indent=4))

    catalog_config: list[str] = job_config.pop("catalog", {})

    job_config["command_type"] = job_config.pop("type")

    run_context.job_definition_file = job_definition_file
    run_context.job = tasks.create_task(job_config)
    run_context.job_catalog_settings = catalog_config


def set_job_spec_from_python(run_context: context.Context, python_module: str):
    # Import the module and call the function to get the task
    module_file = python_module.strip(".py")
    module, func = utils.get_module_and_attr_names(module_file)
    sys.path.insert(0, os.getcwd())  # Need to add the current directory to path
    imported_module = importlib.import_module(module)

    run_context.from_sdk = True
    task = getattr(imported_module, func)().get_task()
    catalog_settings = getattr(imported_module, func)().return_catalog_settings()

    run_context.job_definition_file = python_module
    run_context.job = task
    run_context.job_catalog_settings = catalog_settings


def execute_job_non_local(
    job_definition_file: str,
    configuration_file: str = "",
    mode: str = "yaml",
    tag: str = "",
    run_id: str = "",
    parameters_file: str = "",
):
    run_id = utils.generate_run_id(run_id=run_id)

    run_context = prepare_configurations(
        configuration_file=configuration_file,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
        is_job=True,
    )

    assert isinstance(run_context.executor, BaseJobExecutor)

    if mode == "yaml":
        # Load the yaml file
        set_job_spec_from_yaml(run_context, job_definition_file)
    elif mode == "python":
        # Call the SDK to get the task
        set_job_spec_from_python(run_context, job_definition_file)

    assert run_context.job

    console.print("Working with context:")
    console.print(run_context)
    console.rule(style="[dark orange]")

    logger.info(
        "Executing the job from the user. We are still in the caller's compute environment"
    )

    try:
        run_context.executor.execute_job(
            run_context.job, catalog_settings=run_context.job_catalog_settings
        )
    finally:
        run_context.executor.add_task_log_to_catalog("job")

    run_context.executor.send_return_code()


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
    from runnable import nodes

    configuration_file = os.environ.get(
        "RUNNABLE_CONFIGURATION_FILE", configuration_file
    )

    run_context = prepare_configurations(
        configuration_file=configuration_file,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
    )

    assert isinstance(run_context.executor, BasePipelineExecutor)

    if mode == "yaml":
        # Load the yaml file
        set_pipeline_spec_from_yaml(run_context, pipeline_file)
    elif mode == "python":
        # Call the SDK to get the dag
        set_pipeline_spec_from_python(run_context, pipeline_file)

    console.print("Working with context:")
    console.print(run_context)
    console.rule(style="[dark orange]")

    executor = run_context.executor
    utils.set_runnable_environment_variables(
        run_id=run_id, configuration_file=configuration_file, tag=tag
    )

    step_internal_name = nodes.BaseNode._get_internal_name_from_command_name(step_name)
    node_to_execute, _ = graph.search_node_by_internal_name(
        run_context.dag,  # type: ignore
        step_internal_name,
    )

    map_variable_dict = utils.json_to_ordered_dict(map_variable)

    if in_or_out == "in":
        logger.info("Fanning in for : %s", node_to_execute)
        executor.fan_in(node=node_to_execute, map_variable=map_variable_dict)
    elif in_or_out == "out":
        logger.info("Fanning out for : %s", node_to_execute)
        executor.fan_out(node=node_to_execute, map_variable=map_variable_dict)
    else:
        raise ValueError(f"Invalid mode {mode}")


# if __name__ == "__main__":
#     # This is only for perf testing purposes.
#     prepare_configurations(run_id="abc", pipeline_file="examples/mocking.yaml")

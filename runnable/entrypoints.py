import json
import logging
from typing import Optional, cast

from rich import print

import runnable.context as context
from runnable import defaults, graph, utils
from runnable.defaults import MagnusConfig, ServiceConfig

logger = logging.getLogger(defaults.LOGGER_NAME)


def get_default_configs() -> MagnusConfig:
    """
    User can provide extensions as part of their code base, magnus-config.yaml provides the place to put them.
    """
    user_configs = {}
    if utils.does_file_exist(defaults.USER_CONFIG_FILE):
        user_configs = utils.load_yaml(defaults.USER_CONFIG_FILE)

    if not user_configs:
        return {}

    user_defaults = user_configs.get("defaults", {})
    if user_defaults:
        return user_defaults

    return {}


def prepare_configurations(
    run_id: str,
    configuration_file: str = "",
    pipeline_file: str = "",
    tag: str = "",
    use_cached: str = "",
    parameters_file: str = "",
    force_local_executor: bool = False,
) -> context.Context:
    """
    Replace the placeholders in the dag/config against the variables file.

    Attach the secrets_handler, run_log_store, catalog_handler to the executor and return it.

    Args:
        variables_file (str): The variables file, if used or None
        pipeline_file (str): The config/dag file
        run_id (str): The run id of the run.
        tag (str): If a tag is provided at the run time
        use_cached (str): Provide the run_id of the older run

    Returns:
        executor.BaseExecutor : A prepared executor as per the dag/config
    """
    magnus_defaults = get_default_configs()

    variables = utils.gather_variables()

    templated_configuration = {}
    if configuration_file:
        templated_configuration = utils.load_yaml(configuration_file) or {}

    configuration: MagnusConfig = cast(MagnusConfig, templated_configuration)

    # Run log settings, configuration over-rides everything
    run_log_config: Optional[ServiceConfig] = configuration.get("run_log_store", None)
    if not run_log_config:
        run_log_config = cast(ServiceConfig, magnus_defaults.get("run_log_store", defaults.DEFAULT_RUN_LOG_STORE))
    run_log_store = utils.get_provider_by_name_and_type("run_log_store", run_log_config)

    # Catalog handler settings, configuration over-rides everything
    catalog_config: Optional[ServiceConfig] = configuration.get("catalog", None)
    if not catalog_config:
        catalog_config = cast(ServiceConfig, magnus_defaults.get("catalog", defaults.DEFAULT_CATALOG))
    catalog_handler = utils.get_provider_by_name_and_type("catalog", catalog_config)

    # Secret handler settings, configuration over-rides everything
    secrets_config: Optional[ServiceConfig] = configuration.get("secrets", None)
    if not secrets_config:
        secrets_config = cast(ServiceConfig, magnus_defaults.get("secrets", defaults.DEFAULT_SECRETS))
    secrets_handler = utils.get_provider_by_name_and_type("secrets", secrets_config)

    # experiment tracker settings, configuration over-rides everything
    tracker_config: Optional[ServiceConfig] = configuration.get("experiment_tracker", None)
    if not tracker_config:
        tracker_config = cast(
            ServiceConfig, magnus_defaults.get("experiment_tracker", defaults.DEFAULT_EXPERIMENT_TRACKER)
        )
    tracker_handler = utils.get_provider_by_name_and_type("experiment_tracker", tracker_config)

    # executor configurations, configuration over rides everything
    executor_config: Optional[ServiceConfig] = configuration.get("executor", None)
    if force_local_executor:
        executor_config = ServiceConfig(type="local", config={})

    if not executor_config:
        executor_config = cast(ServiceConfig, magnus_defaults.get("executor", defaults.DEFAULT_EXECUTOR))
    configured_executor = utils.get_provider_by_name_and_type("executor", executor_config)

    # Construct the context
    run_context = context.Context(
        executor=configured_executor,
        run_log_store=run_log_store,
        catalog_handler=catalog_handler,
        secrets_handler=secrets_handler,
        experiment_tracker=tracker_handler,
        variables=variables,
        tag=tag,
        run_id=run_id,
        configuration_file=configuration_file,
        parameters_file=parameters_file,
    )

    if pipeline_file:
        # There are use cases where we are only preparing the executor
        pipeline_config = utils.load_yaml(pipeline_file)

        logger.info("The input pipeline:")
        logger.info(json.dumps(pipeline_config, indent=4))

        # Create the graph
        dag_config = pipeline_config["dag"]
        dag_hash = utils.get_dag_hash(dag_config)
        dag = graph.create_graph(dag_config)

        run_context.pipeline_file = pipeline_file
        run_context.dag = dag
        run_context.dag_hash = dag_hash

    run_context.use_cached = False
    if use_cached:
        run_context.use_cached = True
        run_context.original_run_id = use_cached

    context.run_context = run_context

    return run_context


def execute(
    configuration_file: str,
    pipeline_file: str,
    tag: str = "",
    run_id: str = "",
    use_cached: str = "",
    parameters_file: str = "",
):
    # pylint: disable=R0914,R0913
    """
    The entry point to magnus execution. This method would prepare the configurations and delegates traversal to the
    executor

    Args:
        pipeline_file (str): The config/dag file
        run_id (str): The run id of the run.
        tag (str): If a tag is provided at the run time
        use_cached (str): The previous run_id to use.
        parameters_file (str): The parameters being sent in to the application
    """
    # Re run settings
    run_id = utils.generate_run_id(run_id=run_id)

    run_context = prepare_configurations(
        configuration_file=configuration_file,
        pipeline_file=pipeline_file,
        run_id=run_id,
        tag=tag,
        use_cached=use_cached,
        parameters_file=parameters_file,
    )
    print("Working with context:")
    print(run_context)

    executor = run_context.executor

    run_context.execution_plan = defaults.EXECUTION_PLAN.CHAINED.value

    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

    # Prepare for graph execution
    executor.prepare_for_graph_execution()

    logger.info("Executing the graph")
    executor.execute_graph(dag=run_context.dag)  # type: ignore

    executor.send_return_code()


def execute_single_node(
    configuration_file: str,
    pipeline_file: str,
    step_name: str,
    map_variable: str,
    run_id: str,
    tag: str = "",
    parameters_file: str = "",
):
    """
    The entry point into executing a single node of magnus. Orchestration modes should extensively use this
    entry point.

    It should have similar set up of configurations to execute because orchestrator modes can initiate the execution.

    Args:
        variables_file (str): The variables file, if used or None
        step_name : The name of the step to execute in dot path convention
        pipeline_file (str): The config/dag file
        run_id (str): The run id of the run.
        tag (str): If a tag is provided at the run time
        parameters_file (str): The parameters being sent in to the application

    """
    from runnable import nodes

    run_context = prepare_configurations(
        configuration_file=configuration_file,
        pipeline_file=pipeline_file,
        run_id=run_id,
        tag=tag,
        use_cached="",
        parameters_file=parameters_file,
    )
    print("Working with context:")
    print(run_context)

    executor = run_context.executor
    run_context.execution_plan = defaults.EXECUTION_PLAN.CHAINED.value
    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

    executor.prepare_for_node_execution()

    if not run_context.dag:
        # There are a few entry points that make graph dynamically and do not have a dag defined statically.
        run_log = run_context.run_log_store.get_run_log_by_id(run_id=run_id, full=False)
        run_context.dag = graph.create_graph(run_log.run_config["pipeline"])

    step_internal_name = nodes.BaseNode._get_internal_name_from_command_name(step_name)

    map_variable_dict = utils.json_to_ordered_dict(map_variable)

    node_to_execute, _ = graph.search_node_by_internal_name(run_context.dag, step_internal_name)

    logger.info("Executing the single node of : %s", node_to_execute)
    executor.execute_node(node=node_to_execute, map_variable=map_variable_dict)

    executor.send_return_code(stage="execution")


def execute_single_brach(
    configuration_file: str,
    pipeline_file: str,
    branch_name: str,
    map_variable: str,
    run_id: str,
    tag: str,
):
    """
    The entry point into executing a branch of the graph. Interactive modes in parallel runs use this to execute
    branches in parallel.

    This entry point is never used by its own but rather from a node. So the arguments sent into this are fewer.

    Args:
        variables_file (str): The variables file, if used or None
        branch_name : The name of the branch to execute, in dot.path.convention
        pipeline_file (str): The config/dag file
        run_id (str): The run id of the run.
        tag (str): If a tag is provided at the run time
    """
    from runnable import nodes

    run_context = prepare_configurations(
        configuration_file=configuration_file,
        pipeline_file=pipeline_file,
        run_id=run_id,
        tag=tag,
        use_cached="",
    )
    print("Working with context:")
    print(run_context)

    executor = run_context.executor
    run_context.execution_plan = defaults.EXECUTION_PLAN.CHAINED.value
    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

    branch_internal_name = nodes.BaseNode._get_internal_name_from_command_name(branch_name)

    map_variable_dict = utils.json_to_ordered_dict(map_variable)

    branch_to_execute = graph.search_branch_by_internal_name(run_context.dag, branch_internal_name)  # type: ignore

    logger.info("Executing the single branch of %s", branch_to_execute)
    executor.execute_graph(dag=branch_to_execute, map_variable=map_variable_dict)

    executor.send_return_code()


def execute_notebook(
    entrypoint: str,
    notebook_file: str,
    catalog_config: dict,
    configuration_file: str,
    notebook_output_path: str = "",
    tag: str = "",
    run_id: str = "",
    parameters_file: str = "",
):
    """
    The entry point to magnus execution of a notebook. This method would prepare the configurations and
    delegates traversal to the executor
    """
    run_id = utils.generate_run_id(run_id=run_id)

    run_context = prepare_configurations(
        configuration_file=configuration_file,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
    )

    executor = run_context.executor
    run_context.execution_plan = defaults.EXECUTION_PLAN.UNCHAINED.value
    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

    print("Working with context:")
    print(run_context)

    step_config = {
        "command": notebook_file,
        "command_type": "notebook",
        "notebook_output_path": notebook_output_path,
        "type": "task",
        "next": "success",
        "catalog": catalog_config,
    }
    node = graph.create_node(name="executing job", step_config=step_config)

    if entrypoint == defaults.ENTRYPOINT.USER.value:
        # Prepare for graph execution
        executor.prepare_for_graph_execution()

        logger.info("Executing the job from the user. We are still in the caller's compute environment")
        executor.execute_job(node=node)

    elif entrypoint == defaults.ENTRYPOINT.SYSTEM.value:
        executor.prepare_for_node_execution()
        logger.info("Executing the job from the system. We are in the config's compute environment")
        executor.execute_node(node=node)

        # Update the status of the run log
        step_log = run_context.run_log_store.get_step_log(node._get_step_log_name(), run_id)
        run_context.run_log_store.update_run_log_status(run_id=run_id, status=step_log.status)

    else:
        raise ValueError(f"Invalid entrypoint {entrypoint}")

    executor.send_return_code()


def execute_function(
    entrypoint: str,
    command: str,
    catalog_config: dict,
    configuration_file: str,
    tag: str = "",
    run_id: str = "",
    parameters_file: str = "",
):
    """
    The entry point to magnus execution of a function. This method would prepare the configurations and
    delegates traversal to the executor
    """
    run_id = utils.generate_run_id(run_id=run_id)

    run_context = prepare_configurations(
        configuration_file=configuration_file,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
    )

    executor = run_context.executor

    run_context.execution_plan = defaults.EXECUTION_PLAN.UNCHAINED.value
    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

    print("Working with context:")
    print(run_context)

    # Prepare the graph with a single node
    step_config = {
        "command": command,
        "command_type": "python",
        "type": "task",
        "next": "success",
        "catalog": catalog_config,
    }
    node = graph.create_node(name="executing job", step_config=step_config)

    if entrypoint == defaults.ENTRYPOINT.USER.value:
        # Prepare for graph execution
        executor.prepare_for_graph_execution()

        logger.info("Executing the job from the user. We are still in the caller's compute environment")
        executor.execute_job(node=node)

    elif entrypoint == defaults.ENTRYPOINT.SYSTEM.value:
        executor.prepare_for_node_execution()
        logger.info("Executing the job from the system. We are in the config's compute environment")
        executor.execute_node(node=node)

        # Update the status of the run log
        step_log = run_context.run_log_store.get_step_log(node._get_step_log_name(), run_id)
        run_context.run_log_store.update_run_log_status(run_id=run_id, status=step_log.status)

    else:
        raise ValueError(f"Invalid entrypoint {entrypoint}")

    executor.send_return_code()


def fan(
    configuration_file: str,
    pipeline_file: str,
    step_name: str,
    mode: str,
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

    run_context = prepare_configurations(
        configuration_file=configuration_file,
        pipeline_file=pipeline_file,
        run_id=run_id,
        tag=tag,
        use_cached="",
        parameters_file=parameters_file,
    )
    print("Working with context:")
    print(run_context)

    executor = run_context.executor
    run_context.execution_plan = defaults.EXECUTION_PLAN.CHAINED.value
    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

    executor.prepare_for_node_execution()

    step_internal_name = nodes.BaseNode._get_internal_name_from_command_name(step_name)
    node_to_execute, _ = graph.search_node_by_internal_name(run_context.dag, step_internal_name)  # type: ignore

    map_variable_dict = utils.json_to_ordered_dict(map_variable)

    if mode == "in":
        logger.info("Fanning in for : %s", node_to_execute)
        executor.fan_in(node=node_to_execute, map_variable=map_variable_dict)
    elif mode == "out":
        logger.info("Fanning out for : %s", node_to_execute)
        executor.fan_out(node=node_to_execute, map_variable=map_variable_dict)
    else:
        raise ValueError(f"Invalid mode {mode}")


if __name__ == "__main__":
    # This is only for perf testing purposes.
    prepare_configurations(run_id="abc", pipeline_file="example/mocking.yaml")

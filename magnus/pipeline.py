import json
import logging
from typing import Optional, Union

from magnus import defaults, exceptions, graph, utils

logger = logging.getLogger(defaults.NAME)

# Set this global executor to the fitted executor for access later
global_executor = None  # pylint: disable=invalid-name # type: ignore

# TODO: Tests and mypy


def get_default_configs() -> dict:
    """
    User can provide extensions as part of their code base, magnus-config.yaml provides the place to put them.
    """
    user_configs = {}
    if utils.does_file_exist(defaults.USER_CONFIG_FILE):
        user_configs = utils.load_yaml(defaults.USER_CONFIG_FILE)

    if not user_configs:
        return {}

    user_defaults = user_configs.get('defaults', {})
    if user_defaults:
        return user_defaults

    return {}


def prepare_configurations(
        configuration_file: str = None,
        pipeline_file: str = None,
        run_id: str = None,
        tag: Union[str, None] = None,
        use_cached: Union[str, None] = '',
        parameters_file: str = None):
    # pylint: disable=R0914
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

    configuration = {}
    if configuration_file:
        configuration = utils.load_yaml(configuration_file)

    # apply variables
    configuration = utils.apply_variables(configuration, variables)

    # Run log settings, configuration over-rides everything
    run_log_config = configuration.get('run_log_store', {})
    if not run_log_config:
        run_log_config = magnus_defaults.get('run_log_store', defaults.DEFAULT_RUN_LOG_STORE)
    run_log_store = utils.get_provider_by_name_and_type('run_log_store', run_log_config)

    # Catalog handler settings, configuration over-rides everything
    catalog_config = configuration.get('catalog', {})
    if not catalog_config:
        catalog_config = magnus_defaults.get('catalog', defaults.DEFAULT_CATALOG)
    catalog_handler = utils.get_provider_by_name_and_type('catalog', catalog_config)

    # Secret handler settings, configuration over-rides everything
    secrets_config = configuration.get('secrets', {})
    if not secrets_config:
        secrets_config = magnus_defaults.get('secrets', defaults.DEFAULT_SECRETS)
    secrets_handler = utils.get_provider_by_name_and_type('secrets', secrets_config)

    # experiment tracker settings, configuration over-rides everything
    tracker_config = configuration.get('experiment_tracking', {})
    if not tracker_config:
        tracker_config = magnus_defaults.get('experiment_tracking', defaults.DEFAULT_EXPERIMENT_TRACKER)
    tracker_handler = utils.get_provider_by_name_and_type('experiment_tracking', tracker_config)

    # Mode configurations, configuration over rides everything
    mode_config = configuration.get('mode', {})
    if not mode_config:
        mode_config = magnus_defaults.get('executor', defaults.DEFAULT_EXECUTOR)
    mode_executor = utils.get_provider_by_name_and_type('executor', mode_config)

    if pipeline_file:
        # There are use cases where we are only preparing the executor
        pipeline_config = utils.load_yaml(pipeline_file)
        pipeline_config = utils.apply_variables(pipeline_config, variables=variables)

        logger.info('The input pipeline:')
        logger.info(json.dumps(pipeline_config, indent=4))

        # Create the graph
        dag_config = pipeline_config['dag']
        dag_hash = utils.get_dag_hash(dag_config)
        # TODO: Dag nodes should not self refer themselves
        dag = graph.create_graph(dag_config)

        mode_executor.pipeline_file = pipeline_file
        mode_executor.dag = dag
        mode_executor.dag_hash = dag_hash

    mode_executor.run_id = run_id
    mode_executor.tag = tag
    mode_executor.use_cached = use_cached

    # Set a global executor for inter-module access later
    global global_executor  # pylint: disable=W0603,invalid-name,
    global_executor = mode_executor

    mode_executor.run_log_store = run_log_store
    mode_executor.catalog_handler = catalog_handler
    mode_executor.secrets_handler = secrets_handler
    mode_executor.experiment_tracker = tracker_handler
    mode_executor.configuration_file = configuration_file
    mode_executor.parameters_file = parameters_file
    mode_executor.variables = variables

    return mode_executor


def execute(
        configuration_file: str,
        pipeline_file: str,
        tag: str = None,
        run_id: str = None,
        use_cached: str = None,
        parameters_file: str = None):
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

    mode_executor = prepare_configurations(configuration_file=configuration_file,
                                           pipeline_file=pipeline_file,
                                           run_id=run_id,
                                           tag=tag,
                                           use_cached=use_cached,
                                           parameters_file=parameters_file)
    mode_executor.execution_plan = defaults.EXECUTION_PLAN.pipeline

    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

    previous_run_log = None
    if use_cached:
        try:
            previous_run_log = mode_executor.run_log_store.get_run_log_by_id(run_id=use_cached, full=True)
        except exceptions.RunLogNotFoundError as _e:
            msg = (
                f'There is no run by {use_cached} in the current run log store '
                f'{mode_executor.run_log_store.service_name}. Please ensure that that run log exists to re-run.\n'
                'Note: Even if the previous run used a different run log store, provide the run log store in the format'
                ' accepted by the current run log store.'
            )
            raise Exception(msg) from _e

        if previous_run_log.dag_hash != mode_executor.dag_hash:
            logger.warning('The previous dag does not match to the current one!')
        mode_executor.previous_run_log = previous_run_log
        logger.info('Found a previous run log and using it as cache')

    # Prepare for graph execution
    mode_executor.prepare_for_graph_execution()

    logger.info('Executing the graph')
    mode_executor.execute_graph(dag=mode_executor.dag)

    mode_executor.send_return_code()


def execute_single_step(
        configuration_file: str,
        pipeline_file: str,
        step_name: str,
        run_id: str,
        tag: str = None,
        parameters_file: str = None,
        use_cached: str = None,):
    # pylint: disable=R0914,R0913
    """
    The entry point into executing a single step of magnus.

    It should have similar set up of configurations to execute because orchestrator modes can initiate the execution.

    Args:
        variables_file (str): The variables file, if used or None
        step_name : The name of the step to execute in dot path convention
        pipeline_file (str): The config/dag file
        run_id (str): The run id of the run.
        tag (str): If a tag is provided at the run time
        parameters_file (str): The parameters being sent in to the application

    """
    run_id = utils.generate_run_id(run_id=run_id)

    mode_executor = prepare_configurations(configuration_file=configuration_file,
                                           pipeline_file=pipeline_file,
                                           run_id=run_id,
                                           tag=tag,
                                           use_cached='',
                                           parameters_file=parameters_file)
    mode_executor.execution_plan = defaults.EXECUTION_PLAN.pipeline
    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)
    try:
        _ = mode_executor.dag.get_node_by_name(step_name)
    except exceptions.NodeNotFoundError as e:
        msg = (
            f"The node by name {step_name} is not found in the graph. Please provide a valid node name"
        )
        raise Exception(msg) from e

    previous_run_log = None
    if use_cached:
        try:
            previous_run_log = mode_executor.run_log_store.get_run_log_by_id(run_id=use_cached, full=True)
        except exceptions.RunLogNotFoundError as _e:
            msg = (
                f'There is no run by {use_cached} in the current run log store '
                f'{mode_executor.run_log_store.service_name}. Please ensure that that run log exists to re-run.\n'
                'Note: Even if the previous run used a different run log store, provide the run log store in the format'
                ' accepted by the current run log store.'
            )
            raise Exception(msg) from _e

        if previous_run_log.dag_hash != mode_executor.dag_hash:
            logger.warning('The previous dag does not match to the current one!')
        mode_executor.previous_run_log = previous_run_log
        logger.info('Found a previous run log and using it as cache')

    mode_executor.single_step = step_name
    mode_executor.prepare_for_graph_execution()

    logger.info('Executing the graph')
    mode_executor.execute_graph(dag=mode_executor.dag)

    mode_executor.send_return_code()


def execute_single_node(
        configuration_file: str,
        pipeline_file: str,
        step_name: str,
        map_variable: str,
        run_id: str,
        tag: str = None,
        parameters_file: str = None):
    # pylint: disable=R0914,R0913
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
    from magnus import nodes
    mode_executor = prepare_configurations(configuration_file=configuration_file,
                                           pipeline_file=pipeline_file,
                                           run_id=run_id,
                                           tag=tag,
                                           use_cached='',
                                           parameters_file=parameters_file)
    mode_executor.execution_plan = defaults.EXECUTION_PLAN.pipeline
    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

    mode_executor.prepare_for_node_execution()

    if not mode_executor.dag:
        # There are a few entry points that make graph dynamically and do not have a dag defined statically.
        run_log = mode_executor.run_log_store.get_run_log_by_id(run_id=run_id, full=False)
        mode_executor.dag = graph.create_graph(run_log.run_config['pipeline'])

    step_internal_name = nodes.BaseNode._get_internal_name_from_command_name(step_name)

    map_variable_dict = utils.json_to_ordered_dict(map_variable)

    node_to_execute, _ = graph.search_node_by_internal_name(mode_executor.dag, step_internal_name)

    logger.info('Executing the single node of : %s', node_to_execute)
    mode_executor.execute_node(node=node_to_execute, map_variable=map_variable_dict)

    mode_executor.send_return_code(stage='execution')


def execute_single_brach(
        configuration_file: str,
        pipeline_file: str,
        branch_name: str,
        map_variable: str,
        run_id: str,
        tag: str = None):
    # pylint: disable=R0914,R0913
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
    from magnus import nodes
    mode_executor = prepare_configurations(configuration_file=configuration_file,
                                           pipeline_file=pipeline_file,
                                           run_id=run_id,
                                           tag=tag,
                                           use_cached='')
    mode_executor.execution_plan = defaults.EXECUTION_PLAN.pipeline
    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

    branch_internal_name = nodes.BaseNode._get_internal_name_from_command_name(branch_name)

    map_variable_dict = utils.json_to_ordered_dict(map_variable)

    branch_to_execute = graph.search_branch_by_internal_name(mode_executor.dag, branch_internal_name)

    logger.info('Executing the single branch of %s', branch_to_execute)
    mode_executor.execute_graph(dag=branch_to_execute, map_variable=map_variable_dict)

    mode_executor.send_return_code()


def execute_notebook(
        notebook_file: str,
        catalog_config: dict,
        configuration_file: str,
        tag: str = None,
        run_id: str = None,
        parameters_file: str = None):
    # pylint: disable=R0914,R0913
    """
    The entry point to magnus execution of a notebook. This method would prepare the configurations and
    delegates traversal to the executor
    """
    run_id = utils.generate_run_id(run_id=run_id)

    mode_executor = prepare_configurations(
        configuration_file=configuration_file,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file)

    mode_executor.execution_plan = defaults.EXECUTION_PLAN.notebook
    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

    # Prepare the graph with a single node
    dag = graph.Graph(start_at='executing notebook')
    step_config = {
        'command': notebook_file,
        'command_type': 'notebook',
        'type': 'task',
        'next': 'success',
        'catalog': catalog_config,
    }
    node = graph.create_node(name=f'executing notebook', step_config=step_config)

    dag.add_node(node)
    dag.add_terminal_nodes()

    mode_executor.dag = dag
    # Prepare for graph execution
    mode_executor.prepare_for_graph_execution()

    logger.info('Executing the graph')
    mode_executor.execute_graph(dag=mode_executor.dag)

    mode_executor.send_return_code()


def execute_function(
        command: str,
        catalog_config: dict,
        configuration_file: str,
        tag: str = None,
        run_id: str = None,
        parameters_file: str = None):
    # pylint: disable=R0914,R0913
    """
    The entry point to magnus execution of a function. This method would prepare the configurations and
    delegates traversal to the executor
    """
    run_id = utils.generate_run_id(run_id=run_id)

    mode_executor = prepare_configurations(
        configuration_file=configuration_file,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file)

    mode_executor.execution_plan = defaults.EXECUTION_PLAN.function
    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

    # Prepare the graph with a single node
    dag = graph.Graph(start_at='executing function')
    step_config = {
        'command': command,
        'command_type': 'python',
        'type': 'task',
        'next': 'success',
        'catalog': catalog_config,
    }
    node = graph.create_node(name=f'executing function', step_config=step_config)

    dag.add_node(node)
    dag.add_terminal_nodes()

    mode_executor.dag = dag
    # Prepare for graph execution
    mode_executor.prepare_for_graph_execution()

    logger.info('Executing the graph')
    mode_executor.execute_graph(dag=mode_executor.dag)

    mode_executor.send_return_code()

import json
import logging
from typing import Union

from magnus import defaults, exceptions, graph, nodes, utils

logger = logging.getLogger(defaults.NAME)

# Set this global executor to the fitted executor for access later
global_executor = None  # pylint: disable=invalid-name
magnus_defaults = {}  # pylint: disable=invalid-name


def load_user_extensions():
    """
    User can provide extensions as part of their code base, magnus-config.yaml provides the place to put them.
    Look for them and load the extensions if provided.

    # TODO: With introduction of stevedore, this fails. Should find a way to do this the stevedore way
    """
    user_configs = {}
    if utils.does_file_exist(defaults.USER_CONFIG_FILE):
        user_configs = utils.load_yaml(defaults.USER_CONFIG_FILE)

    if not user_configs:
        return

    extensions = user_configs.get('extensions', [])
    for extension in extensions:
        logger.info('Loading User extension: %s', extension)
        __import__(extension)

    user_defaults = user_configs.get('defaults', {})
    if user_defaults:
        global magnus_defaults  # pylint: disable=W0603,invalid-name,
        magnus_defaults = user_defaults


def prepare_configurations(
        variables_file: str,
        configuration_file: str,
        pipeline_file: str,
        run_id: str,
        tag: Union[str, None],
        use_cached: Union[str, None],
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
    global magnus_defaults

    pipeline_config = utils.load_yaml(pipeline_file)

    variables = {}
    if variables_file:
        variables = utils.load_yaml(variables_file)

    configuration = {}
    if configuration_file:
        configuration = utils.load_yaml(configuration_file)

    # apply variables
    pipeline_config = utils.apply_variables(pipeline_config, variables=variables)

    logger.info('The input pipeline:')
    logger.info(json.dumps(pipeline_config, indent=4))

    # Create the graph
    dag_config = pipeline_config['dag']
    dag_hash = utils.get_dag_hash(dag_config)
    # TODO: Dag nodes should not self refer themselves
    dag = graph.create_graph(dag_config)

    # Run log settings, configuration over-rides everything
    run_log_config = configuration.get('run_log_store', {})
    if not run_log_config:
        default_run_log_config = magnus_defaults.get('run_log_store', defaults.DEFAULT_RUN_LOG_STORE)
        run_log_config = pipeline_config.get('run_log_store', {}) or default_run_log_config
    run_log_store = utils.get_provider_by_name_and_type('run_log_store', run_log_config)

    # Catalog handler settings, configuration over-rides everything
    catalog_config = configuration.get('catalog', {})
    if not catalog_config:
        default_catalog_config = magnus_defaults.get('catalog', defaults.DEFAULT_CATALOG)
        catalog_config = pipeline_config.get('catalog', {}) or default_catalog_config
    catalog_handler = utils.get_provider_by_name_and_type('catalog', catalog_config)

    # Secret handler settings, configuration over-rides everything
    secrets_config = configuration.get('secrets', {})
    if not secrets_config:
        default_secrets_config = magnus_defaults.get('secrets', defaults.DEFAULT_SECRETS)
        secrets_config = pipeline_config.get('secrets', {}) or default_secrets_config
    secrets_handler = utils.get_provider_by_name_and_type('secrets', secrets_config)

    # Mode configurations, configuration over rides everything
    mode_config = configuration.get('mode', {})
    if not mode_config:
        default_mode_config = magnus_defaults.get('executor', defaults.DEFAULT_EXECUTOR)
        mode_config = pipeline_config.get('mode', {}) or default_mode_config
    mode_executor = utils.get_provider_by_name_and_type('executor', mode_config)

    mode_executor.pipeline_file = pipeline_file
    mode_executor.dag = dag
    mode_executor.run_id = run_id
    mode_executor.tag = tag
    mode_executor.use_cached = use_cached

    # Set a global executor for inter-module access later
    global global_executor  # pylint: disable=W0603,invalid-name,
    global_executor = mode_executor

    mode_executor.run_log_store = run_log_store
    mode_executor.catalog_handler = catalog_handler
    mode_executor.dag_hash = dag_hash
    mode_executor.secrets_handler = secrets_handler
    mode_executor.variables_file = variables_file
    mode_executor.configuration_file = configuration_file
    mode_executor.parameters_file = parameters_file

    return mode_executor


def execute(
        variables_file: str,
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
        variables_file (str): The variables file, if used or None
        pipeline_file (str): The config/dag file
        run_id (str): The run id of the run.
        tag (str): If a tag is provided at the run time
        use_cached (str): The previous run_id to use.
        parameters_file (str): The parameters being sent in to the application
    """
    # Re run settings
    run_id = utils.generate_run_id(run_id=run_id)

    mode_executor = prepare_configurations(variables_file=variables_file,
                                           configuration_file=configuration_file,
                                           pipeline_file=pipeline_file,
                                           run_id=run_id,
                                           tag=tag,
                                           use_cached=use_cached,
                                           parameters_file=parameters_file)
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
        variables_file: str,
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

    mode_executor = prepare_configurations(variables_file=variables_file,
                                           configuration_file=configuration_file,
                                           pipeline_file=pipeline_file,
                                           run_id=run_id,
                                           tag=tag,
                                           use_cached='',
                                           parameters_file=parameters_file)

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
        variables_file: str,
        configuration_file: str,
        pipeline_file: str,
        step_name: str,
        map_variable: str,
        run_id: str,
        tag: str = None,
        parameters_file: str = None):
    # pylint: disable=R0914,R0913
    """
    The entry point into executing a single node of magnus. Orchestration modes should extensivesly use this
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
    mode_executor = prepare_configurations(variables_file=variables_file,
                                           configuration_file=configuration_file,
                                           pipeline_file=pipeline_file,
                                           run_id=run_id,
                                           tag=tag,
                                           use_cached='',
                                           parameters_file=parameters_file)

    step_internal_name = nodes.BaseNode.get_internal_name_from_command_name(step_name)

    map_variable_dict = utils.json_to_ordered_dict(map_variable)

    node_to_execute, _ = graph.search_node_by_internal_name(mode_executor.dag, step_internal_name)

    mode_executor.prepare_for_node_execution(node_to_execute, map_variable=map_variable_dict)

    logger.info('Executing the single node of : %s', node_to_execute)
    mode_executor.execute_node(node=node_to_execute, map_variable=map_variable_dict)

    mode_executor.send_return_code(stage='execution')


def execute_single_brach(
        variables_file: str,
        configuration_file: str,
        pipeline_file: str,
        branch_name: str,
        map_variable: str,
        run_id: str):
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
    mode_executor = prepare_configurations(variables_file=variables_file,
                                           configuration_file=configuration_file,
                                           pipeline_file=pipeline_file,
                                           run_id=run_id,
                                           tag=None,
                                           use_cached='')

    branch_internal_name = nodes.BaseNode.get_internal_name_from_command_name(branch_name)

    map_variable_dict = utils.json_to_ordered_dict(map_variable)

    branch_to_execute = graph.search_branch_by_internal_name(mode_executor.dag, branch_internal_name)

    logger.info('Executing the single branch of %s', branch_to_execute)
    mode_executor.execute_graph(dag=branch_to_execute, map_variable=map_variable_dict)

    mode_executor.send_return_code()


load_user_extensions()

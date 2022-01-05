import logging
import re

from magnus.executor import BaseExecutor
from magnus.graph import Graph
from magnus.nodes import BaseNode
from magnus import datastore
from magnus import defaults
from magnus import utils
from magnus import integration
from magnus import exceptions

logger = logging.getLogger(defaults.NAME)

# TODO: Move this to executor as default provided implementation


class DemoRenderer(BaseExecutor):
    """
    This renderer is an example of how you can render required job specifications as per your orchestration tool.

    BaseExecutor implements many of the functionalities that are common and can be safe defaults.
    In this renderer example: We just render a bash script that sequentially calls the steps.
    We do not handle composite steps in this mode.

    Example config:
    mode:
      type: demo-renderer
    """
    service_name = 'demo-renderer'

    def __init__(self, config):
        """
        Use the config mapping to validate and customize any functionality for your executor.

        Args:
            config ([type]): [description]
        """
        super().__init__(config)

    def is_parallel_execution(self) -> bool:  # pylint: disable=R0201
        """
        Controls the parallelization of branches in map and parallel state.

        Most orchestrators control the parralization of the branches outside of magnus control.
        i.e, You would render the parallel job job specification in the language of the orchestrator.

        NOTE: Most often, this should be false for modes that rely upon other orchestration tools.

        Returns:
            bool: True if the mode allows parallel execution of branches.
        """
        return defaults.ENABLE_PARALLEL

    def prepare_for_graph_execution(self):
        """
        This method would be called prior to calling execute_graph.
        Perform any steps required before doing the graph execution.

        For most rendering jobs, we need not do anything but customize according to your needs.

        NOTE: You might want to over-ride this method to do nothing.
        """
        ...

    def prepare_for_node_execution(self, node: BaseNode, map_variable: str = ''):
        """
        This method would be called prior to the node execution in the environment of the compute.

        Use this method to set up the required things for the compute.
        The most common examples might be to ensure that the appropriate run log is in place.

        NOTE: You might need to over-ride this method.
        For interactive modes, prepare_for_graph_execution takes care of a lot of set up. For orchestrated modes,
        the same work has to be done by prepare_for_node_execution.
        """
        integration.validate(self, self.run_log_store)
        integration.configure_for_execution(self, self.run_log_store)

        integration.validate(self, self.catalog_handler)
        integration.configure_for_execution(self, self.run_log_store)

        integration.validate(self, self.secrets_handler)
        integration.configure_for_execution(self, self.run_log_store)

        try:
            # Try to get it if previous steps have created it
            self.run_log_store.get_run_log_by_id(self.run_id)
        except exceptions.RunLogNotFoundError:
            # Create one if they are not created
            self.set_up_run_log()

        # Create the step log also for the step
        step_log = self.run_log_store.create_step_log(node.name, node.get_step_log_name(map_variable))

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING
        self.run_log_store.add_step_log(step_log, self.run_id)

    def sync_catalog(self, node: BaseNode, step_log: datastore.StepLog, stage: str, synced_catalogs=None):
        """
        Syncs the catalog for both get and put stages.

        The default executors implementation just delegates the functionlity to catalog handlers get or pur methods.

        NOTE: Most often, you should not be over-riding this.
        Custom funtionality can be obtained by working on catalog handler implementation.
        """
        super().sync_catalog(node, step_log, stage)

    def execute_node(self, node: BaseNode, map_variable: str = '', **kwargs):
        """
        This method does the actual execution of a task, as-is, success or fail node.

        NOTE: Most often, you should not be over-riding this.
        If you plan to, please look at the implementation of the BaseExecutor to understand the nuances.
        """
        super().execute_node(node, map_variable=map_variable, **kwargs)

        step_log = self.run_log_store.get_step_log(node.get_step_log_name(map_variable), self.run_id)
        if step_log.status == defaults.FAIL:
            raise Exception(f'Step {node.name} failed')

    def add_code_identities(self, node: BaseNode, step_log: object):
        """
        Add code identities specific to the implementation.

        The Base class has an implementation of adding git code identities.

        Args:
            step_log (object): The step log object
            node (BaseNode): The node we are adding the step log for

        NOTE: Most often, you just call the super to add the git code identity and add
        any other code identities that you want part of your implementation
        """
        super().add_code_identities(node, step_log)

    def execute_from_graph(self, node: BaseNode, map_variable: str = '', **kwargs):
        """
        This method delegates the execution of composite nodes to the appropriate methods.

        This method calls add_code_identities and trigger_job as part of its implemetation.
        use them to add the funcionality specific to the compute environment.

        NOTE: Most often, you should not be changing this implementation.
        """
        super().execute_from_graph(node=node, map_variable=map_variable, **kwargs)

    def trigger_job(self, node: BaseNode, map_variable: str = '', **kwargs):
        """
        Executor specific way of triggering jobs.

        This method has to be changed to do what exactly you want as part of your computational engine

        If your compute is not local, use utils.get_node_execution_command(self, node, map_variable=map_variable)
        to get the command to run a single node.

        If the compute is local to the environment, calls prepare_for_node_execution and call execute_node
        NOTE: This method should always be implemented.
        """
        self.prepare_for_node_execution(node, map_variable=map_variable)
        self.execute_node(node=node, map_variable=map_variable, **kwargs)

    def execute_graph(self, dag: Graph, map_variable: str = '', **kwargs):
        """
        Iterate through the graph and frame the bash script.

        For more complex ouputs, dataclasses might be a better option.

        NOTE: This method should be over-written to write the exact specification to the compute engine.

        """
        current_node = dag.start_at
        previous_node = None
        logger.info(f'Rendering job started at {current_node}')
        bash_script_lines = []

        while True:
            working_on = dag.get_node_by_name(current_node)

            if working_on.node_type in ['parallel', 'dag', 'map']:
                raise NotImplementedError('In this demo version, composite nodes are not implemented')

            if previous_node == current_node:
                raise Exception('Potentially running in a infinite loop')

            previous_node = current_node

            logger.info(f'Creating execution log for {working_on}')

            execute_node_command = utils.get_node_execution_command(self, working_on, over_write_run_id='$1')
            current_job_id = re.sub('[^A-Za-z0-9]+', '', f'{current_node}_job_id')
            fail_node_command = utils.get_node_execution_command(self, dag.get_fail_node(), over_write_run_id='$1')

            if working_on.node_type not in ['success', 'fail']:
                bash_script_lines.append(f'{current_job_id}=$({execute_node_command})\n')

                # Write failure node
                bash_script_lines.append(
                    (
                        f'if [${current_job_id}]\nthen\n'
                        f'\t $({fail_node_command})\n'
                        '\texit 1\n'
                        'fi\n'
                    )
                )

            if working_on.node_type == 'success':
                bash_script_lines.append(f'${current_job_id}=$({execute_node_command})')
            if working_on.node_type in ['success', 'fail']:
                break

            current_node = working_on.get_next_node()

        with open('demo-bash.sh', 'w', encoding='utf-8') as fw:
            fw.writelines(bash_script_lines)

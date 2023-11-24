# pragma: no cover
import logging
import re

from magnus import defaults, utils
from magnus.defaults import TypeMapVariable
from magnus.extensions.executor import GenericExecutor
from magnus.extensions.nodes import StubNode
from magnus.graph import Graph
from magnus.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class DemoRenderer(GenericExecutor):
    """
    This renderer is an example of how you can render required job specifications as per your orchestration tool.

    BaseExecutor implements many of the functionalities that are common and can be safe defaults.
    In this renderer example: We just render a bash script that sequentially calls the steps.
    We do not handle composite steps in this execution type.

    Example config:
    executor:
      type: demo-renderer
    """

    service_name: str = "demo-renderer"

    def execute_node(self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs):
        """
        This method does the actual execution of a task, as-is, success or fail node.
        """
        self._set_up_run_log(exists_ok=True)
        # Need to set up the step log for the node as the entry point is different
        step_log = self._context.run_log_store.create_step_log(node.name, node._get_step_log_name(map_variable))

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        super()._execute_node(node, map_variable=map_variable, **kwargs)

        step_log = self._context.run_log_store.get_step_log(node._get_step_log_name(map_variable), self._context.run_id)
        if step_log.status == defaults.FAIL:
            raise Exception(f"Step {node.name} failed")

    def send_return_code(self, stage="traversal"):
        """
        Convenience function used by pipeline to send return code to the caller of the cli

        Raises:
            Exception: If the pipeline execution failed
        """
        if stage != "traversal":  # traversal does no actual execution, so return code is pointless
            run_id = self._context.run_id

            run_log = self._context.run_log_store.get_run_log_by_id(run_id=run_id, full=False)
            if run_log.status == defaults.FAIL:
                raise Exception("Pipeline execution failed")

    def execute_graph(self, dag: Graph, map_variable: TypeMapVariable = None, **kwargs):
        """
        Iterate through the graph and frame the bash script.

        For more complex outputs, dataclasses might be a better option.

        NOTE: This method should be over-written to write the exact specification to the compute engine.

        """
        current_node = dag.start_at
        previous_node = None
        logger.info(f"Rendering job started at {current_node}")
        bash_script_lines = []

        while True:
            working_on = dag.get_node_by_name(current_node)

            if working_on.is_composite:
                raise NotImplementedError("In this demo version, composite nodes are not implemented")

            if working_on.node_type == StubNode.node_type:
                raise NotImplementedError("In this demo version, AsIs nodes are not implemented")

            if previous_node == current_node:
                raise Exception("Potentially running in a infinite loop")

            previous_node = current_node

            logger.info(f"Creating execution log for {working_on}")

            _execute_node_command = utils.get_node_execution_command(working_on, over_write_run_id="$1")
            re.sub("[^A-Za-z0-9]+", "", f"{current_node}_job_id")
            fail_node_command = utils.get_node_execution_command(dag.get_fail_node(), over_write_run_id="$1")

            if working_on.node_type not in ["success", "fail"]:
                bash_script_lines.append(f"{_execute_node_command}\n")

                bash_script_lines.append("exit_code=$?\necho $exit_code\n")
                # Write failure node
                bash_script_lines.append(
                    ("if [ $exit_code -ne 0 ];\nthen\n" f"\t $({fail_node_command})\n" "\texit 1\n" "fi\n")
                )

            if working_on.node_type == "success":
                bash_script_lines.append(f"{_execute_node_command}")
            if working_on.node_type in ["success", "fail"]:
                break

            current_node = working_on._get_next_node()

        with open("demo-bash.sh", "w", encoding="utf-8") as fw:
            fw.writelines(bash_script_lines)

        msg = (
            "demo-bash.sh for running the pipeline is written. To execute it \n"
            "1). Activate the environment:\n"
            "\t for example poetry shell or pipenv shell etc\n"
            "2). Make the shell script executable.\n"
            "\t chmod 755 demo-bash.sh\n"
            "3). Run the script by: source demo-bash.sh <run_id>\n"
            "\t The first argument to the script is the run id you want for the run."
        )
        logger.info(msg)

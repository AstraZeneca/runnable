import logging
from pathlib import Path
from typing import Dict

from pydantic import Field
from rich import print

from extensions.pipeline_executor import GenericPipelineExecutor
from runnable import console, defaults, task_console, utils
from runnable.datastore import StepLog
from runnable.defaults import TypeMapVariable
from runnable.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalContainerExecutor(GenericPipelineExecutor):
    """
    In the mode of local-container, we execute all the commands in a container.

    Ensure that the local compute has enough resources to finish all your jobs.

    The image of the run, could either be provided as default in the configuration of the execution engine
    i.e.:
    execution:
      type: 'local-container'
      config:
        docker_image: the image you want the code to run in.

    or default image could be over-ridden for a single node by providing a docker_image in the step config.
    i.e:
    dag:
      steps:
        step:
          executor_config:
            local-container:
                docker_image: The image that you want that single step to run in.
    This image would only be used for that step only.

    This mode does not build the docker image with the latest code for you, it is still left for the user to build
    and ensure that the docker image provided is the correct one.

    Example config:
    execution:
      type: local-container
      config:
        docker_image: The default docker image to use if the node does not provide one.
    """

    service_name: str = "local-container"
    docker_image: str
    auto_remove_container: bool = True
    environment: Dict[str, str] = Field(default_factory=dict)

    _is_local: bool = False

    _container_log_location = "/tmp/run_logs/"
    _container_catalog_location = "/tmp/catalog/"
    _container_secrets_location = "/tmp/dotenv"
    _volumes: Dict[str, Dict[str, str]] = {}

    def add_code_identities(self, node: BaseNode, step_log: StepLog, **kwargs):
        """
        Call the Base class to add the git code identity and add docker identity

        Args:
            node (BaseNode): The node we are adding the code identity
            step_log (Object): The step log corresponding to the node
        """

        super().add_code_identities(node, step_log)

        if node.node_type in ["success", "fail"]:
            # Need not add code identities if we are in a success or fail node
            return

        executor_config = self._resolve_executor_config(node)

        docker_image = executor_config.get("docker_image", None)
        if docker_image:
            code_id = self._context.run_log_store.create_code_identity()

            code_id.code_identifier = utils.get_local_docker_image_id(docker_image)
            code_id.code_identifier_type = "docker"
            code_id.code_identifier_dependable = True
            code_id.code_identifier_url = "local docker host"
            step_log.code_identities.append(code_id)

    def execute_node(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
        """
        We are already in the container, we just execute the node.
        The node is already prepared for execution.
        """
        self._use_volumes()
        return self._execute_node(node, map_variable, **kwargs)

    def execute_from_graph(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
        """
        This is the entry point to from the graph execution.

        While the self.execute_graph is responsible for traversing the graph, this function is responsible for
        actual execution of the node.

        If the node type is:
            * task : We can delegate to _execute_node after checking the eligibility for re-run in cases of a re-run
            * success: We can delegate to _execute_node
            * fail: We can delegate to _execute_node

        For nodes that are internally graphs:
            * parallel: Delegate the responsibility of execution to the node.execute_as_graph()
            * dag: Delegate the responsibility of execution to the node.execute_as_graph()
            * map: Delegate the responsibility of execution to the node.execute_as_graph()

        Transpilers will NEVER use this method and will NEVER call ths method.
        This method should only be used by interactive executors.

        Args:
            node (Node): The node to execute
            map_variable (dict, optional): If the node if of a map state, this corresponds to the value of iterable.
                    Defaults to None.
        """
        step_log = self._context.run_log_store.create_step_log(
            node.name, node._get_step_log_name(map_variable)
        )

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        logger.info(f"Executing node: {node.get_summary()}")

        # Add the step log to the database as per the situation.
        # If its a terminal node, complete it now
        if node.node_type in ["success", "fail"]:
            self._execute_node(node, map_variable=map_variable, **kwargs)
            return

        # We call an internal function to iterate the sub graphs and execute them
        if node.is_composite:
            node.execute_as_graph(map_variable=map_variable, **kwargs)
            return

        task_console.export_text(clear=True)

        task_name = node._resolve_map_placeholders(node.internal_name, map_variable)
        console.print(
            f":runner: Executing the node {task_name} ... ", style="bold color(208)"
        )
        self.trigger_node_execution(node=node, map_variable=map_variable, **kwargs)

    def trigger_node_execution(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
        """
        We come into this step via execute from graph, use trigger job to spin up the container.

        In local container execution, we just spin the container to execute runnable execute_single_node.

        Args:
            node (BaseNode): The node we are currently executing
            map_variable (str, optional): If the node is part of the map branch. Defaults to ''.
        """
        self._mount_volumes()
        executor_config = self._resolve_executor_config(node)
        auto_remove_container = executor_config.get("auto_remove_container", True)

        logger.debug("Here is the resolved executor config")
        logger.debug(executor_config)

        command = utils.get_node_execution_command(node, map_variable=map_variable)

        self._spin_container(
            node=node,
            command=command,
            map_variable=map_variable,
            auto_remove_container=auto_remove_container,
            **kwargs,
        )

        step_log = self._context.run_log_store.get_step_log(
            node._get_step_log_name(map_variable), self._context.run_id
        )
        if step_log.status != defaults.SUCCESS:
            msg = (
                "Node execution inside the container failed. Please check the logs.\n"
                "Note: If you do not see any docker issue from your side and the code works properly on local execution"
                "please raise a bug report."
            )
            logger.error(msg)
            step_log.status = defaults.FAIL
            self._context.run_log_store.add_step_log(step_log, self._context.run_id)

    def _spin_container(
        self,
        node: BaseNode,
        command: str,
        map_variable: TypeMapVariable = None,
        auto_remove_container: bool = True,
        **kwargs,
    ):
        """
        During the flow run, we have to spin up a container with the docker image mentioned
        and the right log locations
        """
        # Conditional import
        import docker  # pylint: disable=C0415

        try:
            client = docker.from_env()
            api_client = docker.APIClient()
        except Exception as ex:
            logger.exception("Could not get access to docker")
            raise Exception(
                "Could not get the docker socket file, do you have docker installed?"
            ) from ex

        try:
            logger.info(f"Running the command {command}")
            print(command)
            #  Overrides global config with local
            executor_config = self._resolve_executor_config(node)

            docker_image = executor_config.get("docker_image", None)
            environment = executor_config.get("environment", {})
            environment.update(self._context.variables)
            if not docker_image:
                raise Exception(
                    f"Please provide a docker_image using executor_config of the step {node.name} or at global config"
                )

            container = client.containers.create(
                image=docker_image,
                command=command,
                auto_remove=False,
                volumes=self._volumes,
                network_mode="host",
                environment=environment,
            )

            # print(container.__dict__)

            container.start()
            stream = api_client.logs(
                container=container.id, timestamps=True, stream=True, follow=True
            )
            while True:
                try:
                    output = next(stream).decode("utf-8")
                    output = output.strip("\r\n")
                    logger.info(output)
                    print(output)
                except StopIteration:
                    logger.info("Docker Run completed")
                    break

            exit_status = api_client.inspect_container(container.id)["State"][
                "ExitCode"
            ]

            if auto_remove_container:
                container.remove(force=True)

            if exit_status != 0:
                msg = f"Docker command failed with exit code {exit_status}"
                raise Exception(msg)

        except Exception as _e:
            logger.exception("Problems with spinning/running the container")
            raise _e

    def _mount_volumes(self):
        """
        Mount the volumes for the container
        """
        match self._context.run_log_store.service_name:
            case "file-system":
                write_to = self._context.run_log_store.log_folder
                self._volumes[str(Path(write_to).resolve())] = {
                    "bind": f"{self._container_log_location}",
                    "mode": "rw",
                }
            case "chunked-fs":
                write_to = self._context.run_log_store.log_folder
                self._volumes[str(Path(write_to).resolve())] = {
                    "bind": f"{self._container_log_location}",
                    "mode": "rw",
                }

        match self._context.catalog_handler.service_name:
            case "file-system":
                catalog_location = self._context.catalog_handler.catalog_location
                self._volumes[str(Path(catalog_location).resolve())] = {
                    "bind": f"{self._container_catalog_location}",
                    "mode": "rw",
                }

        match self._context.secrets_handler.service_name:
            case "dotenv":
                secrets_location = self._context.secrets_handler.location
                self._volumes[str(Path(secrets_location).resolve())] = {
                    "bind": f"{self._container_secrets_location}",
                    "mode": "ro",
                }

    def _use_volumes(self):
        match self._context.run_log_store.service_name:
            case "file-system":
                self._context.run_log_store.log_folder = self._container_log_location
            case "chunked-fs":
                self._context.run_log_store.log_folder = self._container_log_location

        match self._context.catalog_handler.service_name:
            case "file-system":
                self._context.catalog_handler.catalog_location = (
                    self._container_catalog_location
                )

        match self._context.secrets_handler.service_name:
            case "dotenv":
                self._context.secrets_handler.location = (
                    self._container_secrets_location
                )

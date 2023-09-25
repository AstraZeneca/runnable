import logging
from typing import Dict, Optional, cast

from magnus import defaults, integration, utils
from magnus.datastore import StepLog
from magnus.extensions.executor import GenericExecutor
from magnus.extensions.nodes import TaskNode
from magnus.nodes import BaseNode
from magnus.tasks import ContainerTaskType

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalContainerExecutor(GenericExecutor):
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
        mode_config = self._resolve_executor_config(node)

        docker_image = mode_config.get("docker_image", None)
        if docker_image:
            code_id = self._context.run_log_store.create_code_identity()

            code_id.code_identifier = utils.get_local_docker_image_id(docker_image)
            code_id.code_identifier_type = "docker"
            code_id.code_identifier_dependable = True
            code_id.code_identifier_url = "local docker host"
            step_log.code_identities.append(code_id)

    def execute_node(self, node: BaseNode, map_variable: Optional[dict] = None, **kwargs):
        """
        We are already in the container, we just execute the node.
        """
        return self._execute_node(node, map_variable, **kwargs)

    def execute_job(self, node: TaskNode):
        """
        Set up the step log and call the execute node

        Args:
            node (BaseNode): _description_
        """
        from magnus import integration
        from magnus.tasks import ContainerTaskType

        step_log = self._context.run_log_store.create_step_log(node.name, node._get_step_log_name(map_variable=None))

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        if node.executable.task_type == ContainerTaskType.task_type:
            # Do not change config but only validate the configuration.
            # Trigger the job on local system instead of a container
            # Or if the task type is a container, just spin the container.
            integration.validate(self, self._context.run_log_store)
            integration.validate(self, self._context.catalog_handler)
            integration.validate(self, self._context.secrets_handler)

            self.execute_node(node=node, map_variable={})
        else:
            command = utils.get_job_execution_command(node)
            self._spin_container(node=node, command=command)

        # Check the step log status and warn if necessary. Docker errors are generally suppressed.
        step_log = self._context.run_log_store.get_step_log(
            node._get_step_log_name(map_variable=None), self._context.run_id
        )
        if step_log.status != defaults.SUCCESS:
            msg = (
                "Node execution inside the container failed. Please check the logs.\n"
                "Note: If you do not see any docker issue from your side and the code works properly on local execution"
                "please raise a bug report."
            )
            logger.warning(msg)

    def trigger_job(self, node: BaseNode, map_variable: Optional[dict] = None, **kwargs):
        """
        If the config has "run_in_local: True", we compute it on local system instead of container.
        In local container execution, we just spin the container to execute magnus execute_single_node.

        Args:
            node (BaseNode): The node we are currently executing
            map_variable (str, optional): If the node is part of the map branch. Defaults to ''.
        """
        executor_config = self._resolve_executor_config(node)

        logger.debug("Here is the resolved executor config")
        logger.debug(executor_config)

        if executor_config.get("run_in_local", None) or (
            cast(TaskNode, node).executable.task_type == ContainerTaskType.task_type
        ):
            # Do not change config but only validate the configuration.
            # Trigger the job on local system instead of a container
            # Or if the task type is a container, just spin the container.
            integration.validate(self, self._context.run_log_store)
            integration.validate(self, self._context.catalog_handler)
            integration.validate(self, self._context.secrets_handler)

            self.execute_node(node=node, map_variable=map_variable, **kwargs)
            return

        command = utils.get_node_execution_command(node, map_variable=map_variable)
        self._spin_container(node=node, command=command, map_variable=map_variable, **kwargs)

        # Check for the status of the node log and anything apart from Success is FAIL
        # This typically happens if something is wrong with magnus or settings.
        step_log = self._context.run_log_store.get_step_log(node._get_step_log_name(map_variable), self._context.run_id)
        if step_log.status != defaults.SUCCESS:
            msg = (
                "Node execution inside the container failed. Please check the logs.\n"
                "Note: If you do not see any docker issue from your side and the code works properly on local execution"
                "please raise a bug report."
            )
            logger.warning(msg)
            step_log.status = defaults.FAIL
            self._context.run_log_store.add_step_log(step_log, self._context.run_id)

    def _spin_container(self, node: BaseNode, command: str, map_variable: Optional[dict] = None, **kwargs):
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
            raise Exception("Could not get the docker socket file, do you have docker installed?") from ex

        try:
            logger.info(f"Running the command {command}")
            #  Overrides global config with local
            executor_config = self._resolve_executor_config(node)
            docker_image = executor_config.get("docker_image", None)
            environment = executor_config.get("environment", {})
            environment.update(self._context.variables)
            if not docker_image:
                raise Exception(
                    f"Please provide a docker_image using executor_config of the step {node.name} or at global config"
                )

            # TODO: Should consider using getpass.getuser() when running the docker container? Volume permissions
            container = client.containers.create(
                image=docker_image,
                command=command,
                auto_remove=False,
                volumes=self._volumes,
                network_mode="host",
                environment=environment,
            )
            container.start()
            stream = api_client.logs(container=container.id, timestamps=True, stream=True, follow=True)
            while True:
                try:
                    output = next(stream).decode("utf-8")
                    output = output.strip("\r\n")
                    logger.info(output)
                except StopIteration:
                    logger.info("Docker Run completed")
                    break
            exit_status = api_client.inspect_container(container.id)["State"]["ExitCode"]
            container.remove(force=True)
            if exit_status != 0:
                msg = f"Docker command failed with exit code {exit_status}"
                raise Exception(msg)

        except Exception as _e:
            logger.exception("Problems with spinning/running the container")
            raise _e

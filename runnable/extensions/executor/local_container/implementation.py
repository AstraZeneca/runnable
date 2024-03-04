import logging
from pathlib import Path
from typing import Dict, cast

from pydantic import Field
from rich import print

from runnable import defaults, integration, utils
from runnable.datastore import StepLog
from runnable.defaults import TypeMapVariable
from runnable.extensions.executor import GenericExecutor
from runnable.extensions.nodes import TaskNode
from runnable.integration import BaseIntegration
from runnable.nodes import BaseNode

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
    auto_remove_container: bool = True
    run_in_local: bool = False
    environment: Dict[str, str] = Field(default_factory=dict)

    _local: bool = False

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

    def execute_node(self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs):
        """
        We are already in the container, we just execute the node.
        The node is already prepared for execution.
        """
        return self._execute_node(node, map_variable, **kwargs)

    def execute_job(self, node: TaskNode):
        """
        Set up the step log and call the execute node

        Args:
            node (BaseNode): _description_
        """

        step_log = self._context.run_log_store.create_step_log(node.name, node._get_step_log_name(map_variable=None))

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

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

    def trigger_job(self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs):
        """
        We come into this step via execute from graph, use trigger job to spin up the container.


        If the config has "run_in_local: True", we compute it on local system instead of container.
        In local container execution, we just spin the container to execute runnable execute_single_node.

        Args:
            node (BaseNode): The node we are currently executing
            map_variable (str, optional): If the node is part of the map branch. Defaults to ''.
        """
        executor_config = self._resolve_executor_config(node)
        auto_remove_container = executor_config.get("auto_remove_container", True)

        logger.debug("Here is the resolved executor config")
        logger.debug(executor_config)

        if executor_config.get("run_in_local", False):
            # Do not change config but only validate the configuration.
            # Trigger the job on local system instead of a container
            integration.validate(self, self._context.run_log_store)
            integration.validate(self, self._context.catalog_handler)
            integration.validate(self, self._context.secrets_handler)

            self.execute_node(node=node, map_variable=map_variable, **kwargs)
            return

        command = utils.get_node_execution_command(node, map_variable=map_variable)

        self._spin_container(
            node=node,
            command=command,
            map_variable=map_variable,
            auto_remove_container=auto_remove_container,
            **kwargs,
        )

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
            raise Exception("Could not get the docker socket file, do you have docker installed?") from ex

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

            # TODO: Should consider using getpass.getuser() when running the docker container? Volume permissions
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
            stream = api_client.logs(container=container.id, timestamps=True, stream=True, follow=True)
            while True:
                try:
                    output = next(stream).decode("utf-8")
                    output = output.strip("\r\n")
                    logger.info(output)
                    print(output)
                except StopIteration:
                    logger.info("Docker Run completed")
                    break

            exit_status = api_client.inspect_container(container.id)["State"]["ExitCode"]

            if auto_remove_container:
                container.remove(force=True)

            if exit_status != 0:
                msg = f"Docker command failed with exit code {exit_status}"
                raise Exception(msg)

        except Exception as _e:
            logger.exception("Problems with spinning/running the container")
            raise _e


class LocalContainerComputeFileSystemRunLogstore(BaseIntegration):
    """
    Integration between local container and file system run log store
    """

    executor_type = "local-container"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "file-system"  # The actual implementation of the service

    def configure_for_traversal(self, **kwargs):
        from runnable.extensions.run_log_store.file_system.implementation import FileSystemRunLogstore

        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(FileSystemRunLogstore, self.service)

        write_to = self.service.log_folder_name
        self.executor._volumes[str(Path(write_to).resolve())] = {
            "bind": f"{self.executor._container_log_location}",
            "mode": "rw",
        }

    def configure_for_execution(self, **kwargs):
        from runnable.extensions.run_log_store.file_system.implementation import FileSystemRunLogstore

        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(FileSystemRunLogstore, self.service)

        self.service.log_folder = self.executor._container_log_location


class LocalContainerComputeFileSystemCatalog(BaseIntegration):
    """
    Integration pattern between Local container and File System catalog
    """

    executor_type = "local-container"
    service_type = "catalog"  # One of secret, catalog, datastore
    service_provider = "file-system"  # The actual implementation of the service

    def configure_for_traversal(self, **kwargs):
        from runnable.extensions.catalog.file_system.implementation import FileSystemCatalog

        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(FileSystemCatalog, self.service)

        catalog_location = self.service.catalog_location
        self.executor._volumes[str(Path(catalog_location).resolve())] = {
            "bind": f"{self.executor._container_catalog_location}",
            "mode": "rw",
        }

    def configure_for_execution(self, **kwargs):
        from runnable.extensions.catalog.file_system.implementation import FileSystemCatalog

        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(FileSystemCatalog, self.service)

        self.service.catalog_location = self.executor._container_catalog_location


class LocalContainerComputeDotEnvSecrets(BaseIntegration):
    """
    Integration between local container and dot env secrets
    """

    executor_type = "local-container"
    service_type = "secrets"  # One of secret, catalog, datastore
    service_provider = "dotenv"  # The actual implementation of the service

    def validate(self, **kwargs):
        logger.warning("Using dot env for non local deployments is not ideal, consider options")

    def configure_for_traversal(self, **kwargs):
        from runnable.extensions.secrets.dotenv.implementation import DotEnvSecrets

        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(DotEnvSecrets, self.service)

        secrets_location = self.service.secrets_location
        self.executor._volumes[str(Path(secrets_location).resolve())] = {
            "bind": f"{self.executor._container_secrets_location}",
            "mode": "ro",
        }

    def configure_for_execution(self, **kwargs):
        from runnable.extensions.secrets.dotenv.implementation import DotEnvSecrets

        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(DotEnvSecrets, self.service)

        self.service.location = self.executor._container_secrets_location


class LocalContainerComputeEnvSecretsManager(BaseIntegration):
    """
    Integration between local container and env secrets manager
    """

    executor_type = "local-container"
    service_type = "secrets"  # One of secret, catalog, datastore
    service_provider = "env-secrets-manager"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            "Local container executions cannot be used with environment secrets manager. "
            "Please use a supported secrets manager"
        )
        logger.exception(msg)
        raise Exception(msg)

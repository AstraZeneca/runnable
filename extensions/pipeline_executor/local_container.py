import logging
import os
from pathlib import Path
from typing import Dict, Optional

from pydantic import Field, PrivateAttr

from extensions.pipeline_executor import GenericPipelineExecutor
from runnable import defaults
from runnable.datastore import StepAttempt
from runnable.defaults import IterableParameterModel
from runnable.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalContainerExecutor(GenericPipelineExecutor):
    """
    In the mode of local-container, we execute all the commands in a container.

    Ensure that the local compute has enough resources to finish all your jobs.

    Configuration options:

    ```yaml
    pipeline-executor:
      type: local-container
      config:
        docker_image: <required>
        auto_remove_container: true/false
        environment:
          key: value
        overrides:
          alternate_config:
            docker_image: <required>
            auto_remove_container: true/false
            environment:
              key: value
    ```

    - ```docker_image```: The default docker image to use for all the steps.
    - ```auto_remove_container```: Remove container after execution
    - ```environment```: Environment variables to pass to the container

    Overrides give you the ability to override the default docker image for a single step.
    A step can then then refer to the alternate_config in the task definition.

    Example:

    ```python
    from runnable import PythonTask

    task = PythonTask(
        name="alt_task",
        overrides={
            "local-container": "alternate_config"
            }
        )
    ```

    In the above example, ```alt_task``` will run in the docker image/configuration
    as defined in the alternate_config.

    ```runnable``` does not build the docker image for you, it is still left for the user to build
    and ensure that the docker image provided is the correct one.

    """

    service_name: str = "local-container"
    enable_parallel: bool = Field(default=False)

    docker_image: str
    auto_remove_container: bool = True
    environment: Dict[str, str] = Field(default_factory=dict)

    _should_setup_run_log_at_traversal: bool = PrivateAttr(default=True)

    _container_log_location = "/tmp/run_logs/"
    _container_catalog_location = "/tmp/catalog/"
    _container_secrets_location = "/tmp/dotenv"
    _volumes: Dict[str, Dict[str, str]] = {}

    def _get_docker_image_digest(self, docker_image: str) -> str | None:
        """
        Retrieve the docker image digest, trying local first, then pulling if needed.

        Args:
            docker_image: The docker image name/tag

        Returns:
            The image digest (sha256:...) or None if retrieval fails
        """
        import docker  # pylint: disable=C0415

        try:
            client = docker.from_env()

            # Try to get digest from local image first
            try:
                image = client.images.get(docker_image)
                # Get the RepoDigest which contains the sha256 digest
                if image.attrs.get("RepoDigests"):
                    # RepoDigests is a list like ["registry/repo@sha256:..."]
                    for digest in image.attrs["RepoDigests"]:
                        if "@sha256:" in digest:
                            return digest.split("@")[1]  # Return just "sha256:..."

                # If no RepoDigest, try to get the image ID (less ideal but better than nothing)
                if image.id:
                    return image.id

            except docker.errors.ImageNotFound:
                # Image not found locally, try to pull it
                logger.info(
                    f"Docker image {docker_image} not found locally, pulling..."
                )
                try:
                    pulled_image = client.images.pull(docker_image)

                    # Get digest from pulled image
                    if pulled_image.attrs.get("RepoDigests"):
                        for digest in pulled_image.attrs["RepoDigests"]:
                            if "@sha256:" in digest:
                                return digest.split("@")[1]  # Return just "sha256:..."

                    if pulled_image.id:
                        return pulled_image.id

                except Exception as pull_ex:
                    logger.warning(
                        f"Failed to pull docker image {docker_image}: {pull_ex}"
                    )

        except Exception as ex:
            logger.warning(
                f"Failed to retrieve docker image digest for {docker_image}: {ex}"
            )

        return None

    def add_code_identities(self, node: BaseNode, attempt_log: StepAttempt):
        """
        Call the Base class to add the git code identity and add docker identity

        Args:
            node (BaseNode): The node we are adding the code identity
            attempt_log (StepAttempt): The step attempt log corresponding to the node
        """

        super().add_code_identities(node, attempt_log)

        if node.node_type in ["success", "fail"]:
            # Need not add code identities if we are in a success or fail node
            return

        # Add docker image digest as code identity if available, fall back to image name
        docker_digest = os.getenv("RUNNABLE_CODE_ID_DOCKER_IMAGE_DIGEST")

        if not docker_digest:
            # Fall back to docker image name if digest not available
            executor_config = self._resolve_executor_config(node)
            docker_digest = executor_config.get("docker_image", None)

        if docker_digest:
            code_id = self._context.run_log_store.create_code_identity()

            code_id.code_identifier = docker_digest
            code_id.code_identifier_type = "docker"
            code_id.code_identifier_dependable = True
            code_id.code_identifier_url = "local docker host"
            attempt_log.code_identities.append(code_id)

            logger.debug(f"Added docker image code identity: {docker_digest[:50]}...")

    def execute_node(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        We are already in the container, we just execute the node.
        The node is already prepared for execution.
        """
        self._use_volumes()
        return self._execute_node(node, iter_variable)

    def trigger_node_execution(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        We come into this step via execute from graph, use trigger job to spin up the container.

        In local container execution, we just spin the container to execute runnable execute_single_node.

        Args:
            node (BaseNode): The node we are currently executing
            iter_variable (str, optional): If the node is part of the map branch. Defaults to ''.
        """
        self._mount_volumes()
        executor_config = self._resolve_executor_config(node)
        auto_remove_container = executor_config.get("auto_remove_container", True)

        logger.debug("Here is the resolved executor config")
        logger.debug(executor_config)

        command = self._context.get_node_callable_command(
            node, iter_variable=iter_variable
        )

        self._spin_container(
            node=node,
            command=command,
            iter_variable=iter_variable,
            auto_remove_container=auto_remove_container,
        )

        step_log = self._context.run_log_store.get_step_log(
            node._get_step_log_name(iter_variable), self._context.run_id
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
            raise Exception(msg)

    def _spin_container(
        self,
        node: BaseNode,
        command: str,
        iter_variable: Optional[IterableParameterModel] = None,
        auto_remove_container: bool = True,
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
            #  Overrides global config with local
            executor_config = self._resolve_executor_config(node)

            docker_image = executor_config.get("docker_image", None)
            environment = executor_config.get("environment", {})
            environment.update(self._context.variables)
            if not docker_image:
                raise Exception(
                    f"Please provide a docker_image using executor_config of the step {node.name} or at global config"
                )

            # Retrieve docker image digest and pass it as environment variable
            digest = self._get_docker_image_digest(docker_image)
            if digest:
                environment["RUNNABLE_CODE_ID_DOCKER_IMAGE_DIGEST"] = digest
                logger.info(f"Retrieved docker image digest: {digest[:12]}...")
            else:
                logger.warning(
                    f"Could not retrieve digest for docker image: {docker_image}"
                )

            container = client.containers.create(
                image=docker_image,
                command=command,
                auto_remove=False,
                volumes=self._volumes,
                # network_mode="host",
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
        # TODO: There should be an abstraction on top of service providers
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

        match self._context.catalog.service_name:
            case "file-system":
                catalog_location = self._context.catalog.catalog_location
                self._volumes[str(Path(catalog_location).resolve())] = {
                    "bind": f"{self._container_catalog_location}",
                    "mode": "rw",
                }

        match self._context.secrets.service_name:
            case "dotenv":
                secrets_location = self._context.secrets.location
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

        match self._context.catalog.service_name:
            case "file-system":
                self._context.catalog.catalog_location = (
                    self._container_catalog_location
                )

        match self._context.secrets.service_name:
            case "dotenv":
                self._context.secrets.location = self._container_secrets_location

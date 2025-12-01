import logging
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, PrivateAttr

from extensions.job_executor import GenericJobExecutor
from runnable import context, defaults
from runnable.tasks import BaseTaskType

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalContainerJobExecutor(GenericJobExecutor):
    """
    The LocalJobExecutor is a job executor that runs the job locally.
    """

    service_name: str = "local-container"
    docker_image: str
    auto_remove_container: bool = True
    environment: Dict[str, str] = Field(default_factory=dict)

    _should_setup_run_log_at_traversal: bool = PrivateAttr(default=True)

    _container_log_location = "/tmp/run_logs/"
    _container_catalog_location = "/tmp/catalog/"
    _container_secrets_location = "/tmp/dotenv"
    _volumes: Dict[str, Dict[str, str]] = {}

    def submit_job(self, job: BaseTaskType, catalog_settings=Optional[List[str]]):
        """
        This method gets invoked by the CLI.
        """
        self._set_up_run_log()
        self._mount_volumes()

        # Call the container job
        job_log = self._context.run_log_store.create_job_log()
        self._context.run_log_store.add_job_log(
            run_id=self._context.run_id, job_log=job_log
        )
        self.spin_container()

    def execute_job(self, job: BaseTaskType, catalog_settings=Optional[List[str]]):
        """
        This method gets invoked by the CLI.
        """
        self._use_volumes()
        super().execute_job(job, catalog_settings=catalog_settings)

    def spin_container(self):
        """
        This method spins up the container
        """
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
            assert isinstance(self._context, context.JobContext)
            command = self._context.get_job_callable_command()
            logger.info(f"Running the command {command}")

            docker_image = self.docker_image
            environment = self.environment

            container = client.containers.create(
                image=docker_image,
                command=command,
                auto_remove=False,
                volumes=self._volumes,
                environment=environment,
            )

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

            if self.auto_remove_container:
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

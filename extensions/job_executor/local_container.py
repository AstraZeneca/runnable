import logging
from typing import Dict

from pydantic import Field

from extensions.job_executor import GenericJobExecutor
from runnable import console, defaults, utils
from runnable.tasks import BaseTaskType

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalContainerJobExecutor(GenericJobExecutor):
    """
    The LocalJobExecutor is a job executor that runs the job locally.
    """

    service_name: str = "local-container"
    docker_image: str
    mock: bool = False
    auto_remove_container: bool = True
    environment: Dict[str, str] = Field(default_factory=dict)

    _is_local: bool = False

    _container_log_location = "/tmp/run_logs/"
    _container_catalog_location = "/tmp/catalog/"
    _container_secrets_location = "/tmp/dotenv"
    _volumes: Dict[str, Dict[str, str]] = {}

    def submit_job(self, job: BaseTaskType):
        """
        This method gets invoked by the CLI.
        """
        # Call the container job
        job_log = self._context.run_log_store.create_job_log()
        self._context.run_log_store.add_job_log(
            run_id=self._context.run_id, job_log=job_log
        )
        self.execute_job(job)

    def pre_job_execution(self, job: BaseTaskType):
        """
        This method is called before the job execution.
        We are leaving this empty for local container
        """
        ...

    def execute_job(self, job: BaseTaskType):
        """
        Focusses on execution of the job.
        """
        self.prepare_for_execution()
        logger.info("Trying to execute job")

        job_log = self._context.run_log_store.get_job_log(run_id=self._context.run_id)

        attempt_log = job.execute_command(
            attempt_number=self.step_attempt_number,
            mock=self.mock,
        )

        job_log.status = attempt_log.status
        job_log.attempts.append(attempt_log)

        # data_catalogs_put: Optional[List[DataCatalog]] = self._sync_catalog(stage="put")
        # logger.debug(f"data_catalogs_put: {data_catalogs_put}")

        # step_log.add_data_catalogs(data_catalogs_put or [])

        console.print("Summary of job")
        console.print(job_log.get_summary())

        self._context.run_log_store.add_job_log(
            run_id=self._context.run_id, job_log=job_log
        )

    def post_job_execution(self, job: BaseTaskType):
        """
        This method is called after the job execution.
        Leaving it empty for local container
        """
        ...

    def _spin_contariner(self, task: BaseTaskType):
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
            command = utils.get_job_execution_command()
            logger.info(f"Running the command {command}")
            print(command)

            docker_image = self.docker_image
            environment = self.environment

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

            if self.auto_remove_container:
                container.remove(force=True)

            if exit_status != 0:
                msg = f"Docker command failed with exit code {exit_status}"
                raise Exception(msg)

        except Exception as _e:
            logger.exception("Problems with spinning/running the container")
            raise _e

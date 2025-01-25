import logging
import os
from typing import Dict, List, Optional

from runnable import context, defaults, exceptions, parameters, task_console, utils
from runnable.datastore import DataCatalog, JobLog, JsonParameter
from runnable.executor import BaseJobExecutor

logger = logging.getLogger(defaults.LOGGER_NAME)


class GenericJobExecutor(BaseJobExecutor):
    """
    The skeleton of an executor class.
    Any implementation of an executor should inherit this class and over-ride accordingly.

    This is a loaded base class which has a lot of methods already implemented for "typical" executions.
    Look at the function docs to understand how to use them appropriately.

    For any implementation:
    1). Who/when should the run log be set up?
    2). Who/When should the step log be set up?

    """

    service_name: str = ""
    service_type: str = "job_executor"

    @property
    def _context(self):
        assert context.run_context
        return context.run_context

    def _get_parameters(self) -> Dict[str, JsonParameter]:
        """
        Consolidate the parameters from the environment variables
        and the parameters file.

        The parameters defined in the environment variables take precedence over the parameters file.

        Returns:
            _type_: _description_
        """
        params: Dict[str, JsonParameter] = {}
        if self._context.parameters_file:
            user_defined = utils.load_yaml(self._context.parameters_file) or {}

            for key, value in user_defined.items():
                params[key] = JsonParameter(value=value, kind="json")

        # Update these with some from the environment variables
        params.update(parameters.get_user_set_parameters())
        logger.debug(f"parameters as seen by executor: {params}")
        return params

    def _set_up_run_log(self, exists_ok=False):
        """
        Create a run log and put that in the run log store

        If exists_ok, we allow the run log to be already present in the run log store.
        """
        try:
            attempt_run_log = self._context.run_log_store.get_run_log_by_id(
                run_id=self._context.run_id, full=False
            )

            logger.warning(
                f"The run log by id: {self._context.run_id} already exists, is this designed?"
            )
            raise exceptions.RunLogExistsError(
                f"The run log by id: {self._context.run_id} already exists and is {attempt_run_log.status}"
            )
        except exceptions.RunLogNotFoundError:
            pass
        except exceptions.RunLogExistsError:
            if exists_ok:
                return
            raise

        # Consolidate and get the parameters
        params = self._get_parameters()

        self._context.run_log_store.create_run_log(
            run_id=self._context.run_id,
            tag=self._context.tag,
            status=defaults.PROCESSING,
            dag_hash=self._context.dag_hash,
        )
        # Any interaction with run log store attributes should happen via API if available.
        self._context.run_log_store.set_parameters(
            run_id=self._context.run_id, parameters=params
        )

        # Update run_config
        run_config = utils.get_run_config()
        logger.debug(f"run_config as seen by executor: {run_config}")
        self._context.run_log_store.set_run_config(
            run_id=self._context.run_id, run_config=run_config
        )

    @property
    def step_attempt_number(self) -> int:
        """
        The attempt number of the current step.
        Orchestrators should use this step to submit multiple attempts of the job.

        Returns:
            int: The attempt number of the current step. Defaults to 1.
        """
        return int(os.environ.get(defaults.ATTEMPT_NUMBER, 1))

    def add_code_identities(self, job_log: JobLog, **kwargs):
        """
        Add code identities specific to the implementation.

        The Base class has an implementation of adding git code identities.

        Args:
            step_log (object): The step log object
            node (BaseNode): The node we are adding the step log for
        """
        job_log.code_identities.append(utils.get_git_code_identity())

    def send_return_code(self, stage="traversal"):
        """
        Convenience function used by pipeline to send return code to the caller of the cli

        Raises:
            Exception: If the pipeline execution failed
        """
        run_id = self._context.run_id

        run_log = self._context.run_log_store.get_run_log_by_id(
            run_id=run_id, full=False
        )
        if run_log.status == defaults.FAIL:
            raise exceptions.ExecutionFailedError(run_id=run_id)

    def _sync_catalog(
        self,
        catalog_settings=Optional[List[str]],
    ) -> List[DataCatalog] | None:
        if not catalog_settings:
            logger.info("No catalog settings found")
            return None

        compute_data_folder = self._context.catalog_handler.compute_data_folder

        data_catalogs = []
        for name_pattern in catalog_settings:
            data_catalog = self._context.catalog_handler.put(
                name=name_pattern,
                run_id=self._context.run_id,
                compute_data_folder=compute_data_folder,
            )

            logger.debug(f"Added data catalog: {data_catalog} to job log")
            data_catalogs.extend(data_catalog)

        return data_catalogs

    def add_task_log_to_catalog(
        self, name: str, map_variable: Dict[str, str | int | float] | None = None
    ):
        log_file_name = utils.make_log_file_name(
            name=name,
            map_variable=map_variable,
        )
        task_console.save_text(log_file_name)
        # Put the log file in the catalog
        self._context.catalog_handler.put(
            name=log_file_name, run_id=self._context.run_id
        )
        os.remove(log_file_name)

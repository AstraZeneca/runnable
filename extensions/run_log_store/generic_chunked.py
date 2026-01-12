import json
import logging
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Union

from runnable import defaults, exceptions
from runnable.datastore import (
    BaseRunLogStore,
    BranchLog,
    RunLog,
    StepLog,
)

logger = logging.getLogger(defaults.LOGGER_NAME)


class ChunkedRunLogStore(BaseRunLogStore):
    """
    A generic implementation of a RunLogStore that stores RunLogs in chunks.
    """

    service_name: str = ""
    supports_parallel_writes: bool = True

    class LogTypes(Enum):
        RUN_LOG = "RunLog"
        BRANCH_LOG = "BranchLog"

    class ModelTypes(Enum):
        RUN_LOG = RunLog
        BRANCH_LOG = BranchLog

    def get_file_name(self, log_type: LogTypes, name: str = "") -> str:
        """
        Get the exact file name for a log type.

        Args:
            log_type (LogTypes): Either RUN_LOG or BRANCH_LOG
            name (str, optional): The internal_branch_name for BranchLog. Defaults to ''.

        Raises:
            Exception: If log_type is not recognized or name is missing for BRANCH_LOG

        Returns:
            str: The exact file name
        """
        if log_type == self.LogTypes.RUN_LOG:
            return self.LogTypes.RUN_LOG.value

        if log_type == self.LogTypes.BRANCH_LOG:
            if not name:
                raise Exception("Name (internal_branch_name) required for BRANCH_LOG")
            return f"{self.LogTypes.BRANCH_LOG.value}-{name}"

        raise Exception(f"Unexpected log type: {log_type}")

    @abstractmethod
    def _exists(self, run_id: str, name: str) -> bool:
        """
        Check if a file exists in the persistence layer.

        Args:
            run_id (str): The run id
            name (str): The exact file name to check

        Returns:
            bool: True if file exists, False otherwise
        """
        ...

    @abstractmethod
    def _list_branch_logs(self, run_id: str) -> list[str]:
        """
        List all branch log file names for a run_id.

        Args:
            run_id (str): The run id

        Returns:
            list[str]: List of branch log file names (e.g., ["BranchLog-map.1", "BranchLog-map.2"])
        """
        ...

    @abstractmethod
    def _store(self, run_id: str, contents: dict, name: str, insert: bool = False):
        """
        Store the contents against the name in the persistence layer.

        Args:
            run_id (str): The run id
            contents (dict): The dict to store
            name (str): The name to store as
        """
        ...

    @abstractmethod
    def _retrieve(self, run_id: str, name: str) -> dict:
        """
        Does the job of retrieving from the persistent layer.

        Args:
            name (str): the name of the file to retrieve

        Returns:
            dict: The contents
        """
        ...

    def store(self, run_id: str, log_type: LogTypes, contents: dict, name: str = ""):
        """Store a log in the persistence layer.

        Args:
            run_id (str): The run id to store against
            log_type (LogTypes): The type of log to store (RUN_LOG or BRANCH_LOG)
            contents (dict): The dict of contents to store
            name (str, optional): The internal_branch_name for BRANCH_LOG. Defaults to ''.
        """
        file_name = self.get_file_name(log_type=log_type, name=name)

        # Check if file exists to determine if this is an update or insert
        insert = not self._exists(run_id=run_id, name=file_name)

        if not insert:
            # File exists - merge with existing contents
            existing_contents = self._retrieve(run_id=run_id, name=file_name)
            contents = dict(existing_contents, **contents)

        self._store(run_id=run_id, contents=contents, name=file_name, insert=insert)

    def retrieve(self, run_id: str, log_type: LogTypes, name: str = "") -> Any:
        """
        Retrieve a log model by type and name.

        Args:
            run_id (str): The run id
            log_type (LogTypes): Either RUN_LOG or BRANCH_LOG
            name (str, optional): The internal_branch_name for BRANCH_LOG. Defaults to ''.

        Raises:
            Exception: If name is missing for BRANCH_LOG
            EntityNotFoundError: If the file is not found

        Returns:
            Union[RunLog, BranchLog]: The requested log object
        """
        if log_type == self.LogTypes.BRANCH_LOG and not name:
            raise Exception("Name (internal_branch_name) required for BRANCH_LOG")

        file_name = self.get_file_name(log_type=log_type, name=name)

        if not self._exists(run_id=run_id, name=file_name):
            raise exceptions.EntityNotFoundError()

        contents = self._retrieve(run_id=run_id, name=file_name)
        model_class = self.ModelTypes[log_type.name].value
        return model_class.model_validate(contents)

    def _get_parent_branch(self, name: str) -> Union[str, None]:
        """
        Returns the name of the parent branch.
        If the step is part of main dag, return None.

        Args:
            name (str): The name of the step.

        Returns:
            str: The name of the branch containing the step.
        """
        dot_path = name.split(".")

        if len(dot_path) == 1:
            return None
        # Ignore the step name
        return ".".join(dot_path[:-1])

    def _get_parent_step(self, name: str) -> Union[str, None]:
        """
        Returns the step containing the step, useful when we have steps within a branch.
        Returns None, if the step belongs to parent dag.

        Args:
            name (str): The name of the step to find the parent step it belongs to.

        Returns:
            str: The parent step the step belongs to, None if the step belongs to parent dag.
        """
        dot_path = name.split(".")

        if len(dot_path) == 1:
            return None
        # Ignore the branch.step_name
        return ".".join(dot_path[:-2])

    def _prepare_full_run_log(self, run_log: RunLog):
        """
        Populate run log with branch logs.

        Since branches now contain their own steps and parameters,
        we just need to attach branches to their parent steps.
        """
        run_id = run_log.run_id

        # Get all branch log file names
        branch_file_names = self._list_branch_logs(run_id=run_id)
        if not branch_file_names:
            return

        # Load all branch logs
        branch_logs: Dict[str, BranchLog] = {}
        for file_name in branch_file_names:
            contents = self._retrieve(run_id=run_id, name=file_name)
            branch_log = BranchLog.model_validate(contents)
            branch_logs[branch_log.internal_name] = branch_log

        # Attach branches to their parent steps
        for branch_name, branch_log in branch_logs.items():
            # For a branch like "conditional.heads", parent step is "conditional"
            # For a branch like "map.a.nested", parent step is "map.a"
            dot_path = branch_name.split(".")
            if len(dot_path) < 2:
                # Branches must have at least step.branch format
                continue

            parent_step_name = ".".join(dot_path[:-1])

            # Find parent step (could be in run_log or another branch)
            parent_branch_name = self._get_parent_branch(parent_step_name)
            if parent_branch_name and parent_branch_name in branch_logs:
                parent_step = branch_logs[parent_branch_name].steps.get(
                    parent_step_name
                )
            else:
                parent_step = run_log.steps.get(parent_step_name)

            if parent_step:
                parent_step.branches[branch_name] = branch_log

    def create_run_log(
        self,
        run_id: str,
        dag_hash: str = "",
        use_cached: bool = False,
        tag: str = "",
        original_run_id: str = "",
        status: str = defaults.CREATED,
    ):
        """
        Creates a Run Log object by using the config

        Logically the method should do the following:
            * Creates a Run log
            * Adds it to the db
            * Return the log
        """
        try:
            self.get_run_log_by_id(run_id=run_id, full=False)
            raise exceptions.RunLogExistsError(run_id=run_id)
        except exceptions.RunLogNotFoundError:
            pass

        logger.info(f"{self.service_name} Creating a Run Log for : {run_id}")
        run_log = RunLog(
            run_id=run_id,
            dag_hash=dag_hash,
            tag=tag,
            status=status,
        )

        self.store(
            run_id=run_id,
            contents=json.loads(run_log.model_dump_json()),
            log_type=self.LogTypes.RUN_LOG,
        )
        return run_log

    def get_run_log_by_id(self, run_id: str, full: bool = False) -> RunLog:
        """
        Retrieves a Run log from the database using the config and the run_id

        Args:
            run_id (str): The run_id of the run
            full (bool): return the full run log store or only the RunLog object

        Returns:
            RunLog: The RunLog object identified by the run_id

        Logically the method should:
            * Returns the run_log defined by id from the data store defined by the config

        """
        try:
            logger.info(f"{self.service_name} Getting a Run Log for : {run_id}")
            run_log = self.retrieve(run_id=run_id, log_type=self.LogTypes.RUN_LOG)

            if full:
                self._prepare_full_run_log(run_log=run_log)

            return run_log
        except exceptions.EntityNotFoundError as e:
            raise exceptions.RunLogNotFoundError(run_id) from e

    def put_run_log(self, run_log: RunLog):
        """
        Puts the Run Log in the database as defined by the config

        Args:
            run_log (RunLog): The Run log of the run

        Logically the method should:
            Puts the run_log into the database

        Raises:
            NotImplementedError: This is a base class and therefore has no default implementation
        """
        run_id = run_log.run_id
        self.store(
            run_id=run_id,
            contents=json.loads(run_log.model_dump_json()),
            log_type=self.LogTypes.RUN_LOG,
        )

    def get_parameters(self, run_id: str, internal_branch_name: str = "") -> dict:
        """
        Get parameters from RunLog or BranchLog.

        Args:
            run_id (str): The run_id of the run
            internal_branch_name (str): If provided, get from that branch

        Returns:
            dict: Parameters from the specified scope
        """
        if internal_branch_name:
            branch = self.retrieve(
                run_id=run_id,
                log_type=self.LogTypes.BRANCH_LOG,
                name=internal_branch_name,
            )
            return branch.parameters

        run_log = self.get_run_log_by_id(run_id=run_id)
        return run_log.parameters

    def set_parameters(
        self, run_id: str, parameters: dict, internal_branch_name: str = ""
    ):
        """
        Set parameters on RunLog or BranchLog.

        Args:
            run_id (str): The run_id of the run
            parameters (dict): Parameters to set
            internal_branch_name (str): If provided, set on that branch
        """
        if internal_branch_name:
            branch = self.retrieve(
                run_id=run_id,
                log_type=self.LogTypes.BRANCH_LOG,
                name=internal_branch_name,
            )
            branch.parameters.update(parameters)
            self.store(
                run_id=run_id,
                log_type=self.LogTypes.BRANCH_LOG,
                contents=json.loads(branch.model_dump_json()),
                name=internal_branch_name,
            )
        else:
            run_log = self.get_run_log_by_id(run_id=run_id)
            run_log.parameters.update(parameters)
            self.put_run_log(run_log)

    def get_run_config(self, run_id: str) -> dict:
        """
        Given a run_id, return the run_config used to perform the run.

        Args:
            run_id (str): The run_id of the run

        Returns:
            dict: The run config used for the run
        """

        run_log = self.get_run_log_by_id(run_id=run_id)
        return run_log.run_config

    def set_run_config(self, run_id: str, run_config: dict):
        """Set the run config used to run the run_id

        Args:
            run_id (str): The run_id of the run
            run_config (dict): The run_config of the run
        """

        run_log = self.get_run_log_by_id(run_id=run_id)
        run_log.run_config.update(run_config)
        self.put_run_log(run_log=run_log)

    def get_step_log(self, internal_name: str, run_id: str) -> StepLog:
        """
        Get a step log from the datastore for run_id and the internal naming of the step log

        The internal naming of the step log is a dot path convention.

        The method should:
            * Call get_run_log_by_id(run_id) to retrieve the run_log
            * Identify the step location by decoding the internal naming
            * Return the step log

        Args:
            internal_name (str): The internal name of the step log
            run_id (str): The run_id of the run

        Returns:
            StepLog: The step log object for the step defined by the internal naming and run_id

        Raises:
            RunLogNotFoundError: If the run log for run_id is not found in the datastore
            StepLogNotFoundError: If the step log for internal_name is not found in the datastore for run_id
        """
        logger.info(
            f"{self.service_name} Getting the step log: {internal_name} of {run_id}"
        )

        # Determine if step is in a branch or root
        parent_branch = self._get_parent_branch(internal_name)

        if not parent_branch:
            # Root-level step - get from RunLog
            run_log = self.get_run_log_by_id(run_id=run_id)
            if internal_name not in run_log.steps:
                raise exceptions.StepLogNotFoundError(
                    run_id=run_id, step_name=internal_name
                )
            return run_log.steps[internal_name]
        else:
            # Branch step - get from BranchLog
            try:
                branch_log = self.retrieve(
                    run_id=run_id,
                    log_type=self.LogTypes.BRANCH_LOG,
                    name=parent_branch,
                )
                if internal_name not in branch_log.steps:
                    raise exceptions.StepLogNotFoundError(
                        run_id=run_id, step_name=internal_name
                    )
                return branch_log.steps[internal_name]
            except exceptions.EntityNotFoundError as e:
                raise exceptions.StepLogNotFoundError(
                    run_id=run_id, step_name=internal_name
                ) from e

    def add_step_log(self, step_log: StepLog, run_id: str):
        """
        Add the step log to its parent (RunLog or BranchLog).

        Args:
            step_log (StepLog): The Step log to add
            run_id (str): The run id of the run
        """
        logger.info(f"{self.service_name} Adding step log: {step_log.internal_name}")

        internal_name = step_log.internal_name
        parent_branch = self._get_parent_branch(internal_name)

        if not parent_branch:
            # Root-level step - add to RunLog
            run_log = self.get_run_log_by_id(run_id=run_id)
            run_log.steps[internal_name] = step_log
            self.put_run_log(run_log)
        else:
            # Branch step - add to BranchLog
            branch_log = self.retrieve(
                run_id=run_id,
                log_type=self.LogTypes.BRANCH_LOG,
                name=parent_branch,
            )
            branch_log.steps[internal_name] = step_log
            self.store(
                run_id=run_id,
                log_type=self.LogTypes.BRANCH_LOG,
                contents=json.loads(branch_log.model_dump_json()),
                name=parent_branch,
            )

    def get_branch_log(
        self, internal_branch_name: str, run_id: str
    ) -> Union[BranchLog, RunLog]:
        """
        Returns the branch log by the internal branch name for the run id

        If the internal branch name is none, returns the run log

        Args:
            internal_branch_name (str): The internal branch name to retrieve.
            run_id (str): The run id of interest

        Returns:
            BranchLog: The branch log or the run log as requested.
        """
        try:
            if not internal_branch_name:
                return self.get_run_log_by_id(run_id=run_id)
            branch = self.retrieve(
                run_id=run_id,
                log_type=self.LogTypes.BRANCH_LOG,
                name=internal_branch_name,
            )
            return branch
        except exceptions.EntityNotFoundError as e:
            raise exceptions.BranchLogNotFoundError(
                run_id=run_id, branch_name=internal_branch_name
            ) from e

    def add_branch_log(
        self,
        branch_log: Union[BranchLog, RunLog],
        run_id: str,
    ):
        """
        The method should:
        # Get the run log
        # Get the branch and step containing the branch
        # Add the branch to the step
        # Write the run_log

        The branch log could some times be a Run log and should be handled appropriately

        Args:
            branch_log (BranchLog): The branch log/run log to add to the database
            run_id (str): The run id to which the branch/run log is added
        """
        if not isinstance(branch_log, BranchLog):
            self.put_run_log(branch_log)
            return

        internal_branch_name = branch_log.internal_name

        logger.info(
            f"{self.service_name} Adding the branch log to DB: {branch_log.internal_name}"
        )
        self.store(
            run_id=run_id,
            log_type=self.LogTypes.BRANCH_LOG,
            contents=json.loads(branch_log.model_dump_json()),
            name=internal_branch_name,
        )

import json
import logging
import time
from enum import Enum
from pathlib import Path
from string import Template
from typing import Any, List, Optional, Union

from magnus import defaults, exceptions, utils
from magnus.datastore import BaseRunLogStore, BranchLog, RunLog, StepLog

logger = logging.getLogger(defaults.LOGGER_NAME)


class ChunkedFileSystemRunLogStore(BaseRunLogStore):
    """
    File system run log store but chunks the run log into thread safe chunks.
    This enables executions to be parallel.
    """

    service_name: str = "chunked-fs"
    log_folder: str = defaults.LOG_LOCATION_FOLDER

    @property
    def log_folder_name(self):
        return self.log_folder

    class LogTypes(Enum):
        RUN_LOG: str = "RunLog"
        PARAMETER: str = "Parameter"
        STEP_LOG: str = "StepLog"
        BRANCH_LOG: str = "BranchLog"

    class ModelTypes(Enum):
        RUN_LOG = RunLog
        PARAMETER = dict
        STEP_LOG = StepLog
        BRANCH_LOG = BranchLog

    def naming_pattern(self, log_type: LogTypes, name: str = "") -> str:
        """
        Naming pattern to store RunLog, Parameter, StepLog or BranchLog.

        The reasoning for name to be defaulted to empty string:
            Its actually conditionally empty. For RunLog and Parameter it is empty.
            For StepLog and BranchLog it should be provided.

        Args:
            log_type (LogTypes): One of RunLog, Parameter, StepLog or BranchLog
            name (str, optional): The name to be included or left. Defaults to ''.

        Raises:
            Exception: If log_type is not recognized

        Returns:
            str: The naming pattern
        """
        if log_type == self.LogTypes.RUN_LOG:
            return f"{self.LogTypes.RUN_LOG.value}"

        if log_type == self.LogTypes.PARAMETER:
            return "-".join([self.LogTypes.PARAMETER.value, name])

        if not name:
            raise Exception(f"Name should be provided for naming pattern for {log_type}")

        if log_type == self.LogTypes.STEP_LOG:
            return "-".join([self.LogTypes.STEP_LOG.value, name, "${creation_time}"])

        if log_type == self.LogTypes.BRANCH_LOG:
            return "-".join([self.LogTypes.BRANCH_LOG.value, name, "${creation_time}"])

        raise Exception("Unexpected log type sent")

    def get_matches(self, run_id: str, name: str, multiple_allowed: bool = False) -> Optional[Union[List[Path], Path]]:
        """
        Get contents of files matching the pattern name*

        Args:
            run_id (str): The run id
            name (str): The suffix of the file name to check in the run log store.
        """
        log_folder = self.log_folder_with_run_id(run_id=run_id)

        sub_name = Template(name).safe_substitute({"creation_time": ""})

        matches = list(log_folder.glob(f"{sub_name}*"))
        if matches:
            if not multiple_allowed:
                if len(matches) > 1:
                    msg = f"Multiple matches found for {name} while multiple is not allowed"
                    raise Exception(msg)
                return matches[0]
            return matches

        return None

    def log_folder_with_run_id(self, run_id: str) -> Path:
        """
        Utility function to get the log folder for a run id.

        Args:
            run_id (str): The run id

        Returns:
            Path: The path to the log folder with the run id
        """
        return Path(self.log_folder_name) / run_id

    def safe_suffix_json(self, name: Path) -> str:
        """
        Safely attach a suffix to a json file.

        Args:
            name (Path): The name of the file with or without suffix of json

        Returns:
            str : The name of the file with .json
        """
        if str(name).endswith("json"):
            return str(name)

        return str(name) + ".json"

    def _store(self, run_id: str, contents: dict, name: Path):
        """
        Store the contents against the name in the folder.

        Args:
            run_id (str): The run id
            contents (dict): The dict to store
            name (str): The name to store as
        """
        utils.safe_make_dir(self.log_folder_with_run_id(run_id=run_id))

        with open(self.safe_suffix_json(name), "w") as fw:
            json.dump(contents, fw, ensure_ascii=True, indent=4)

    def _retrieve(self, name: Path) -> dict:
        """
        Does the job of retrieving from the folder.

        Args:
            name (str): the name of the file to retrieve

        Returns:
            dict: The contents
        """
        contents: dict = {}

        with open(self.safe_suffix_json(name), "r") as fr:
            contents = json.load(fr)

        return contents

    def store(self, run_id: str, log_type: LogTypes, contents: dict, name: str = ""):
        """Store a SINGLE log type in the file system

        Args:
            run_id (str): The run id to store against
            log_type (LogTypes): The type of log to store
            contents (dict): The dict of contents to store
            name (str, optional): The name against the contents have to be stored. Defaults to ''.
        """
        naming_pattern = self.naming_pattern(log_type=log_type, name=name)
        match = self.get_matches(run_id=run_id, name=naming_pattern, multiple_allowed=False)
        # The boolean multiple allowed confuses mypy a lot!
        name_to_give: Path = None  # type: ignore
        if match:
            existing_contents = self._retrieve(name=match)  # type: ignore
            contents = dict(existing_contents, **contents)
            name_to_give = match  # type: ignore
        else:
            _name = Template(naming_pattern).safe_substitute({"creation_time": str(int(time.time_ns()))})
            name_to_give = self.log_folder_with_run_id(run_id=run_id) / _name

        self._store(run_id=run_id, contents=contents, name=name_to_give)

    def retrieve(self, run_id: str, log_type: LogTypes, name: str = "", multiple_allowed=False) -> Any:
        """
        Retrieve the model given a log_type and a name.
        Use multiple_allowed to control if you are expecting multiple of them.
        eg: There could be multiple of Parameters- but only one of StepLog-stepname

        The reasoning for name to be defaulted to empty string:
            Its actually conditionally empty. For RunLog and Parameter it is empty.
            For StepLog and BranchLog it should be provided.

        Args:
            run_id (str): The run id
            log_type (LogTypes): One of RunLog, Parameter, StepLog, BranchLog
            name (str, optional): The name to match. Defaults to ''.
            multiple_allowed (bool, optional): Are multiple allowed. Defaults to False.

        Raises:
            FileNotFoundError: If there is no match found

        Returns:
            Any: One of StepLog, BranchLog, Parameter or RunLog
        """
        # The reason of any is it could be one of Logs or dict or list of the
        if not name and log_type not in [
            self.LogTypes.RUN_LOG,
            self.LogTypes.PARAMETER,
        ]:
            raise Exception(f"Name is required during retrieval for {log_type}")

        naming_pattern = self.naming_pattern(log_type=log_type, name=name)
        matches = self.get_matches(run_id=run_id, name=naming_pattern, multiple_allowed=multiple_allowed)
        if matches:
            if not multiple_allowed:
                contents = self._retrieve(name=matches)  # type: ignore
                model = self.ModelTypes[log_type.name].value
                return model(**contents)

            models = []
            for match in matches:  # type: ignore
                contents = self._retrieve(name=match)
                model = self.ModelTypes[log_type.name].value
                models.append(model(**contents))
            return models

        raise FileNotFoundError()

    def orderly_retrieve(self, run_id: str, log_type: LogTypes) -> dict[str, Union[StepLog, BranchLog]]:
        """Should only be used by prepare full run log.

        Retrieves the StepLog or BranchLog sorted according to creation time.

        Args:
            run_id (str): _description_
            log_type (LogTypes): _description_
        """
        prefix: str = self.LogTypes.STEP_LOG.value

        if log_type == self.LogTypes.BRANCH_LOG:
            prefix = self.LogTypes.BRANCH_LOG.value

        matches = self.get_matches(run_id=run_id, name=prefix, multiple_allowed=True)

        if log_type == self.LogTypes.BRANCH_LOG and not matches:
            # No branch logs are found
            return {}
        # Forcing get_matches to always return a list is a better design
        epoch_created = [str(match).split("-")[-1] for match in matches]  # type: ignore

        # sort matches by epoch created
        epoch_created, matches = zip(*sorted(zip(epoch_created, matches)))  # type: ignore

        logs: dict[str, Union[StepLog, BranchLog]] = {}

        for match in matches:  # type: ignore
            model = self.ModelTypes[log_type.name].value
            log_model = model(**self._retrieve(match))
            logs[log_model.internal_name] = log_model  # type: ignore

        return logs

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
        Populates the run log with the branches and steps.

        Args:
            run_log (RunLog): The partial run log containing empty step logs
        """
        run_id = run_log.run_id
        run_log.parameters = self.get_parameters(run_id=run_id)

        ordered_steps = self.orderly_retrieve(run_id=run_id, log_type=self.LogTypes.STEP_LOG)
        ordered_branches = self.orderly_retrieve(run_id=run_id, log_type=self.LogTypes.BRANCH_LOG)

        current_branch: Any = None  # It could be str, None, RunLog
        for step_internal_name in ordered_steps:
            current_branch = self._get_parent_branch(step_internal_name)
            step_to_add_branch = self._get_parent_step(step_internal_name)

            if not current_branch:
                current_branch = run_log
            else:
                current_branch = ordered_branches[current_branch]
                step_to_add_branch = ordered_steps[step_to_add_branch]  # type: ignore
                step_to_add_branch.branches[current_branch.internal_name] = current_branch  # type: ignore

            current_branch.steps[step_internal_name] = ordered_steps[step_internal_name]

    def create_run_log(
        self,
        run_id: str,
        dag_hash: str = "",
        use_cached: bool = False,
        tag: str = "",
        original_run_id: str = "",
        status: str = defaults.CREATED,
        **kwargs,
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
            use_cached=use_cached,
            tag=tag,
            original_run_id=original_run_id,
            status=status,
        )

        self.store(run_id=run_id, contents=run_log.dict(), log_type=self.LogTypes.RUN_LOG)
        return run_log

    def get_run_log_by_id(self, run_id: str, full: bool = False, **kwargs) -> RunLog:
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
            run_log = self.retrieve(run_id=run_id, log_type=self.LogTypes.RUN_LOG, multiple_allowed=False)

            if full:
                self._prepare_full_run_log(run_log=run_log)

            return run_log
        except FileNotFoundError as e:
            raise exceptions.RunLogNotFoundError(run_id) from e

    def put_run_log(self, run_log: RunLog, **kwargs):
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
        self.store(run_id=run_id, contents=run_log.dict(), log_type=self.LogTypes.RUN_LOG)

    def get_parameters(self, run_id: str, **kwargs) -> dict:
        """
        Get the parameters from the Run log defined by the run_id

        Args:
            run_id (str): The run_id of the run

        The method should:
            * Call get_run_log_by_id(run_id) to retrieve the run_log
            * Return the parameters as identified in the run_log

        Returns:
            dict: A dictionary of the run_log parameters
        Raises:
            RunLogNotFoundError: If the run log for run_id is not found in the datastore
        """
        parameters = {}
        try:
            parameters_list = self.retrieve(run_id=run_id, log_type=self.LogTypes.PARAMETER, multiple_allowed=True)
            parameters = {key: value for param in parameters_list for key, value in param.items()}
        except FileNotFoundError:
            # No parameters are set
            pass

        return parameters

    def set_parameters(self, run_id: str, parameters: dict, **kwargs):
        """
        Update the parameters of the Run log with the new parameters

        This method would over-write the parameters, if the parameter exists in the run log already

        The method should:
            * Call get_run_log_by_id(run_id) to retrieve the run_log
            * Update the parameters of the run_log
            * Call put_run_log(run_log) to put the run_log in the datastore

        Args:
            run_id (str): The run_id of the run
            parameters (dict): The parameters to update in the run log
        Raises:
            RunLogNotFoundError: If the run log for run_id is not found in the datastore
        """
        for key, value in parameters.items():
            self.store(
                run_id=run_id,
                log_type=self.LogTypes.PARAMETER,
                contents={key: value},
                name=key,
            )

    def get_run_config(self, run_id: str, **kwargs) -> dict:
        """
        Given a run_id, return the run_config used to perform the run.

        Args:
            run_id (str): The run_id of the run

        Returns:
            dict: The run config used for the run
        """

        run_log = self.get_run_log_by_id(run_id=run_id)
        return run_log.run_config

    def set_run_config(self, run_id: str, run_config: dict, **kwargs):
        """Set the run config used to run the run_id

        Args:
            run_id (str): The run_id of the run
            run_config (dict): The run_config of the run
        """

        run_log = self.get_run_log_by_id(run_id=run_id)
        run_log.run_config.update(run_config)
        self.put_run_log(run_log=run_log)

    def get_step_log(self, internal_name: str, run_id: str, **kwargs) -> StepLog:
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
        logger.info(f"{self.service_name} Getting the step log: {internal_name} of {run_id}")

        step_log = self.retrieve(
            run_id=run_id,
            log_type=self.LogTypes.STEP_LOG,
            name=internal_name,
            multiple_allowed=False,
        )

        return step_log

    def add_step_log(self, step_log: StepLog, run_id: str, **kwargs):
        """
        Add the step log in the run log as identified by the run_id in the datastore

        The method should:
             * Call get_run_log_by_id(run_id) to retrieve the run_log
             * Identify the branch to add the step by decoding the step_logs internal name
             * Add the step log to the identified branch log
             * Call put_run_log(run_log) to put the run_log in the datastore

        Args:
            step_log (StepLog): The Step log to add to the database
            run_id (str): The run id of the run

        Raises:
            RunLogNotFoundError: If the run log for run_id is not found in the datastore
            BranchLogNotFoundError: If the branch of the step log for internal_name is not found in the datastore
                                    for run_id
        """
        logger.info(f"{self.service_name} Adding the step log to DB: {step_log.internal_name}")

        self.store(
            run_id=run_id,
            log_type=self.LogTypes.STEP_LOG,
            contents=step_log.dict(),
            name=step_log.internal_name,
        )

    def get_branch_log(self, internal_branch_name: str, run_id: str, **kwargs) -> Union[BranchLog, RunLog]:
        """
        Returns the branch log by the internal branch name for the run id

        If the internal branch name is none, returns the run log

        Args:
            internal_branch_name (str): The internal branch name to retrieve.
            run_id (str): The run id of interest

        Returns:
            BranchLog: The branch log or the run log as requested.
        """
        if not internal_branch_name:
            return self.get_run_log_by_id(run_id=run_id)
        branch = self.retrieve(run_id=run_id, log_type=self.LogTypes.BRANCH_LOG, name=internal_branch_name)
        return branch

    def add_branch_log(self, branch_log: Union[BranchLog, RunLog], run_id: str, **kwargs):
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

        logger.info(f"{self.service_name} Adding the branch log to DB: {branch_log.internal_name}")
        self.store(
            run_id=run_id,
            log_type=self.LogTypes.BRANCH_LOG,
            contents=branch_log.dict(),
            name=internal_branch_name,
        )

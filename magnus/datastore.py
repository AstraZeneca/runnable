from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

from pydantic import BaseModel, Extra

from magnus import defaults, exceptions, utils

logger = logging.getLogger(defaults.NAME)

# Once defined these classes are sealed to any additions unless a default is provided
# Breaking this rule might make magnus backwardly incompatible


class DataCatalog(BaseModel, extra=Extra.allow):  # type: ignore
    """
    The captured attributes of a catalog item.
    """
    name: str  #  The name of the dataset
    data_hash: str = ''  # The sha1 hash of the file
    catalog_relative_path: str = ''  # The file path relative the catalog location
    catalog_handler_location: str = ''  # The location of the catalog
    stage: str = ''  # The stage at which we recorded it get, put etc

    # Needed for set operations to work on DataCatalog objects
    def __hash__(self):
        return hash(self.name)

    # Needed for set operations to work on DataCatalog objects
    def __eq__(self, other):
        if not isinstance(other, DataCatalog):
            return False
        return other.name == self.name


class StepAttempt(BaseModel):
    """
    The captured attributes of an Attempt of a step.
    """
    attempt_number: int = 0
    start_time: str = ''
    end_time: str = ''
    duration: str = ''  #  end_time - start_time
    status: str = 'FAIL'
    message: str = ''
    parameters: dict = {}


class CodeIdentity(BaseModel, extra=Extra.allow):  # type: ignore
    """
    The captured attributes of a code identity of a step.
    """
    code_identifier: Optional[str] = ''  # GIT sha code or docker image id
    code_identifier_type: Optional[str] = ''  # git or docker
    code_identifier_dependable: Optional[bool] = False  # If git, checks if the tree is clean.
    code_identifier_url: Optional[str] = ''  # The git remote url or docker repository url
    code_identifier_message: Optional[str] = ''  # Any optional message


class StepLog(BaseModel):
    """
    The data class capturing the data of a Step
    """
    name: str
    internal_name: str  # Should be the dot notation of the step
    status: str = 'FAIL'  #  Should have a better default
    step_type: str = 'task'
    message: str = ''
    mock: bool = False
    code_identities: List[CodeIdentity] = []
    attempts: List[StepAttempt] = []
    user_defined_metrics: dict = {}
    branches: Dict[str, BranchLog] = {}  # Keyed in by the branch key name
    data_catalog: List[DataCatalog] = []

    def get_data_catalogs_by_stage(self, stage='put') -> List[DataCatalog]:
        """
        Given a stage, return the data catalogs according to the stage

        Args:
            stage (str, optional): The stage at which the data was cataloged. Defaults to 'put'.

        Raises:
            Exception: If the stage was not in get or put.

        Returns:
            List[DataCatalog]: The list of data catalogs as per the stage.
        """
        if stage not in ['get', 'put']:
            raise Exception('Stage should be in get or put')

        data_catalogs = []
        if self.branches:
            for _, branch in self.branches.items():
                data_catalogs.extend(branch.get_data_catalogs_by_stage(stage=stage))

        return [dc for dc in self.data_catalog if dc.stage == stage] + data_catalogs

    def add_data_catalogs(self, data_catalogs: List[DataCatalog]):
        """
        Add the data catalogs as asked by the user

        Args:
            dict_catalogs ([DataCatalog]): A list of data catalog items
        """
        if not self.data_catalog:
            self.data_catalog = []
        for data_catalog in data_catalogs:
            self.data_catalog.append(data_catalog)


class BranchLog(BaseModel):
    """
    The dataclass of captured data about a branch of a composite node.

    Returns:
        [type]: [description]
    """
    internal_name: str
    status: str = 'FAIL'
    steps: OrderedDict[str, StepLog] = {}  # type: ignore # StepLogs keyed by internal name

    def get_data_catalogs_by_stage(self, stage='put') -> List[DataCatalog]:
        """
        Given a stage, return the data catalogs according to the stage

        Args:
            stage (str, optional): The stage at which the data was cataloged. Defaults to 'put'.

        Raises:
            Exception: If the stage was not in get or put.

        Returns:
            List[DataCatalog]: The list of data catalogs as per the stage.
        """
        if stage not in ['get', 'put']:
            raise Exception('Stage should be in get or put')

        data_catalogs = []
        for _, step in self.steps.items():
            data_catalogs.extend(step.get_data_catalogs_by_stage(stage=stage))

        return data_catalogs


# Needed for BranchLog of StepLog to be referenced
StepLog.update_forward_refs()


class RunLog(BaseModel):
    """
    The data captured as part of Run Log
    """
    run_id: str
    dag_hash: Optional[str] = None
    use_cached: bool = False
    tag: Optional[str] = ''
    original_run_id: Optional[str] = ''
    status: str = defaults.FAIL
    steps: OrderedDict[str, StepLog] = {}  # type: ignore # Has the steps keyed by internal_name
    parameters: dict = {}
    run_config: dict = {}

    def get_data_catalogs_by_stage(self, stage: str = 'put') -> List[DataCatalog]:
        """
        Return all the cataloged data by the stage at which they were cataloged.

        Raises:
            Exception: If stage was not either put or get.

        Args:
            stage (str, optional): [description]. Defaults to 'put'.
        """
        if stage not in ['get', 'put']:
            raise Exception('Only get or put are allowed in stage')

        data_catalogs = []
        for _, step in self.steps.items():
            data_catalogs.extend(step.get_data_catalogs_by_stage(stage=stage))

        return list(set(data_catalogs))

    def search_branch_by_internal_name(self, i_name: str) -> Tuple[Union[BranchLog, RunLog], Union[StepLog, None]]:
        """
        Given a branch internal name, search for it in the run log.

        If the branch internal name is none, its the run log itself.

        Args:
            i_name (str): [description]

        Raises:
            exceptions.BranchLogNotFoundError: [description]

        Returns:
            Tuple[BranchLog, StepLog]: [description]
        """
        # internal name is null for base dag
        if not i_name:
            return self, None

        dot_path = i_name.split('.')

        # any internal name of a branch when split against .
        # goes step.branch.step.branch
        # If its odd, its a step, if its even its a branch
        current_steps = self.steps
        current_step = None
        current_branch = None

        for i in range(len(dot_path)):
            if i % 2:
                # Its odd, so we are in branch
                # Get the branch that holds the step
                current_branch = current_step.branches['.'.join(dot_path[:i + 1])]  # type: ignore
                current_steps = current_branch.steps
                logger.debug(f'Finding branch {i_name} in branch: {current_branch}')
            else:
                # Its even, so we are in step, we start here!
                # Get the step that holds the branch
                current_step = current_steps['.'.join(dot_path[:i + 1])]
                logger.debug(f'Finding branch for {i_name} in step: {current_step}')

        logger.debug(f'current branch : {current_branch}, current step {current_step}')
        if current_branch and current_step:
            return current_branch, current_step

        raise exceptions.BranchLogNotFoundError(self.run_id, i_name)

    def search_step_by_internal_name(self, i_name: str) -> Tuple[StepLog, Union[BranchLog, None]]:
        """
        Given a steps internal name, search for the step name.

        If the step name when split against '.' is 1, it is the run log

        Args:
            i_name (str): [description]

        Raises:
            exceptions.StepLogNotFoundError: [description]

        Returns:
            Tuple[StepLog, BranchLog]: [description]
        """
        dot_path = i_name.split('.')
        if len(dot_path) == 1:
            return self.steps[i_name], None

        current_steps = self.steps
        current_step = None
        current_branch = None
        for i in range(len(dot_path)):
            if i % 2:
                # Its odd, so we are in brach name
                current_branch = current_step.branches['.'.join(dot_path[:i + 1])]  # type: ignore
                current_steps = current_branch.steps
                logger.debug(f'Finding step log for {i_name} in branch: {current_branch}')
            else:
                # Its even, so we are in step, we start here!
                current_step = current_steps['.'.join(dot_path[:i + 1])]
                logger.debug(f'Finding step log for {i_name} in step: {current_step}')

        logger.debug(f'current branch : {current_branch}, current step {current_step}')
        if current_branch and current_step:
            return current_step, current_branch

        raise exceptions.StepLogNotFoundError(self.run_id, i_name)


# All outside modules should interact with dataclasses using the RunLogStore to promote extensibility
# If you want to customize dataclass, extend BaseRunLogStore and implement the methods as per the specification

class BaseRunLogStore:
    """
    The base class of a Run Log Store with many common methods implemented.

    Note: As a general guideline, do not extract anything from the config to set class level attributes.
          Integration patterns modify the config after init to change behaviors.
          Access config properties using getters/property of the class.
    """
    service_name = ''

    class Config(BaseModel):
        pass

    def __init__(self, config):
        config = config or {}
        self.config = self.Config(**config)

    def create_run_log(self, run_id: str, dag_hash: str = '', use_cached: bool = False,
                       tag: str = '', original_run_id: str = '', status: str = defaults.CREATED, **kwargs):
        """
        Creates a Run Log object by using the config

        Logically the method should do the following:
            * Creates a Run log
            * Adds it to the db
            * Return the log
        Raises:
            NotImplementedError: This is a base class and therefore has no default implementation
        """

        raise NotImplementedError

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

        Raises:
            NotImplementedError: This is a base class and therefore has no default implementation
            RunLogNotFoundError: If the run log for run_id is not found in the datastore
        """

        raise NotImplementedError

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
        raise NotImplementedError

    def get_parameters(self, run_id: str, **kwargs) -> dict:  # pylint: disable=unused-argument
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
        run_log = self.get_run_log_by_id(run_id=run_id)
        return run_log.parameters

    def set_parameters(self, run_id: str, parameters: dict, **kwargs):  # pylint: disable=unused-argument
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
        run_log = self.get_run_log_by_id(run_id=run_id)
        run_log.parameters.update(parameters)
        self.put_run_log(run_log=run_log)

    def get_run_config(self, run_id: str, **kwargs) -> dict:  # pylint: disable=unused-argument
        """
        Given a run_id, return the run_config used to perform the run.

        Args:
            run_id (str): The run_id of the run

        Returns:
            dict: The run config used for the run
        """

        run_log = self.get_run_log_by_id(run_id=run_id)
        return run_log.run_config

    def set_run_config(self, run_id: str, run_config: dict, **kwargs):  # pylint: disable=unused-argument
        """ Set the run config used to run the run_id

        Args:
            run_id (str): The run_id of the run
            run_config (dict): The run_config of the run
        """

        run_log = self.get_run_log_by_id(run_id=run_id)
        run_log.run_config.update(run_config)
        self.put_run_log(run_log=run_log)

    def create_step_log(self, name: str, internal_name: str, **kwargs):  # pylint: disable=unused-argument
        """
        Create a step log by the name and internal name

        The method does not update the Run Log with the step log at this point in time.
        This method is just an interface for external modules to create a step log


        Args:
            name (str): The friendly name of the step log
            internal_name (str): The internal naming of the step log. The internal naming is a dot path convention

        Returns:
            StepLog: A uncommitted step log object
        """
        logger.info(f'{self.service_name} Creating a Step Log: {internal_name}')
        return StepLog(name=name, internal_name=internal_name, status=defaults.CREATED)

    def get_step_log(self, internal_name: str, run_id: str, **kwargs) -> StepLog:  # pylint: disable=unused-argument
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
        logger.info(f'{self.service_name} Getting the step log: {internal_name} of {run_id}')
        run_log = self.get_run_log_by_id(run_id=run_id)
        step_log, _ = run_log.search_step_by_internal_name(internal_name)
        return step_log

    def add_step_log(self, step_log: StepLog, run_id: str, **kwargs):  # pylint: disable=unused-argument
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
        logger.info(f'{self.service_name} Adding the step log to DB: {step_log.name}')
        run_log = self.get_run_log_by_id(run_id=run_id)

        branch_to_add = '.'.join(step_log.internal_name.split('.')[:-1])
        branch, _ = run_log.search_branch_by_internal_name(branch_to_add)

        if branch is None:
            branch = run_log
        branch.steps[step_log.internal_name] = step_log
        self.put_run_log(run_log=run_log)

    def create_branch_log(self, internal_branch_name: str, **kwargs) -> BranchLog:  # pylint: disable=unused-argument
        """
        Creates a uncommitted branch log object by the internal name given

        Args:
            internal_branch_name (str): Creates a branch log by name internal_branch_name

        Returns:
            BranchLog: Uncommitted and initialized with defaults BranchLog object
        """
        # Create a new BranchLog
        logger.info(f'{self.service_name} Creating a Branch Log : {internal_branch_name}')
        return BranchLog(internal_name=internal_branch_name, status=defaults.CREATED)

    def get_branch_log(self, internal_branch_name: str, run_id: str, **kwargs) -> Union[BranchLog, RunLog]:  # pylint: disable=unused-argument
        """
        Returns the branch log by the internal branch name for the run id

        If the internal branch name is none, returns the run log

        Args:
            internal_branch_name (str): The internal branch name to retrieve.
            run_id (str): The run id of interest

        Returns:
            BranchLog: The branch log or the run log as requested.
        """
        run_log = self.get_run_log_by_id(run_id=run_id)
        if not internal_branch_name:
            return run_log
        branch, _ = run_log.search_branch_by_internal_name(internal_branch_name)
        return branch

    def add_branch_log(self, branch_log: Union[BranchLog, RunLog], run_id: str, **kwargs):  # pylint: disable=unused-argument
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

        internal_branch_name = None

        if isinstance(branch_log, BranchLog):
            internal_branch_name = branch_log.internal_name

        if not internal_branch_name:
            self.put_run_log(branch_log)  # type: ignore # We are dealing with base dag here
            return

        run_log = self.get_run_log_by_id(run_id=run_id)

        step_name = '.'.join(internal_branch_name.split('.')[:-1])
        step, _ = run_log.search_step_by_internal_name(step_name)

        step.branches[internal_branch_name] = branch_log  # type: ignore
        self.put_run_log(run_log)

    def create_attempt_log(self, **kwargs) -> StepAttempt:  # pylint: disable=unused-argument
        """
        Returns an uncommitted step attempt log.

        Returns:
            StepAttempt: An uncommitted step attempt log
        """
        logger.info(f'{self.service_name} Creating an attempt log')
        return StepAttempt()

    def create_code_identity(self, **kwargs) -> CodeIdentity:  # pylint: disable=unused-argument
        """
        Creates an uncommitted Code identity class

        Returns:
            CodeIdentity: An uncommitted code identity class
        """
        logger.info(f'{self.service_name} Creating Code identity')
        return CodeIdentity()

    def create_data_catalog(self, name: str, **kwargs) -> DataCatalog:  # pylint: disable=unused-argument
        """
        Create a uncommitted data catalog object

        Args:
            name (str): The name of the data catalog item to put

        Returns:
            DataCatalog: The DataCatalog object.
        """
        logger.info(f'{self.service_name} Creating Data Catalog for {name}')
        return DataCatalog(name=name)


class BufferRunLogstore(BaseRunLogStore):
    """
    A in-memory run log store.

    This Run Log store will not persist any results.

    When to use:
     When testing some part of the pipeline.

    Do not use:
     When you need to compare between runs or in production set up

    This Run Log Store is concurrent write safe as it is in memory

    Example config:
    run_log:
      type: buffered

    """
    service_name = 'buffered'

    def __init__(self, config):
        super().__init__(config)
        self.run_log = None  # For a buffered Run Log, this is the database

    def create_run_log(self, run_id: str, dag_hash: str = '', use_cached: bool = False,
                       tag: str = '', original_run_id: str = '', status: str = defaults.CREATED, **kwargs) -> RunLog:
        # Creates a Run log
        # Adds it to the db
        # Return the log
        logger.info(f'{self.service_name} Creating a Run Log and adding it to DB')
        self.run_log = RunLog(run_id=run_id, dag_hash=dag_hash, use_cached=use_cached,
                              tag=tag, original_run_id=original_run_id, status=status)
        return self.run_log

    def get_run_log_by_id(self, run_id: str, full: bool = False, **kwargs):
        # Returns the run_log defined by id
        # Raises Exception if not found
        logger.info(f'{self.service_name} Getting the run log from DB for {run_id}')
        if self.run_log:
            return self.run_log

        raise exceptions.RunLogNotFoundError(run_id)

    def put_run_log(self, run_log: RunLog, **kwargs):
        # Puts the run_log into the database
        logger.info(f'{self.service_name} Putting the run log in the DB: {run_log.run_id}')
        self.run_log = run_log


class FileSystemRunLogstore(BaseRunLogStore):
    """
    In this type of Run Log store, we use a file system to store the JSON run log.

    Every single run is stored as a different file which makes it compatible across other store types.

    When to use:
        When locally testing a pipeline and have the need to compare across runs.
        Its fully featured and perfectly fine if your local environment is where you would do everyhing.

    Do not use:
        If you need parallelization on local, this run log would not support it.

    Example config:

    run_log:
      type: file-system
      config:
        log_folder: The folder to out the logs. Defaults to .run_log_store
    """
    service_name = 'file-system'

    class Config(BaseModel):
        log_folder: str = defaults.LOG_LOCATION_FOLDER

    @property
    def log_folder_name(self) -> str:
        return self.config.log_folder

    def write_to_folder(self, run_log: RunLog):
        """
        Write the run log to the folder

        Args:
            run_log (RunLog): The run log to be added to the database
        """
        write_to = self.log_folder_name
        utils.safe_make_dir(write_to)

        write_to_path = Path(write_to)
        run_id = run_log.run_id
        json_file_path = write_to_path / f'{run_id}.json'

        with json_file_path.open('w') as fw:
            json.dump(run_log.dict(), fw, ensure_ascii=True, indent=4)  # pylint: disable=no-member

    def get_from_folder(self, run_id: str) -> RunLog:
        """
        Look into the run log folder for the run log for the run id.

        If the run log does not exist, raise an exception. If it does, decode it
        as a RunLog and return it

        Args:
            run_id (str): The requested run id to retrieve the run log store

        Raises:
            FileNotFoundError: If the Run Log has not been found.

        Returns:
            RunLog: The decoded Run log
        """
        write_to = self.log_folder_name

        read_from_path = Path(write_to)
        json_file_path = read_from_path / f'{run_id}.json'

        if not json_file_path.exists():
            raise FileNotFoundError(f'Expected {json_file_path} is not present')

        with json_file_path.open('r') as fr:
            json_str = json.load(fr)
            run_log = RunLog(**json_str)  # pylint: disable=no-member
        return run_log

    def create_run_log(self, run_id: str, dag_hash: str = '', use_cached: bool = False,
                       tag: str = '', original_run_id: str = '', status: str = defaults.CREATED, **kwargs) -> RunLog:
        # Creates a Run log
        # Adds it to the db
        try:
            self.get_run_log_by_id(run_id=run_id, full=False)
            raise exceptions.RunLogExistsError(run_id=run_id)
        except exceptions.RunLogNotFoundError:
            pass

        logger.info(f'{self.service_name} Creating a Run Log for : {run_id}')
        run_log = RunLog(run_id=run_id, dag_hash=dag_hash, use_cached=use_cached,
                         tag=tag, original_run_id=original_run_id, status=status)
        self.write_to_folder(run_log)
        return run_log

    def get_run_log_by_id(self, run_id: str, full: bool = False, **kwargs) -> RunLog:
        # Returns the run_log defined by id
        # Raises Exception if not found
        try:
            logger.info(f'{self.service_name} Getting a Run Log for : {run_id}')
            run_log = self.get_from_folder(run_id)
            return run_log
        except FileNotFoundError as e:
            raise exceptions.RunLogNotFoundError(run_id) from e

    def put_run_log(self, run_log: RunLog, **kwargs):
        # Puts the run_log into the database
        logger.info(f'{self.service_name} Putting the run log in the DB: {run_log.run_id}')
        self.write_to_folder(run_log)


class ChunkedFileSystemRunLogStore(BaseRunLogStore):
    """
    File system run log store but chunks the run log into thread safe chunks.
    This enables executions to be parallel.
    """
    service_name = 'chunked-fs'

    class Config(BaseModel):
        log_folder: str = defaults.LOG_LOCATION_FOLDER

    class LogTypes(Enum):
        RUN_LOG: str = 'RunLog'
        PARAMETER: str = 'Parameter'
        STEP_LOG: str = 'StepLog'
        BRANCH_LOG: str = 'BranchLog'

    class ModelTypes(Enum):
        RUN_LOG = RunLog
        PARAMETER = dict
        STEP_LOG = StepLog
        BRANCH_LOG = BranchLog

    def naming_pattern(self, log_type: LogTypes, name: str = '') -> str:
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
                    msg = (
                        f"Multiple matches found for {name} while multiple is not allowed"
                    )
                    raise Exception(msg)
                return matches[0]
            return matches

        return None

    @property
    def log_folder_name(self) -> str:
        return self.config.log_folder

    def log_folder_with_run_id(self, run_id: str) -> Path:
        return Path(self.log_folder_name) / run_id

    def safe_suffix_json(self, name: Path):
        if str(name).endswith('json'):
            return str(name)

        return str(name) + '.json'

    def _store(self, run_id: str, contents: dict, name: Path):
        """
        Store the contents against the name in the folder.

        Args:
            run_id (str): The run id
            contents (dict): The dict to store
            name (str): The name to store as
        """
        utils.safe_make_dir(self.log_folder_with_run_id(run_id=run_id))

        with open(self.safe_suffix_json(name), 'w') as fw:
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

        with open(self.safe_suffix_json(name), 'r') as fr:
            contents = json.load(fr)

        return contents

    def store(self, run_id: str, log_type: LogTypes, contents: dict, name: str = ''):
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

    def retrieve(self, run_id: str, log_type: LogTypes, name: str = '', multiple_allowed=False) -> Any:
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
        if not name and log_type not in [self.LogTypes.RUN_LOG, self.LogTypes.PARAMETER]:
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
        epoch_created = [str(match).split('-')[-1] for match in matches]  # type: ignore

        # sort matches by epoch created
        epoch_created, matches = zip(*sorted(zip(epoch_created, matches)))  # type: ignore

        logs: dict[str, Union[StepLog, BranchLog]] = {}

        for match in matches:  # type: ignore
            model = self.ModelTypes[log_type.name].value
            log_model = model(**self._retrieve(match))
            logs[log_model.internal_name] = log_model  # type: ignore

        return logs

    def _get_parent_branch(self, name: str) -> Union[str, None]:  # pylint: disable=R0201
        """
        Returns the name of the parent branch.
        If the step is part of main dag, return None.

        Args:
            name (str): The name of the step.

        Returns:
            str: The name of the branch containing the step.
        """
        dot_path = name.split('.')

        if len(dot_path) == 1:
            return None
        # Ignore the step name
        return '.'.join(dot_path[:-1])

    def _get_parent_step(self, name: str) -> Union[str, None]:  # pylint: disable=R0201
        """
        Returns the step containing the step, useful when we have steps within a branch.
        Returns None, if the step belongs to parent dag.

        Args:
            name (str): The name of the step to find the parent step it belongs to.

        Returns:
            str: The parent step the step belongs to, None if the step belongs to parent dag.
        """
        dot_path = name.split('.')

        if len(dot_path) == 1:
            return None
        # Ignore the branch.step_name
        return '.'.join(dot_path[:-2])

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
                current_branch = ordered_branches[current_branch]  # type: ignore
                step_to_add_branch = ordered_steps[step_to_add_branch]  # type: ignore
                step_to_add_branch.branches[current_branch.internal_name] = current_branch  # type: ignore

            current_branch.steps[step_internal_name] = ordered_steps[step_internal_name]

    def create_run_log(self, run_id: str, dag_hash: str = '', use_cached: bool = False,
                       tag: str = '', original_run_id: str = '', status: str = defaults.CREATED, **kwargs):
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

        logger.info(f'{self.service_name} Creating a Run Log for : {run_id}')
        run_log = RunLog(run_id=run_id, dag_hash=dag_hash, use_cached=use_cached,
                         tag=tag, original_run_id=original_run_id, status=status)

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
            logger.info(f'{self.service_name} Getting a Run Log for : {run_id}')
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

    def get_parameters(self, run_id: str, **kwargs) -> dict:  # pylint: disable=unused-argument
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

    def set_parameters(self, run_id: str, parameters: dict, **kwargs):  # pylint: disable=unused-argument
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
            self.store(run_id=run_id, log_type=self.LogTypes.PARAMETER, contents={key: value}, name=key)

    def get_run_config(self, run_id: str, **kwargs) -> dict:  # pylint: disable=unused-argument
        """
        Given a run_id, return the run_config used to perform the run.

        Args:
            run_id (str): The run_id of the run

        Returns:
            dict: The run config used for the run
        """

        run_log = self.get_run_log_by_id(run_id=run_id)
        return run_log.run_config

    def set_run_config(self, run_id: str, run_config: dict, **kwargs):  # pylint: disable=unused-argument
        """ Set the run config used to run the run_id

        Args:
            run_id (str): The run_id of the run
            run_config (dict): The run_config of the run
        """

        run_log = self.get_run_log_by_id(run_id=run_id)
        run_log.run_config.update(run_config)
        self.put_run_log(run_log=run_log)

    def get_step_log(self, internal_name: str, run_id: str, **kwargs) -> StepLog:  # pylint: disable=unused-argument
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
        logger.info(f'{self.service_name} Getting the step log: {internal_name} of {run_id}')

        step_log = self.retrieve(run_id=run_id, log_type=self.LogTypes.STEP_LOG,
                                 name=internal_name, multiple_allowed=False)

        return step_log

    def add_step_log(self, step_log: StepLog, run_id: str, **kwargs):  # pylint: disable=unused-argument
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
        logger.info(f'{self.service_name} Adding the step log to DB: {step_log.internal_name}')

        self.store(run_id=run_id, log_type=self.LogTypes.STEP_LOG,
                   contents=step_log.dict(), name=step_log.internal_name)

    def get_branch_log(self, internal_branch_name: str, run_id: str, **kwargs) -> Union[BranchLog, RunLog]:  # pylint: disable=unused-argument
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

    def add_branch_log(self, branch_log: Union[BranchLog, RunLog], run_id: str, **kwargs):  # pylint: disable=unused-argument
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
            self.put_run_log(branch_log)  # type: ignore # We are dealing with base dag here
            return

        internal_branch_name = branch_log.internal_name

        logger.info(f'{self.service_name} Adding the branch log to DB: {branch_log.internal_name}')
        self.store(run_id=run_id, log_type=self.LogTypes.BRANCH_LOG,
                   contents=branch_log.dict(), name=internal_branch_name)

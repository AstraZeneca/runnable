from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Tuple, Union

from pydantic import BaseModel

from magnus import defaults, exceptions, utils

logger = logging.getLogger(defaults.NAME)

# Once defined these classes are sealed to any additions unless a default is provided
# Breaking this rule might make magnus backwardly incompatible


class DataCatalog(BaseModel):
    """
    The captured attributes of a catalog item.
    """
    name: str  #  The name of the dataset
    data_hash: str = ''  # The sha1 hash of the file
    catalog_relative_path: str = ''  # The file path
    catalog_handler_location: str = ''
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


class CodeIdentity(BaseModel):
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
    status: str = 'FAIL'
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

    def __init__(self, config):
        self.config = config or {}

    def create_run_log(self, run_id: str, **kwargs):
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

    def get_run_log_by_id(self, run_id: str, full: bool = True, **kwargs) -> RunLog:
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
        logger.info(f'{self.service_name} Creating a Step Log: {name}')
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
        Creates a uncomitted branch log object by the internal name given

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

    def create_run_log(self, run_id: str, **kwargs) -> RunLog:
        # Creates a Run log
        # Adds it to the db
        # Return the log
        logger.info(f'{self.service_name} Creating a Run Log and adding it to DB')
        self.run_log = RunLog(run_id=run_id, status=defaults.CREATED)
        return self.run_log

    def get_run_log_by_id(self, run_id: str, full: bool = True, **kwargs):
        # Returns the run_log defined by id
        # Raises Exception if not found
        logger.info(f'{self.service_name} Getting the run log from DB for {run_id}')
        return self.run_log

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
    CONFIG_KEY_LOG_FOLDER = 'log_folder'

    @property
    def log_folder_name(self) -> str:
        if self.config:
            return self.config.get(self.CONFIG_KEY_LOG_FOLDER, defaults.LOG_LOCATION_FOLDER)

        return defaults.LOG_LOCATION_FOLDER

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

    def create_run_log(self, run_id: str, **kwargs) -> RunLog:
        # Creates a Run log
        # Adds it to the db
        logger.info(f'{self.service_name} Creating a Run Log for : {run_id}')
        run_log = RunLog(run_id=run_id, status=defaults.CREATED)
        self.write_to_folder(run_log)
        return run_log

    def get_run_log_by_id(self, run_id: str, full: bool = True, **kwargs) -> RunLog:
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

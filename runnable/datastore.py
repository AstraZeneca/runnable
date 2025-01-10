from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

from pydantic import BaseModel, Field, computed_field

from runnable import defaults, exceptions

logger = logging.getLogger(defaults.LOGGER_NAME)


JSONType = Union[
    Union[None, bool, str, float, int, List[Any], Dict[str, Any]]
]  # This is actually JSONType, but pydantic doesn't support TypeAlias yet


class DataCatalog(BaseModel, extra="allow"):
    """
    The captured attributes of a catalog item.
    """

    name: str  #  The name of the dataset
    data_hash: str = ""  # The sha1 hash of the file
    catalog_relative_path: str = ""  # The file path relative the catalog location
    catalog_handler_location: str = ""  # The location of the catalog
    stage: str = ""  # The stage at which we recorded it get, put etc

    # Needed for set operations to work on DataCatalog objects
    def __hash__(self):
        """
        Needed to Uniqueize DataCatalog objects.
        """
        return hash(self.name)

    def __eq__(self, other):
        """
        Needed for set operations to work on DataCatalog objects
        """
        if not isinstance(other, DataCatalog):
            return False
        return other.name == self.name


# The theory behind reduced:
#     parameters returned by steps in map node are only reduced by the end of the map step, fan-in.
#     If they are accessed within the map step, the value should be the value returned by the step in the map step.
#     Once the map state is complete, we can set the reduce to true and have the value as
#     the reduced value. Its either a list or a custom function return.


class JsonParameter(BaseModel):
    kind: Literal["json"]
    value: JSONType
    reduced: bool = True

    @computed_field  # type: ignore
    @property
    def description(self) -> JSONType:
        return self.value

    def get_value(self) -> JSONType:
        return self.value


class MetricParameter(BaseModel):
    kind: Literal["metric"]
    value: JSONType
    reduced: bool = True

    @computed_field  # type: ignore
    @property
    def description(self) -> JSONType:
        return self.value

    def get_value(self) -> JSONType:
        return self.value


class ObjectParameter(BaseModel):
    kind: Literal["object"]
    value: str  # The name of the pickled object
    reduced: bool = True

    @computed_field  # type: ignore
    @property
    def description(self) -> str:
        if context.run_context.object_serialisation:
            return f"Pickled object stored in catalog as: {self.value}"

        return f"Object stored in memory as: {self.value}"

    @property
    def file_name(self) -> str:
        return f"{self.value}{context.run_context.pickler.extension}"

    def get_value(self) -> Any:
        # If there was no serialisation, return the object from the return objects
        if not context.run_context.object_serialisation:
            return context.run_context.return_objects[self.value]

        # If the object was serialised, get it from the catalog
        catalog_handler = context.run_context.catalog_handler
        catalog_handler.get(name=self.file_name, run_id=context.run_context.run_id)
        obj = context.run_context.pickler.load(path=self.file_name)
        os.remove(self.file_name)  # Remove after loading
        return obj

    def put_object(self, data: Any) -> None:
        if not context.run_context.object_serialisation:
            context.run_context.return_objects[self.value] = data
            return

        # If the object was serialised, put it in the catalog
        context.run_context.pickler.dump(data=data, path=self.file_name)

        catalog_handler = context.run_context.catalog_handler
        catalog_handler.put(name=self.file_name, run_id=context.run_context.run_id)
        os.remove(self.file_name)  # Remove after loading


Parameter = Annotated[
    Union[JsonParameter, ObjectParameter, MetricParameter], Field(discriminator="kind")
]


class StepAttempt(BaseModel):
    """
    The captured attributes of an Attempt of a step.
    """

    attempt_number: int = 1
    start_time: str = ""
    end_time: str = ""
    status: str = "FAIL"
    message: str = ""
    input_parameters: Dict[str, Parameter] = Field(default_factory=dict)
    output_parameters: Dict[str, Parameter] = Field(default_factory=dict)
    user_defined_metrics: Dict[str, Parameter] = Field(default_factory=dict)

    @property
    def duration(self):
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)

        return str(end - start)


class CodeIdentity(BaseModel, extra="allow"):
    """
    The captured attributes of a code identity of a step.
    """

    code_identifier: Optional[str] = ""  # GIT sha code or docker image id
    code_identifier_type: Optional[str] = ""  # git or docker
    code_identifier_dependable: Optional[bool] = (
        False  # If git, checks if the tree is clean.
    )
    code_identifier_url: Optional[str] = (
        ""  # The git remote url or docker repository url
    )
    code_identifier_message: Optional[str] = ""  # Any optional message


class StepLog(BaseModel):
    """
    The data class capturing the data of a Step
    """

    name: str
    internal_name: str  # Should be the dot notation of the step
    status: str = "FAIL"  #  Should have a better default
    step_type: str = "task"
    message: str = ""
    mock: bool = False
    code_identities: List[CodeIdentity] = Field(default_factory=list)
    attempts: List[StepAttempt] = Field(default_factory=list)
    branches: Dict[str, BranchLog] = Field(default_factory=dict)
    data_catalog: List[DataCatalog] = Field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """
        Summarize the step log to log
        """
        summary: Dict[str, Any] = {}

        summary["Name"] = self.internal_name
        summary["Input catalog content"] = [
            dc.name for dc in self.data_catalog if dc.stage == "get"
        ]
        summary["Available parameters"] = [
            (p, v.description)
            for attempt in self.attempts
            for p, v in attempt.input_parameters.items()
        ]

        summary["Output catalog content"] = [
            dc.name for dc in self.data_catalog if dc.stage == "put"
        ]
        summary["Output parameters"] = [
            (p, v.description)
            for attempt in self.attempts
            for p, v in attempt.output_parameters.items()
        ]

        summary["Metrics"] = [
            (p, v.description)
            for attempt in self.attempts
            for p, v in attempt.user_defined_metrics.items()
        ]

        cis = []
        for ci in self.code_identities:
            message = f"{ci.code_identifier_type}:{ci.code_identifier}"
            if not ci.code_identifier_dependable:
                message += " but is not dependable"
            cis.append(message)

        summary["Code identities"] = cis

        summary["status"] = self.status

        return summary

    def get_data_catalogs_by_stage(self, stage="put") -> List[DataCatalog]:
        """
        Given a stage, return the data catalogs according to the stage

        Args:
            stage (str, optional): The stage at which the data was cataloged. Defaults to 'put'.

        Raises:
            Exception: If the stage was not in get or put.

        Returns:
            List[DataCatalog]: The list of data catalogs as per the stage.
        """
        if stage not in ["get", "put"]:
            raise Exception("Stage should be in get or put")

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
    status: str = "FAIL"
    steps: OrderedDict[str, StepLog] = Field(default_factory=OrderedDict)

    def get_data_catalogs_by_stage(self, stage="put") -> List[DataCatalog]:
        """
        Given a stage, return the data catalogs according to the stage

        Args:
            stage (str, optional): The stage at which the data was cataloged. Defaults to 'put'.

        Raises:
            Exception: If the stage was not in get or put.

        Returns:
            List[DataCatalog]: The list of data catalogs as per the stage.
        """
        if stage not in ["get", "put"]:
            raise Exception("Stage should be in get or put")

        data_catalogs = []
        for _, step in self.steps.items():
            data_catalogs.extend(step.get_data_catalogs_by_stage(stage=stage))

        return data_catalogs


# Needed for BranchLog of StepLog to be referenced
StepLog.model_rebuild()


class JobLog(BaseModel):
    """
    The data class capturing the data of a job
    This should be treated as a step log
    """

    status: str = defaults.FAIL
    message: str = ""
    mock: bool = False
    code_identities: List[CodeIdentity] = Field(default_factory=list)
    attempts: List[StepAttempt] = Field(default_factory=list)
    data_catalog: List[DataCatalog] = Field(default_factory=list)

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

    def get_summary(self) -> Dict[str, Any]:
        """
        Summarize the step log to log
        """
        summary: Dict[str, Any] = {}

        summary["Available parameters"] = [
            (p, v.description)
            for attempt in self.attempts
            for p, v in attempt.input_parameters.items()
        ]

        summary["Output catalog content"] = [
            dc.name for dc in self.data_catalog if dc.stage == "put"
        ]
        summary["Output parameters"] = [
            (p, v.description)
            for attempt in self.attempts
            for p, v in attempt.output_parameters.items()
        ]

        summary["Metrics"] = [
            (p, v.description)
            for attempt in self.attempts
            for p, v in attempt.user_defined_metrics.items()
        ]

        cis = []
        for ci in self.code_identities:
            message = f"{ci.code_identifier_type}:{ci.code_identifier}"
            if not ci.code_identifier_dependable:
                message += " but is not dependable"
            cis.append(message)

        summary["Code identities"] = cis

        summary["status"] = self.status

        return summary


class RunLog(BaseModel):
    """
    The data captured as part of Run Log
    """

    run_id: str
    dag_hash: Optional[str] = None
    tag: Optional[str] = ""
    status: str = defaults.FAIL
    steps: OrderedDict[str, StepLog] = Field(default_factory=OrderedDict)
    job: Optional[JobLog] = None
    parameters: Dict[str, Parameter] = Field(default_factory=dict)
    run_config: Dict[str, Any] = Field(default_factory=dict)

    def get_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        _context = context.run_context

        summary["Unique execution id"] = self.run_id
        summary["status"] = self.status

        summary["Catalog Location"] = _context.catalog_handler.get_summary()
        summary["Full Run log present at: "] = _context.run_log_store.get_summary()

        run_log = _context.run_log_store.get_run_log_by_id(
            run_id=_context.run_id, full=True
        )

        summary["Final Parameters"] = {
            p: v.description for p, v in run_log.parameters.items()
        }
        summary["Collected metrics"] = {
            p: v.description
            for p, v in run_log.parameters.items()
            if v.kind == "metric"
        }

        return summary

    def get_data_catalogs_by_stage(self, stage: str = "put") -> List[DataCatalog]:
        """
        Return all the cataloged data by the stage at which they were cataloged.

        Raises:
            Exception: If stage was not either put or get.

        Args:
            stage (str, optional): [description]. Defaults to 'put'.
        """
        if stage not in ["get", "put"]:
            raise Exception("Only get or put are allowed in stage")

        data_catalogs = []
        for _, step in self.steps.items():
            data_catalogs.extend(step.get_data_catalogs_by_stage(stage=stage))

        return list(set(data_catalogs))

    def search_branch_by_internal_name(
        self, i_name: str
    ) -> Tuple[Union[BranchLog, RunLog], Union[StepLog, None]]:
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

        dot_path = i_name.split(".")

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
                current_branch = current_step.branches[".".join(dot_path[: i + 1])]  # type: ignore
                current_steps = current_branch.steps
                logger.debug(f"Finding branch {i_name} in branch: {current_branch}")
            else:
                # Its even, so we are in step, we start here!
                # Get the step that holds the branch
                current_step = current_steps[".".join(dot_path[: i + 1])]
                logger.debug(f"Finding branch for {i_name} in step: {current_step}")

        logger.debug(f"current branch : {current_branch}, current step {current_step}")
        if current_branch and current_step:
            return current_branch, current_step

        raise exceptions.BranchLogNotFoundError(self.run_id, i_name)

    def search_step_by_internal_name(
        self, i_name: str
    ) -> Tuple[StepLog, Union[BranchLog, None]]:
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
        dot_path = i_name.split(".")
        if len(dot_path) == 1:
            try:
                return self.steps[i_name], None
            except KeyError as e:
                raise exceptions.StepLogNotFoundError(self.run_id, i_name) from e

        current_steps = self.steps
        current_step = None
        current_branch = None
        for i in range(len(dot_path)):
            if i % 2:
                # Its odd, so we are in brach name
                current_branch = current_step.branches[".".join(dot_path[: i + 1])]  # type: ignore
                current_steps = current_branch.steps
                logger.debug(
                    f"Finding step log for {i_name} in branch: {current_branch}"
                )
            else:
                # Its even, so we are in step, we start here!
                current_step = current_steps[".".join(dot_path[: i + 1])]
                logger.debug(f"Finding step log for {i_name} in step: {current_step}")

        logger.debug(f"current branch : {current_branch}, current step {current_step}")
        if current_branch and current_step:
            return current_step, current_branch

        raise exceptions.StepLogNotFoundError(self.run_id, i_name)


class BaseRunLogStore(ABC, BaseModel):
    """
    The base class of a Run Log Store with many common methods implemented.
    """

    service_name: str = ""
    service_type: str = "run_log_store"

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]: ...

    @property
    def _context(self):
        return context.run_context

        """
        Retrieves a Job log from the database using the config and the job_id

        Args:
            job_id (str): The job_id of the job

        Returns:
            JobLog: The JobLog object identified by the job_id

        Logically the method should:
            * Returns the job_log defined by id from the data store defined by the config

        Raises:
            NotImplementedError: This is a base class and therefore has no default implementation
            JobLogNotFoundError: If the job log for job_id is not found in the datastore
        """

    @abstractmethod
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
        Raises:
            NotImplementedError: This is a base class and therefore has no default implementation
        """

        raise NotImplementedError

    @abstractmethod
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

        Raises:
            NotImplementedError: This is a base class and therefore has no default implementation
            RunLogNotFoundError: If the run log for run_id is not found in the datastore
        """

        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    def update_run_log_status(self, run_id: str, status: str):
        """
        Updates the status of the Run Log defined by the run_id

        Args:
            run_id (str): The run_id of the run
            status (str): The new status of the run
        """
        logger.info(f"Updating status of run_id {run_id} to {status}")
        run_log = self.get_run_log_by_id(run_id, full=False)
        run_log.status = status
        self.put_run_log(run_log)

    def get_parameters(self, run_id: str) -> Dict[str, Parameter]:
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

    def set_parameters(self, run_id: str, parameters: Dict[str, Parameter]):
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

    def create_step_log(self, name: str, internal_name: str):
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
        logger.info(f"{self.service_name} Creating a Step Log: {internal_name}")
        return StepLog(name=name, internal_name=internal_name, status=defaults.CREATED)

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
        run_log = self.get_run_log_by_id(run_id=run_id)
        step_log, _ = run_log.search_step_by_internal_name(internal_name)
        return step_log

    def add_step_log(self, step_log: StepLog, run_id: str):
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
        logger.info(f"{self.service_name} Adding the step log to DB: {step_log.name}")
        run_log = self.get_run_log_by_id(run_id=run_id)

        branch_to_add = ".".join(step_log.internal_name.split(".")[:-1])
        branch, _ = run_log.search_branch_by_internal_name(branch_to_add)

        if branch is None:
            branch = run_log
        branch.steps[step_log.internal_name] = step_log
        self.put_run_log(run_log=run_log)

    def create_branch_log(self, internal_branch_name: str) -> BranchLog:
        """
        Creates a uncommitted branch log object by the internal name given

        Args:
            internal_branch_name (str): Creates a branch log by name internal_branch_name

        Returns:
            BranchLog: Uncommitted and initialized with defaults BranchLog object
        """
        # Create a new BranchLog
        logger.info(
            f"{self.service_name} Creating a Branch Log : {internal_branch_name}"
        )
        return BranchLog(internal_name=internal_branch_name, status=defaults.CREATED)

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
        run_log = self.get_run_log_by_id(run_id=run_id)
        if not internal_branch_name:
            return run_log
        branch, _ = run_log.search_branch_by_internal_name(internal_branch_name)
        return branch

    def add_branch_log(self, branch_log: Union[BranchLog, RunLog], run_id: str):
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

        step_name = ".".join(internal_branch_name.split(".")[:-1])
        step, _ = run_log.search_step_by_internal_name(step_name)

        step.branches[internal_branch_name] = branch_log  # type: ignore
        self.put_run_log(run_log)

    def create_code_identity(self) -> CodeIdentity:
        """
        Creates an uncommitted Code identity class

        Returns:
            CodeIdentity: An uncommitted code identity class
        """
        logger.info(f"{self.service_name} Creating Code identity")
        return CodeIdentity()

    def create_data_catalog(self, name: str) -> DataCatalog:
        """
        Create a uncommitted data catalog object

        Args:
            name (str): The name of the data catalog item to put

        Returns:
            DataCatalog: The DataCatalog object.
        """
        logger.info(f"{self.service_name} Creating Data Catalog for {name}")
        return DataCatalog(name=name)

    def create_job_log(self) -> JobLog:
        """
        Creates a Job log and adds it to the db

        Refer to BaseRunLogStore.create_job_log
        """
        logger.info(f"{self.service_name} Creating a Job Log and adding it to DB")
        return JobLog(status=defaults.CREATED)

    def get_job_log(self, run_id: str) -> JobLog:
        """
        Returns the run_log defined by id

        Raises Exception if not found
        """
        logger.info(f"{self.service_name} Getting the run log from DB for {run_id}")
        run_log = self.get_run_log_by_id(run_id)

        try:
            assert run_log.job
        except AssertionError as exc:
            raise exceptions.JobLogNotFoundError(run_id) from exc

        return run_log.job

    def add_job_log(self, run_id: str, job_log: JobLog):
        """
        Adds the job log to the run log

        Args:
            run_id (str): The run_id of the run
            job_log (JobLog): The job log to add to the run log
        """
        logger.info(f"{self.service_name} Adding the job log to DB for: {run_id}")
        run_log = self.get_run_log_by_id(run_id=run_id)
        run_log.job = job_log
        run_log.status = job_log.status
        self.put_run_log(run_log=run_log)


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

    service_name: str = "buffered"

    run_log: Optional[RunLog] = Field(
        default=None, exclude=True
    )  # For a buffered Run Log, this is the database
    job_log: Optional[JobLog] = Field(
        default=None, exclude=True
    )  # For a buffered Run Log, this is the database

    def get_summary(self) -> Dict[str, Any]:
        summary = {"Type": self.service_name, "Location": "Not persisted"}

        return summary

    def create_run_log(
        self,
        run_id: str,
        dag_hash: str = "",
        use_cached: bool = False,
        tag: str = "",
        original_run_id: str = "",
        status: str = defaults.CREATED,
    ) -> RunLog:
        """
        # Creates a Run log
        # Adds it to the db
        # Return the log

        Refer to BaseRunLogStore.create_run_log
        """

        logger.info(f"{self.service_name} Creating a Run Log and adding it to DB")
        self.run_log = RunLog(
            run_id=run_id,
            dag_hash=dag_hash,
            tag=tag,
            status=status,
        )
        return self.run_log

    def get_run_log_by_id(self, run_id: str, full: bool = False):
        """
        # Returns the run_log defined by id
        # Raises Exception if not found
        """

        logger.info(f"{self.service_name} Getting the run log from DB for {run_id}")
        if self.run_log:
            return self.run_log

        raise exceptions.RunLogNotFoundError(run_id)

    def put_run_log(self, run_log: RunLog):
        """
        # Puts the run log in the db
        # Raises Exception if not found
        """
        logger.info(
            f"{self.service_name} Putting the run log in the DB: {run_log.run_id}"
        )
        self.run_log = run_log


import runnable.context as context  # noqa: F401, E402

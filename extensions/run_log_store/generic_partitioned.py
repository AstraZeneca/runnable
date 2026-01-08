import logging
from abc import abstractmethod
from typing import Dict

from runnable import defaults
from runnable.datastore import BaseRunLogStore, Parameter, StepLog, BranchLog

logger = logging.getLogger(defaults.LOGGER_NAME)


class GenericPartitionedRunLogStore(BaseRunLogStore):
    """
    A generic implementation of a RunLogStore that supports partitioned parameter storage.

    This store enables branch-specific parameter isolation by storing parameters
    in separate partitions/files per branch execution context.
    """

    service_name: str = "generic-partitioned"
    supports_parallel_writes: bool = True

    # Root partition abstract methods
    @abstractmethod
    def _store_root_parameters(
        self, run_id: str, parameters: Dict[str, Parameter]
    ) -> None:
        """Store parameters in root partition."""
        ...

    @abstractmethod
    def _retrieve_root_parameters(self, run_id: str) -> Dict[str, Parameter]:
        """Retrieve parameters from root partition."""
        ...

    @abstractmethod
    def _store_root_step_log(self, run_id: str, step_log: StepLog) -> None:
        """Store step log in root partition."""
        ...

    @abstractmethod
    def _retrieve_root_step_log(self, run_id: str, step_name: str) -> StepLog:
        """Retrieve step log from root partition."""
        ...

    @abstractmethod
    def _store_root_branch_log(self, run_id: str, branch_log: BranchLog) -> None:
        """Store branch log in root partition."""
        ...

    @abstractmethod
    def _retrieve_root_branch_log(self, run_id: str, branch_name: str) -> BranchLog:
        """Retrieve branch log from root partition."""
        ...

    # Branch partition abstract methods
    @abstractmethod
    def _store_branch_parameters(
        self, run_id: str, internal_branch_name: str, parameters: Dict[str, Parameter]
    ) -> None:
        """Store parameters in branch partition."""
        ...

    @abstractmethod
    def _retrieve_branch_parameters(
        self, run_id: str, internal_branch_name: str
    ) -> Dict[str, Parameter]:
        """Retrieve parameters from branch partition."""
        ...

    @abstractmethod
    def _store_branch_step_log(
        self, run_id: str, internal_branch_name: str, step_log: StepLog
    ) -> None:
        """Store step log in branch partition."""
        ...

    @abstractmethod
    def _retrieve_branch_step_log(
        self, run_id: str, internal_branch_name: str, step_name: str
    ) -> StepLog:
        """Retrieve step log from branch partition."""
        ...

    @abstractmethod
    def _store_branch_branch_log(
        self, run_id: str, internal_branch_name: str, branch_log: BranchLog
    ) -> None:
        """Store branch log in branch partition."""
        ...

    @abstractmethod
    def _retrieve_branch_branch_log(
        self, run_id: str, internal_branch_name: str, branch_name: str
    ) -> BranchLog:
        """Retrieve branch log from branch partition."""
        ...

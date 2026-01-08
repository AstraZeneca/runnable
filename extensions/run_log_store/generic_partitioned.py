import logging
from abc import abstractmethod
from typing import Dict, Optional, Union

from runnable import defaults
from runnable.datastore import BaseRunLogStore, Parameter, RunLog, StepLog, BranchLog

logger = logging.getLogger(defaults.LOGGER_NAME)


class GenericPartitionedRunLogStore(BaseRunLogStore):
    """
    A generic implementation of a RunLogStore that supports full partitioned storage.

    This implementation provides complete partitioning of all data types (parameters,
    step logs, branch logs) with hierarchical storage structure. Each branch execution
    context gets isolated storage while supporting parameter inheritance from parent contexts.

    Key Features:
    - Full data partitioning per branch execution context
    - Hierarchical storage: root -> branches -> nested branches
    - Parameter inheritance during branch creation
    - Transparent routing based on internal_branch_name
    - Backward compatibility with existing domain code

    Storage Structure:
        /run_id/                    # Root partition
        ├── parameters/             # Root parameters
        ├── step_logs/              # Root step logs
        ├── branch_logs/            # Root branch logs
        └── branches/
            └── branch_a/           # Branch partition
                ├── parameters/     # Branch-specific parameters
                ├── step_logs/      # Branch-specific step logs
                ├── branch_logs/    # Branch-specific branch logs
                └── branches/       # Nested branches...

    Abstract Methods:
        Concrete implementations must provide storage methods for:
        - Root partition: _store_root_*, _retrieve_root_*
        - Branch partition: _store_branch_*, _retrieve_branch_*

        Each method handles one data type (parameters, step_logs, branch_logs)
        in one partition type (root or branch).
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

    def copy_parameters_to_branch(
        self, run_id: str, source_branch: Optional[str], target_branch: str
    ):
        """
        Copy parameters from source to target branch during branch creation.
        This ensures nested branch steps have access to parent context parameters.

        Args:
            run_id: The run ID
            source_branch: Source branch name (None for root parameters)
            target_branch: Target branch name to inherit parameters
        """
        if source_branch is None:
            # Copy from root to new branch
            source_params = self._retrieve_root_parameters(run_id)
        else:
            # Copy from parent branch to nested branch
            source_params = self._retrieve_branch_parameters(run_id, source_branch)

        # Deep copy parameters to target branch for isolation
        self._store_branch_parameters(run_id, target_branch, source_params.copy())

    def get_parameters(
        self, run_id: str, internal_branch_name: Optional[str] = None
    ) -> Dict[str, Parameter]:
        """
        Get parameters with branch-specific support.

        Args:
            run_id: The run ID
            internal_branch_name: If provided, get parameters for specific branch

        Returns:
            Dictionary of parameters
        """
        if internal_branch_name is None:
            # Get main run parameters
            return self._retrieve_root_parameters(run_id)
        else:
            # Get branch-specific parameters
            return self._retrieve_branch_parameters(run_id, internal_branch_name)

    def set_parameters(
        self,
        run_id: str,
        parameters: Dict[str, Parameter],
        internal_branch_name: Optional[str] = None,
    ):
        """
        Set parameters with branch-specific support.

        Args:
            run_id: The run ID
            parameters: Parameters to set
            internal_branch_name: If provided, set parameters for specific branch
        """
        if internal_branch_name is None:
            # Set main run parameters
            self._store_root_parameters(run_id, parameters)
        else:
            # Set branch-specific parameters
            self._store_branch_parameters(run_id, internal_branch_name, parameters)

    def get_step_log(
        self,
        internal_name: str,
        run_id: str,
        internal_branch_name: Optional[str] = None,
    ) -> StepLog:
        """
        Get a step log from the datastore for run_id and the internal naming of the step log.

        Args:
            internal_name: The internal name of the step log
            run_id: The run_id of the run
            internal_branch_name: If provided, get from specific branch partition

        Returns:
            StepLog: The step log object for the step
        """
        if internal_branch_name is None:
            # Get from root partition
            return self._retrieve_root_step_log(run_id, internal_name)
        else:
            # Get from branch partition
            return self._retrieve_branch_step_log(
                run_id, internal_branch_name, internal_name
            )

    def add_step_log(
        self, step_log: StepLog, run_id: str, internal_branch_name: Optional[str] = None
    ):
        """
        Add the step log in the run log as identified by the run_id in the datastore.

        Args:
            step_log: The Step log to add to the database
            run_id: The run id of the run
            internal_branch_name: If provided, store in specific branch partition
        """
        if internal_branch_name is None:
            # Store in root partition
            self._store_root_step_log(run_id, step_log)
        else:
            # Store in branch partition
            self._store_branch_step_log(run_id, internal_branch_name, step_log)

    def get_branch_log(
        self,
        internal_branch_name: str,
        run_id: str,
        parent_branch_name: Optional[str] = None,
    ) -> Union[BranchLog, RunLog]:
        """
        Returns the branch log by the internal branch name for the run id.

        If the internal branch name is empty, returns the run log.

        Args:
            internal_branch_name: The internal branch name to retrieve
            run_id: The run id of interest
            parent_branch_name: The parent branch containing this branch (None for root)

        Returns:
            BranchLog or RunLog: The branch log or the run log as requested
        """
        if not internal_branch_name:
            # Return root run log
            return self.get_run_log_by_id(run_id=run_id)

        if parent_branch_name is None:
            # Retrieve from root partition
            return self._retrieve_root_branch_log(run_id, internal_branch_name)
        else:
            # Retrieve from parent branch partition
            return self._retrieve_branch_branch_log(
                run_id, parent_branch_name, internal_branch_name
            )

    def add_branch_log(
        self,
        branch_log: Union[BranchLog, RunLog],
        run_id: str,
        parent_branch_name: Optional[str] = None,
    ):
        """
        Add the branch log to the appropriate partition.

        Args:
            branch_log: The branch log/run log to add to the database
            run_id: The run id to which the branch/run log is added
            parent_branch_name: The parent branch containing this branch (None for root)
        """
        if not isinstance(branch_log, BranchLog):
            # This is a RunLog, store it directly
            self.put_run_log(branch_log)
            return

        if parent_branch_name is None:
            # Store in root partition
            self._store_root_branch_log(run_id, branch_log)
        else:
            # Store in parent branch partition
            self._store_branch_branch_log(run_id, parent_branch_name, branch_log)

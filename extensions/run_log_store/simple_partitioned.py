"""
Simplified PartitionedRunLogStore implementation.

This replaces the complex hierarchical GenericPartitionedRunLogStore with a much simpler
approach: one JSON file per branch, each containing a complete BranchLog.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

from pydantic import Field
from runnable import defaults, exceptions
from runnable.datastore import BaseRunLogStore, RunLog, BranchLog

logger = logging.getLogger(defaults.LOGGER_NAME)


class SimplePartitionedRunLogStore(BaseRunLogStore):
    """
    A simplified partitioned RunLogStore that uses one JSON file per branch.

    Key Insight: The complexity reduces to just "finding the right BranchLog".
    This store implements the same interface as BaseRunLogStore, but finds
    BranchLogs in separate JSON files instead of within the main run_log.

    Storage Structure:
        /run_id/
        ├── run_log.json         # Root RunLog with root parameters
        ├── branch_a.json        # Complete BranchLog for branch_a
        └── branch_b.json        # Complete BranchLog for branch_b

    Benefits:
    - Eliminates 12 abstract storage methods from GenericPartitionedRunLogStore
    - Same interface as BaseRunLogStore (identical get_parameters/set_parameters)
    - Only difference: where BranchLogs are found (separate files vs single JSON)
    - Supports parallel writes since each branch has its own file
    """

    service_name: str = "simple-partitioned"
    supports_parallel_writes: bool = True
    log_folder: Path = Field(default_factory=lambda: Path(".runnable"))

    def __init__(self, log_folder: str = ".runnable"):
        super().__init__()
        self.log_folder = Path(log_folder)

    def get_summary(self) -> Dict:
        return {"Type": self.service_name, "Location": str(self.log_folder)}

    def create_run_log(
        self,
        run_id: str,
        dag_hash: str = "",
        use_cached: bool = False,
        tag: str = "",
        original_run_id: str = "",
        status: str = defaults.CREATED,
    ) -> RunLog:
        """Create run log and ensure directory exists."""
        run_log = RunLog(run_id=run_id, dag_hash=dag_hash, tag=tag, status=status)
        self._ensure_run_dir(run_id)
        self.put_run_log(run_log)
        return run_log

    def get_run_log_by_id(self, run_id: str, full: bool = False) -> RunLog:
        """Load run log from root JSON file."""
        run_log_path = self._get_run_log_path(run_id)
        if not run_log_path.exists():
            raise exceptions.RunLogNotFoundError(run_id)

        with open(run_log_path, "r") as f:
            data = json.load(f)

        return RunLog.model_validate(data)

    def put_run_log(self, run_log: RunLog):
        """Save run log to root JSON file."""
        run_log_path = self._get_run_log_path(run_log.run_id)
        self._ensure_run_dir(run_log.run_id)

        with open(run_log_path, "w") as f:
            json.dump(run_log.model_dump(), f, indent=2)

    def get_branch_log(
        self,
        internal_branch_name: str,
        run_id: str,
        parent_branch_name: Optional[str] = None,
    ) -> Union[BranchLog, RunLog]:
        """
        KEY DIFFERENCE: Load BranchLog from separate JSON file.

        This is the only method that differs from BaseRunLogStore - it finds
        the BranchLog in a separate file instead of within the run_log.
        """
        if not internal_branch_name:
            return self.get_run_log_by_id(run_id)

        branch_log_path = self._get_branch_log_path(run_id, internal_branch_name)
        if not branch_log_path.exists():
            raise exceptions.BranchLogNotFoundError(run_id, internal_branch_name)

        with open(branch_log_path, "r") as f:
            data = json.load(f)

        return BranchLog.model_validate(data)

    def add_branch_log(
        self,
        branch_log: Union[BranchLog, RunLog],
        run_id: str,
        parent_branch_name: Optional[str] = None,
    ):
        """
        Save BranchLog to separate JSON file.

        This enables parallel writes - each branch can be updated independently.
        """
        if isinstance(branch_log, RunLog):
            # This is the root run log
            self.put_run_log(branch_log)
            return

        # Save branch to separate file
        branch_log_path = self._get_branch_log_path(run_id, branch_log.internal_name)
        self._ensure_run_dir(run_id)

        with open(branch_log_path, "w") as f:
            json.dump(branch_log.model_dump(), f, indent=2)

    # Helper methods for file paths
    def _get_run_log_path(self, run_id: str) -> Path:
        """Get path to root run_log.json file."""
        return self.log_folder / run_id / "run_log.json"

    def _get_branch_log_path(self, run_id: str, internal_branch_name: str) -> Path:
        """Get path to branch JSON file."""
        # Simple naming: replace dots with underscores for file safety
        safe_name = internal_branch_name.replace(".", "_")
        return self.log_folder / run_id / f"{safe_name}.json"

    def _ensure_run_dir(self, run_id: str):
        """Ensure run directory exists."""
        run_dir = self.log_folder / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: get_parameters and set_parameters are inherited from BaseRunLogStore!
    # They already implement the unified interface design correctly.
    # The only difference is that get_branch_log/add_branch_log use separate files.

import json
import logging
from pathlib import Path
from typing import Any, Dict

from extensions.run_log_store.generic_partitioned import GenericPartitionedRunLogStore
from runnable import defaults, utils
from runnable.datastore import (
    Parameter,
    RunLog,
    StepLog,
    BranchLog,
    JsonParameter,
    MetricParameter,
    ObjectParameter,
)

logger = logging.getLogger(defaults.LOGGER_NAME)


class FileSystemPartitionedRunLogStore(GenericPartitionedRunLogStore):
    """
    File system implementation of GenericPartitionedRunLogStore with hierarchical storage.

    Storage Structure:
        /log_folder/run_id/                    # Root partition
        ├── parameters/                        # Root parameters as JSON files
        ├── steps/                             # Root step logs as JSON files
        ├── branches/                          # Root branch logs as JSON files
        └── branch_partitions/
            └── branch_name/                   # Branch partition
                ├── parameters/                # Branch-specific parameters
                ├── steps/                     # Branch-specific step logs
                ├── branches/                  # Branch-specific branch logs
                └── branch_partitions/         # Nested branches...
    """

    service_name: str = "partitioned-fs"
    log_folder: str = defaults.LOG_LOCATION_FOLDER
    supports_parallel_writes: bool = True

    def get_summary(self) -> Dict[str, Any]:
        return {"Type": self.service_name, "Location": self.log_folder}

    def log_folder_with_run_id(self, run_id: str) -> Path:
        """Get the root folder for a run id."""
        return Path(self.log_folder) / run_id

    def safe_suffix_json(self, name: str) -> str:
        """Safely attach .json suffix to filename."""
        if str(name).endswith(".json"):
            return str(name)
        return str(name) + ".json"

    def _get_branch_path(self, run_id: str, internal_branch_name: str) -> Path:
        """Get the path for a branch partition."""
        base_path = self.log_folder_with_run_id(run_id)
        branch_parts = internal_branch_name.split(".")

        # Navigate through nested branch structure
        current_path = base_path
        for part in branch_parts:
            current_path = current_path / "branch_partitions" / part

        return current_path

    # Root partition implementations

    def _store_root_parameters(
        self, run_id: str, parameters: Dict[str, Parameter]
    ) -> None:
        """Store parameters in root partition as individual JSON files."""
        params_folder = self.log_folder_with_run_id(run_id) / "parameters"
        utils.safe_make_dir(params_folder)

        for param_name, param in parameters.items():
            param_file = params_folder / self.safe_suffix_json(param_name)
            param_data = json.loads(param.model_dump_json(by_alias=True))

            with open(param_file, "w") as fw:
                json.dump(param_data, fw, ensure_ascii=True, indent=4)

    def _retrieve_root_parameters(self, run_id: str) -> Dict[str, Parameter]:
        """Retrieve parameters from root partition."""
        params_folder = self.log_folder_with_run_id(run_id) / "parameters"
        parameters = {}

        if not params_folder.exists():
            return parameters

        for param_file in params_folder.glob("*.json"):
            param_name = param_file.stem

            with open(param_file, "r") as fr:
                param_json = fr.read()

            # Reconstruct Parameter object based on kind
            param_data = json.loads(param_json)
            if param_data["kind"] == "json":
                parameters[param_name] = JsonParameter.model_validate_json(param_json)
            elif param_data["kind"] == "metric":
                parameters[param_name] = MetricParameter.model_validate_json(param_json)
            elif param_data["kind"] == "object":
                parameters[param_name] = ObjectParameter.model_validate_json(param_json)

        return parameters

    def _store_root_step_log(self, run_id: str, step_log: StepLog) -> None:
        """Store step log in root partition."""
        steps_folder = self.log_folder_with_run_id(run_id) / "steps"
        utils.safe_make_dir(steps_folder)

        step_file = steps_folder / self.safe_suffix_json(step_log.internal_name)
        step_data = json.loads(step_log.model_dump_json())

        with open(step_file, "w") as fw:
            json.dump(step_data, fw, ensure_ascii=True, indent=4)

    def _retrieve_root_step_log(self, run_id: str, step_name: str) -> StepLog:
        """Retrieve step log from root partition."""
        steps_folder = self.log_folder_with_run_id(run_id) / "steps"
        step_file = steps_folder / self.safe_suffix_json(step_name)

        with open(step_file, "r") as fr:
            step_json = fr.read()

        return StepLog.model_validate_json(step_json)

    def _store_root_branch_log(self, run_id: str, branch_log: BranchLog) -> None:
        """Store branch log in root partition."""
        branches_folder = self.log_folder_with_run_id(run_id) / "branches"
        utils.safe_make_dir(branches_folder)

        branch_file = branches_folder / self.safe_suffix_json(branch_log.internal_name)
        branch_data = json.loads(branch_log.model_dump_json())

        with open(branch_file, "w") as fw:
            json.dump(branch_data, fw, ensure_ascii=True, indent=4)

    def _retrieve_root_branch_log(self, run_id: str, branch_name: str) -> BranchLog:
        """Retrieve branch log from root partition."""
        branches_folder = self.log_folder_with_run_id(run_id) / "branches"
        branch_file = branches_folder / self.safe_suffix_json(branch_name)

        with open(branch_file, "r") as fr:
            branch_json = fr.read()

        return BranchLog.model_validate_json(branch_json)

    # Branch partition implementations

    def _store_branch_parameters(
        self, run_id: str, internal_branch_name: str, parameters: Dict[str, Parameter]
    ) -> None:
        """Store parameters in branch partition."""
        branch_path = self._get_branch_path(run_id, internal_branch_name)
        params_folder = branch_path / "parameters"
        utils.safe_make_dir(params_folder)

        for param_name, param in parameters.items():
            param_file = params_folder / self.safe_suffix_json(param_name)
            param_data = json.loads(param.model_dump_json(by_alias=True))

            with open(param_file, "w") as fw:
                json.dump(param_data, fw, ensure_ascii=True, indent=4)

    def _retrieve_branch_parameters(
        self, run_id: str, internal_branch_name: str
    ) -> Dict[str, Parameter]:
        """Retrieve parameters from branch partition."""
        branch_path = self._get_branch_path(run_id, internal_branch_name)
        params_folder = branch_path / "parameters"
        parameters = {}

        if not params_folder.exists():
            return parameters

        for param_file in params_folder.glob("*.json"):
            param_name = param_file.stem

            with open(param_file, "r") as fr:
                param_json = fr.read()

            # Reconstruct Parameter object based on kind
            param_data = json.loads(param_json)
            if param_data["kind"] == "json":
                parameters[param_name] = JsonParameter.model_validate_json(param_json)
            elif param_data["kind"] == "metric":
                parameters[param_name] = MetricParameter.model_validate_json(param_json)
            elif param_data["kind"] == "object":
                parameters[param_name] = ObjectParameter.model_validate_json(param_json)

        return parameters

    def _store_branch_step_log(
        self, run_id: str, internal_branch_name: str, step_log: StepLog
    ) -> None:
        """Store step log in branch partition."""
        branch_path = self._get_branch_path(run_id, internal_branch_name)
        steps_folder = branch_path / "steps"
        utils.safe_make_dir(steps_folder)

        step_file = steps_folder / self.safe_suffix_json(step_log.internal_name)
        step_data = json.loads(step_log.model_dump_json())

        with open(step_file, "w") as fw:
            json.dump(step_data, fw, ensure_ascii=True, indent=4)

    def _retrieve_branch_step_log(
        self, run_id: str, internal_branch_name: str, step_name: str
    ) -> StepLog:
        """Retrieve step log from branch partition."""
        branch_path = self._get_branch_path(run_id, internal_branch_name)
        steps_folder = branch_path / "steps"
        step_file = steps_folder / self.safe_suffix_json(step_name)

        with open(step_file, "r") as fr:
            step_json = fr.read()

        return StepLog.model_validate_json(step_json)

    def _store_branch_branch_log(
        self, run_id: str, internal_branch_name: str, branch_log: BranchLog
    ) -> None:
        """Store branch log in branch partition."""
        branch_path = self._get_branch_path(run_id, internal_branch_name)
        branches_folder = branch_path / "branches"
        utils.safe_make_dir(branches_folder)

        branch_file = branches_folder / self.safe_suffix_json(branch_log.internal_name)
        branch_data = json.loads(branch_log.model_dump_json())

        with open(branch_file, "w") as fw:
            json.dump(branch_data, fw, ensure_ascii=True, indent=4)

    def _retrieve_branch_branch_log(
        self, run_id: str, internal_branch_name: str, branch_name: str
    ) -> BranchLog:
        """Retrieve branch log from branch partition."""
        branch_path = self._get_branch_path(run_id, internal_branch_name)
        branches_folder = branch_path / "branches"
        branch_file = branches_folder / self.safe_suffix_json(branch_name)

        with open(branch_file, "r") as fr:
            branch_json = fr.read()

        return BranchLog.model_validate_json(branch_json)

    # BaseRunLogStore abstract method implementations

    def create_run_log(
        self,
        run_id: str,
        dag_hash: str = "",
        use_cached: bool = False,
        tag: str = "",
        original_run_id: str = "",
        status: str = defaults.CREATED,
    ) -> RunLog:
        """Create a new run log."""
        run_log = RunLog(
            run_id=run_id,
            dag_hash=dag_hash,
            tag=tag,
            status=status,
        )
        self.put_run_log(run_log)
        return run_log

    def get_run_log_by_id(self, run_id: str, full: bool = False) -> RunLog:
        """Get run log by ID."""
        run_file = self.log_folder_with_run_id(run_id) / "run_log.json"

        with open(run_file, "r") as fr:
            run_json = fr.read()

        return RunLog.model_validate_json(run_json)

    def put_run_log(self, run_log: RunLog):
        """Store the run log."""
        run_folder = self.log_folder_with_run_id(run_log.run_id)
        utils.safe_make_dir(run_folder)

        run_file = run_folder / "run_log.json"
        run_data = json.loads(run_log.model_dump_json())

        with open(run_file, "w") as fw:
            json.dump(run_data, fw, ensure_ascii=True, indent=4)

from typing import Dict, Optional

from pydantic import BaseModel, SerializeAsAny

from magnus.catalog import BaseCatalog
from magnus.datastore import BaseRunLogStore
from magnus.executor import BaseExecutor
from magnus.experiment_tracker import BaseExperimentTracker
from magnus.graph import Graph
from magnus.secrets import BaseSecrets


class Context(BaseModel):
    executor: SerializeAsAny[BaseExecutor]
    run_log_store: SerializeAsAny[BaseRunLogStore]
    secrets_handler: SerializeAsAny[BaseSecrets]
    catalog_handler: SerializeAsAny[BaseCatalog]
    experiment_tracker: SerializeAsAny[BaseExperimentTracker]

    pipeline_file: Optional[str] = ""
    parameters_file: Optional[str] = ""
    configuration_file: Optional[str] = ""

    tag: str = ""
    run_id: str = ""
    variables: Dict[str, str] = {}
    use_cached: bool = False
    original_run_id: str = ""
    dag: Optional[Graph] = None
    dag_hash: str = ""
    execution_plan: str = ""


run_context = None  # type: Context # type: ignore

from typing import Dict, Optional

from pydantic import BaseModel

from magnus.catalog import BaseCatalog
from magnus.datastore import BaseRunLogStore
from magnus.executor import BaseExecutor
from magnus.experiment_tracker import BaseExperimentTracker
from magnus.graph import Graph
from magnus.secrets import BaseSecrets


class Context(BaseModel):
    executor: BaseExecutor
    run_log_store: BaseRunLogStore
    secrets_handler: BaseSecrets
    catalog_handler: BaseCatalog
    experiment_tracker: BaseExperimentTracker

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

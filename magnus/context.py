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

    tag: Optional[str] = ""
    run_id: str = ""
    variables: Dict[str, str] = {}
    use_cached: Optional[str] = ""
    dag: Optional[Graph] = None  # TODO: Check how the dict() will work
    dag_hash: Optional[str] = ""
    execution_plan: str = ""

    class Config:
        arbitrary_types_allowed = True


run_context = None  # type: Context # type: ignore

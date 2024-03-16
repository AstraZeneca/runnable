from typing import Dict, Optional

from pydantic import BaseModel, SerializeAsAny

from runnable.catalog import BaseCatalog
from runnable.datastore import BaseRunLogStore
from runnable.executor import BaseExecutor
from runnable.experiment_tracker import BaseExperimentTracker
from runnable.graph import Graph
from runnable.pickler import BasePickler
from runnable.secrets import BaseSecrets


class Context(BaseModel):
    executor: SerializeAsAny[BaseExecutor]
    run_log_store: SerializeAsAny[BaseRunLogStore]
    secrets_handler: SerializeAsAny[BaseSecrets]
    catalog_handler: SerializeAsAny[BaseCatalog]
    experiment_tracker: SerializeAsAny[BaseExperimentTracker]
    pickler: SerializeAsAny[BasePickler]

    pipeline_file: Optional[str] = ""
    parameters_file: Optional[str] = ""
    configuration_file: Optional[str] = ""

    tag: str = ""
    run_id: str = ""
    variables: Dict[str, str] = {}
    dag: Optional[Graph] = None
    dag_hash: str = ""
    execution_plan: str = ""


run_context = None  # type: Context # type: ignore

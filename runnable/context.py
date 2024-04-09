from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
from rich.progress import Progress

from runnable.catalog import BaseCatalog
from runnable.datastore import BaseRunLogStore
from runnable.executor import BaseExecutor
from runnable.graph import Graph
from runnable.pickler import BasePickler
from runnable.secrets import BaseSecrets


class Context(BaseModel):
    executor: SerializeAsAny[BaseExecutor]
    run_log_store: SerializeAsAny[BaseRunLogStore]
    secrets_handler: SerializeAsAny[BaseSecrets]
    catalog_handler: SerializeAsAny[BaseCatalog]
    pickler: SerializeAsAny[BasePickler]
    progress: SerializeAsAny[Optional[Progress]] = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

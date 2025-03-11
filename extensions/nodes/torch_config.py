from enum import Enum

from pydantic import BaseModel, Field


class StartMethod(str, Enum):
    spawn = "spawn"
    fork = "fork"
    forkserver = "forkserver"


class TorchConfig(BaseModel):
    nnodes: str = Field(default="1:1")
    nproc_per_node: int = Field(default=1)

    rdzv_backend: str | None = Field(default="static")
    rdzv_endpoint: str | None = Field(default="")
    rdzv_id: str | None = Field(default="none")
    rdzv_conf: str | None = Field(default="")

    max_restarts: int | None = Field(default=None)
    monitor_interval: float | None = Field(default=0.1)
    start_method: str | None = Field(default=StartMethod.spawn)
    role: str | None = Field(default="default")
    log_dir: str | None = Field(default="torch_logs")
    redirects: str | None = Field(default="0")
    tee: str | None = Field(default="0")
    master_addr: str | None = Field(default="localhost")
    master_port: str | None = Field(default="29500")
    training_script: str = Field(default="dummy_training_script")
    training_script_args: str = Field(default="")

    # Optional fields
    local_ranks_filter: str | None = Field(default="0")
    node_rank: int | None = Field(default=0)
    standalone: bool | None = Field(default=None)
    module: bool | None = Field(default=False)
    no_python: bool | None = Field(default=False)
    run_path: bool | None = Field(default=False)

from pydantic import BaseModel, Field


class TorchConfig(BaseModel):
    nnodes: str = Field(default="1:1")
    nproc_per_node: int = Field(default=4)

    rdzv_backend: str = Field(default="static")
    rdzv_endpoint: str = Field(default="")
    rdzv_id: str | None = Field(default=None)
    rdzv_conf: str = Field(default="")

    max_restarts: int = Field(default=3)
    monitor_interval: float = Field(default=0.1)
    start_method: str = Field(default="spawn")
    role: str = Field(default="default_role")
    log_dir: str = Field(default="torch_logs")
    redirects: str = Field(default="1")
    tee: str = Field(default="1")
    master_addr: str = Field(default="localhost")
    master_port: str = Field(default="29500")
    training_script: str = Field(default="dummy_training_script")
    training_script_args: str = Field(default="")

    # Optional fields
    local_ranks_filter: str = Field(default="")
    node_rank: int = Field(default=0)
    local_addr: str | None = Field(default=None)
    logs_specs: str | None = Field(default=None)
    standalone: bool = Field(default=False)
    module: bool = Field(default=False)
    no_python: bool = Field(default=False)
    run_path: bool = Field(default=False)

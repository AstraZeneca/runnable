from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field


class StartMethod(str, Enum):
    spawn = "spawn"
    fork = "fork"
    forkserver = "forkserver"


# min_nodes: int
# max_nodes: int
# nproc_per_node: int

# logs_specs: Optional[LogsSpecs] = None
# run_id: str = ""
# role: str = "default_role"

# rdzv_endpoint: str = ""
# rdzv_backend: str = "etcd"
# rdzv_configs: dict[str, Any] = field(default_factory=dict)
# rdzv_timeout: int = -1

# max_restarts: int = 3
# monitor_interval: float = 0.1
# start_method: str = "spawn"
# log_line_prefix_template: Optional[str] = None
# metrics_cfg: dict[str, str] = field(default_factory=dict)
# local_addr: Optional[str] = None

## The idea is the following:
# Users can configure any of the options present in TorchConfig class.
# The LaunchConfig class will be created from torch config.
# The LogSpecs is sent as a parameter to the launch config.
# None as much as possible to get

## NO idea of standalone and how to send it


class InternalLogSpecs(BaseModel):
    log_dir: Optional[str] = Field(default="torch_logs")
    redirects: int | None = Field(default=None)
    tee: int | None = Field(default=None)
    local_ranks_filter: Optional[set[int]] = Field(default=None)

    model_config = ConfigDict(extra="ignore")


class TorchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nnodes: str = Field(default="1:1", exclude=True)
    nproc_per_node: int = Field(default=1)

    # will be used to create the log specs
    log_dir: Optional[str] = Field(default="torch_logs", exclude=True)
    redirects: int | None = Field(default=None, exclude=True)
    tee: int | None = Field(default=None, exclude=True)
    local_ranks_filter: Optional[set[int]] = Field(default=None, exclude=True)

    role: str | None = Field(default=None)
    # run_id would be the run_id of the context
    # and sent at the creation of the LaunchConfig

    rdzv_backend: str | None = Field(default="static")
    rdzv_endpoint: str | None = Field(default="")
    rdzv_configs: dict[str, Any] = Field(default_factory=dict)
    rdzv_timeout: int | None = Field(default=None)

    max_restarts: int | None = Field(default=None)
    monitor_interval: float | None = Field(default=None)
    start_method: str | None = Field(default=StartMethod.spawn)
    log_line_prefix_template: str | None = Field(default=None)
    local_addr: Optional[str] = None

    # https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py#L753
    # master_addr: str | None = Field(default="localhost")
    # master_port: str | None = Field(default="29500")
    # training_script: str = Field(default="dummy_training_script")
    # training_script_args: str = Field(default="")


class EasyTorchConfig(TorchConfig):
    model_config = ConfigDict(extra="ignore")

    # TODO: Validate min < max
    @computed_field  # type: ignore
    @property
    def min_nodes(self) -> int:
        return int(self.nnodes.split(":")[0])

    @computed_field  # type: ignore
    @property
    def max_nodes(self) -> int:
        return int(self.nnodes.split(":")[1])

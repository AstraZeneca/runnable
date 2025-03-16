from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field


class StartMethod(str, Enum):
    spawn = "spawn"
    fork = "fork"
    forkserver = "forkserver"


## The idea is the following:
# Users can configure any of the options present in TorchConfig class.
# The LaunchConfig class will be created from TorchConfig.
# The LogSpecs is sent as a parameter to the launch config.

## NO idea of standalone and how to send it


# The user sees this as part of the config of the node.
# It is kept as similar as possible to torchrun
class TorchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # excluded as LaunchConfig requires min and max nodes
    nnodes: str = Field(default="1:1", exclude=True, description="min:max")
    nproc_per_node: int = Field(default=1, description="Number of processes per node")

    # will be used to create the log specs
    # But they are excluded from dump as logs specs is a class for LaunchConfig
    # from_str("0") -> Std.NONE
    # from_str("1") -> Std.OUT
    # from_str("0:3,1:0,2:1,3:2") -> {0: Std.ALL, 1: Std.NONE, 2: Std.OUT, 3: Std.ERR}
    log_dir: Optional[str] = Field(default="torch_logs", exclude=True)
    redirects: str = Field(default="0", exclude=True)  # Std.NONE
    tee: str = Field(default="0", exclude=True)  # Std.NONE
    local_ranks_filter: Optional[set[int]] = Field(default=None, exclude=True)

    role: str | None = Field(default=None)

    # run_id would be the run_id of the context
    # and sent at the creation of the LaunchConfig

    # This section is about the communication between nodes/processes
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

from typing import List, Optional, Union

from pydantic import Field, field_validator

from runnable.datastore import StepAttempt
from runnable.defaults import TypeMapVariable
from runnable.tasks import BaseTaskType, TaskReturns


def run_torch_task(
    rank: int = 1,
    world_size: int = 1,
    entrypoint: str = "some function",
    catalog: Optional[dict[str, List[str]]] = None,
    task_returns: Optional[List[Union[str, TaskReturns]]] = None,
    secrets: Optional[list[str]] = None,
):
    # Entry point that creates a python job using simpler python types
    # and and executes them. The run_id for the job is set to be run_id_rank

    # In pseudocode, the following is done:
    # Create the catalog object
    # Create the secrets and other objects required for the Runnable job
    # Init the process group using:
    # https://github.com/pytorch/examples/blob/main/imagenet/main.py#L140
    # Execute the job, the job is expected to use the environmental variables
    # to identify the rank or can have them as variable in the signature.
    # Once the job is executed, we destroy the process group
    pass


class TorchTaskType(BaseTaskType):
    task_type: str = Field(default="torch", serialization_alias="command_type")
    command: str
    num_gpus: int = Field(default=1, description="Number of GPUs to use")
    num_nodes: int = Field(
        default=1,
        description="Number of nodes to use, currently we only support single node, multi gpu",
        le=1,
    )

    @field_validator("num_gpus")
    def check_if_cuda_is_available(self, num_gpus: int) -> int:
        return num_gpus

    def execute_command(
        self,
        map_variable: TypeMapVariable = None,
    ) -> StepAttempt:
        pass

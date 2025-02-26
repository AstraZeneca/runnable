from typing import List, Optional

from pydantic import Field, field_validator

from runnable import defaults
from runnable.datastore import StepAttempt
from runnable.defaults import TypeMapVariable
from runnable.tasks import BaseTaskType


def run_torch_task(
    rank: int = 1,
    world_size: int = 1,
    entrypoint: str = "some function",
    catalog: Optional[dict[str, List[str]]] = None,
    task_returns: Optional[List[str]] = None,
    secrets: Optional[list[str]] = None,
):
    # Entry point that creates a python job using simpler python types
    # and and executes them. The run_id for the job is set to be run_id_rank
    # Since the configuration file is passes as environmental variable,
    # The job will use the configuration file to get the required information.

    # In pseudocode, the following is done:
    # Create the catalog object
    # Create the secrets and other objects required for the PythonJob
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

    @field_validator("num_gpus")
    @classmethod
    def check_if_cuda_is_available(cls, num_gpus: int) -> int:
        # Import torch and check if cuda is available
        # validate if the number of gpus is less than or equal to available gpus
        return num_gpus

    def execute_command(
        self,
        map_variable: TypeMapVariable = None,
    ) -> StepAttempt:
        # We have to spawn here
        return StepAttempt(attempt_number=1, status=defaults.SUCCESS)

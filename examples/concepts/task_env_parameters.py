"""
An example pipeline of accessing initial parameters and passing parameters between tasks
using environment variables.

You can run this pipeline by:
    python examples/concepts/task_env_parameters.py

"""

import json
import os

from pydantic import BaseModel

from runnable import Pipeline, Task


class EggsModel(BaseModel):
    ham: str


class EverythingModel(BaseModel):
    spam: str
    eggs: EggsModel


def modify_initial():
    """
    Access initial parameters by the keys.
    """
    spam = os.environ["MAGNUS_PRM_spam"]
    eggs = EggsModel.model_validate_json(os.environ["MAGNUS_PRM_eggs"])
    print(spam)
    ">>> Hello"
    print(eggs)
    ">>> ham='Yes, please!!'"

    # modify parameters
    os.environ["MAGNUS_PRM_spam"] = "World"
    os.environ["MAGNUS_PRM_eggs"] = json.dumps(eggs.model_dump(by_alias=True))


def consume():
    """
    Access only a subset of the parameters.
    """
    # the value is set by the modify_initial function.
    # Use cast_as to type hint the return value.
    eggs = EggsModel.model_validate_json(os.environ["MAGNUS_PRM_eggs"])
    print(eggs)
    ">>> ham='No, Thank you!!'"

    os.environ["MAGNUS_PRM_eggs"] = json.dumps(EggsModel(ham="May be one more!!").model_dump_json(by_alias=True))


def main():
    modify = Task(
        name="Modify",
        command="examples.concepts.task_api_parameters.modify_initial",
    )

    consume = Task(
        name="Consume",
        command="examples.concepts.task_api_parameters.consume",
        terminate_with_success=True,
    )

    modify >> consume

    pipeline = Pipeline(
        steps=[modify, consume],
        start_at=modify,
        add_terminal_nodes=True,
    )
    pipeline.execute(parameters_file="examples/concepts/parameters.yaml")


if __name__ == "__main__":
    main()

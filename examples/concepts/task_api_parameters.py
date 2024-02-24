"""
An example pipeline of accessing initial parameters and passing parameters between tasks
using the python API.

You can run this pipeline by:
    python examples/concepts/task_api_parameters.py

"""

from pydantic import BaseModel

from runnable import Pipeline, Task, get_parameter, set_parameter


class EggsModel(BaseModel):
    ham: str


class EverythingModel(BaseModel):
    spam: str
    eggs: EggsModel


def modify_initial():
    """
    Access initial parameters by the keys.
    """
    spam = get_parameter("spam")
    eggs = get_parameter("eggs", cast_as=EggsModel)
    print(spam)
    ">>> Hello"
    print(eggs)
    ">>> ham='Yes, please!!'"

    # modify parameters
    set_parameter(spam="World", eggs=EggsModel(ham="No, Thank you!!"))


def consume():
    """
    Access only a subset of the parameters.
    """
    # the value is set by the modify_initial function.
    # Use cast_as to type hint the return value.
    eggs = get_parameter("eggs", cast_as=EggsModel)
    print(eggs)
    ">>> ham='No, Thank you!!'"

    set_parameter(eggs=EggsModel(ham="May be one more!!"))


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

"""
An example pipeline of accessing initial parameters and passing parameters between tasks.

You can run this pipeline by:
    python examples/concepts/task_native_parameters.py

"""

from pydantic import BaseModel, create_model


class EggsModel(BaseModel):
    ham: str


class EverythingModel(BaseModel):
    spam: str
    eggs: EggsModel


def modify_initial(spam: str, eggs: EggsModel):
    """
    Access initial parameters by the keys.
    Type annotation helps in casting to the right model type.
    """
    print(spam)
    ">>> Hello"
    print(eggs)
    ">>> ham='Yes, please!!'"

    # Return modified parameters
    # Use this pattern to create or modify parameters at the root level.
    return EverythingModel(spam="World", eggs=EggsModel(ham="No, Thank you!!"))


def consume(eggs: EggsModel):
    """
    Access only a subset of the parameters.
    """
    # the value is set by the modify_initial function.
    print(eggs)
    ">>> ham='No, Thank you!!'"

    # Magnus supports only pydantic models as return types.
    # You can modify a subset of the parameters by creating a dynamic pydantic model.
    # https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation

    # CAUTION: Returning "eggs" would result in a new parameter "ham" at the root level
    # as it looses the nested structure.
    return create_model(
        "DynamicModel",
        eggs=(EggsModel, EggsModel(ham="May be one more!!")),
    )()


def main():
    from magnus import Pipeline, Task

    modify = Task(
        name="Modify",
        command="examples.concepts.task_native_parameters.modify_initial",
    )

    consume = Task(
        name="Consume",
        command="examples.concepts.task_native_parameters.consume",
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

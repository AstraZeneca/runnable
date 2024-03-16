"""
The initial parameters defined in the parameters file are:
simple: 1
inner:
  x: 10
  y: "hello"

You can execute this pipeline by: python examples/parameters.py

"""

from pydantic import BaseModel


class InnerModel(BaseModel):
    """
    Captures the "inner" part of the parameters.
    The model definition can be as nested as you want.
    """

    x: int
    y: str


class NestedModel(BaseModel):  # (1)
    """
    Captures the whole parameter space of the application.
    """

    simple: int
    inner: InnerModel


def display(x: int, y: str):  # (2)
    """
    The parameter "simple" and "inner" can be accessed by name.
    runnable understands the parameter "inner" as a pydantic model from
    annotation and casts it as a pydantic model.
    """
    print(x)
    print(y)


class ObjectType:
    def __init__(self):
        self.salute = "hello"


def return_parameters(inner: InnerModel):
    """
    The parameter "simple" and "inner" can be accessed by name.
    You can redefine the parameters by returning a pydantic model.
    """
    x = 2
    y = "world!!"

    return x, y, ObjectType()


def display_object(obj: ObjectType):
    print(obj.salute)


"""
The below code is only to provide a full working example.

In the real world, you can "box runnable" in pipeline definition either in
python or yaml without cluttering your application code.
"""


def main():
    from runnable import Pipeline, PythonTask

    display_task = PythonTask(name="display", function=display)

    return_parameters_task = PythonTask(
        name="return_parameters",
        function=return_parameters,
        returns=["x", "y", "obj"],
    )

    display_object_task = PythonTask(
        name="display_object",
        function=display_object,
        terminate_with_success=True,
    )

    display_task >> return_parameters_task >> display_object_task

    pipeline = Pipeline(
        start_at=display_task,
        steps=[display_task, return_parameters_task, display_object_task],
        add_terminal_nodes=True,
    )

    run_log = pipeline.execute(parameters_file="examples/parameters_initial.yaml")
    params = run_log.parameters

    ## Reflects the changes done by "return_parameters" function call.
    assert params["x"].value == 2
    assert params["y"].value == "world!!"


if __name__ == "__main__":
    main()

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


def display(simple: int, inner: InnerModel):  # (2)
    """
    The parameter "simple" and "inner" can be accessed by name.
    runnable understands the parameter "inner" as a pydantic model from
    annotation and casts it as a pydantic model.
    """
    assert simple == 1
    assert inner.x == 10
    assert inner.y == "hello world!!"


class ObjectType:
    def __init__(self):
        self.salute = "hello"


def return_parameters():
    """
    The parameter "simple" and "inner" can be accessed by name.
    You can redefine the parameters by returning a pydantic model.
    """
    x = 2
    y = "hello Universe!!"

    return x, y, ObjectType()


def display_object(obj: ObjectType):
    print(obj.salute)


def main():
    from runnable import Pipeline, PythonTask, pickled

    display_task = PythonTask(name="display", function=display)

    return_parameters_task = PythonTask(
        name="return_parameters",
        function=return_parameters,
        returns=["x", "y", pickled("obj")],
    )

    display_object_task = PythonTask(
        name="display_object",
        function=display_object,
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[display_task, return_parameters_task, display_object_task],
        add_terminal_nodes=True,
    )

    _ = pipeline.execute(parameters_file="examples/parameters_initial.yaml")

    return pipeline


if __name__ == "__main__":
    main()

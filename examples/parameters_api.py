"""
The initial parameters defined in the parameters file are:
simple: 1
inner:
  x: 10
  y: "hello"

"""

from pydantic import BaseModel


class InnerModel(BaseModel):
    """
    Captures the "inner" part of the parameters.
    The model definition can be as nested as you want.
    """

    x: int
    y: str


class NestedModel(BaseModel):
    """
    Captures the whole parameter space of the application.
    """

    simple: int
    inner: InnerModel


def display(simple: int, inner: InnerModel):
    """
    The parameter "simple" and "inner" can be accessed by name.
    Magnus understands the parameter "inner" as a pydantic model
    from annotation and returns a pydantic model
    """
    print(simple)
    print(inner)


def set_and_get():
    """
    You can also use the python API for fine grained control if functional
    specification does not fit your needs.

    get_parameter can be used to either
        - return a specific parameter/model if a key is provided.
        - return the whole parameter space casted as a
            pydantic model or as a dictionary.

    set_parameter can be used to set a parameter/model.

    """
    from magnus import get_parameter, set_parameter

    # You can also get all the parameters as a pydantic model.
    all_parameters = get_parameter(cast_as=NestedModel)  # (1)
    print(all_parameters)
    ">>> # simple=1 inner=InnerModel(x=10, y='hello')"

    # get the parameter "inner" and cast it as InnerModel
    b = get_parameter(key="inner", cast_as=InnerModel)

    b.x = 100
    b.y = "world"

    # set the parameter "inner" to the new value
    set_parameter(inner=b)  # (2)


"""
The below code is only to provide a full working example.

In the real world, you can "box magnus" in pipeline definition
either in python or yaml without cluttering your application code.
"""


def main():
    from magnus import Pipeline, Task

    display = Task(name="display", command="examples.parameters.display")

    set_and_get = Task(
        name="set_and_get",
        command="examples.parameters.set_and_get",
        terminate_with_success=True,
    )

    display >> set_and_get

    pipeline = Pipeline(
        start_at=display,
        steps=[display, set_and_get],
        add_terminal_nodes=True,
    )

    pipeline.execute()


if __name__ == "__main__":
    main()

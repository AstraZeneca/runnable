from pydantic import BaseModel


class InnerModel(BaseModel):
    """
    A pydantic model representing a group of related parameters.
    """

    foo: int
    bar: str


class Parameter(BaseModel):
    """
    A pydantic model representing the parameters of the whole pipeline.
    """

    x: int
    y: InnerModel


def return_parameter() -> Parameter:
    """
    A example python task that does something interesting and returns
    a parameter to be used in downstream steps.

    The annotation of the return type of the function is not mandatory
    but it is a good practice.

    Returns:
        Parameter: The parameters that should be used in downstream steps.
    """
    # Return type of a function should be a pydantic model
    return Parameter(x=1, y=InnerModel(foo=10, bar="hello world"))


def display_parameter(x: int, y: InnerModel):
    """
    An example python task that does something interesting with
    input parameters.

    Annotating the arguments of the function is important for
    runnable to understand the type of parameters you want.

    Without annotations, runnable would inject a python dictionary.

    Input args can be a pydantic model or the individual attributes
    of the non-nested model
    """
    print(x)
    # >>> prints 1
    print(y)
    # >>> prints InnerModel(foo=10, bar="hello world")


"""
Without any framework, the "driver" code would be the
main function.
"""


def main():
    my_param = return_parameter()
    display_parameter(my_param.x, my_param.y)


if __name__ == "__main__":
    main()

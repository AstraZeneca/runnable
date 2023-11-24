import logging

from pydantic import BaseModel

# Magnus logging levels are different to your logging levels.
logger = logging.getLogger("application")
logger.setLevel(logging.DEBUG)

"""
A lot of design is based on the idea that you should be able to call these functions independent of magnus.
The below 2 functions are simple python functions that have no idea of where they would be run.
"""


class Parameter(BaseModel):
    x: int


def return_parameter() -> Parameter:
    # Return type of a function should be a pydantic model
    return Parameter(x=1)


def get_parameter(model: Parameter):
    # Input args can be a pydantic model or the individual attributes of the model.
    # You can also do get_parameter(x: int). magnus will provide that parameter for you.
    logger.info(f"I got a parameter: {model}")

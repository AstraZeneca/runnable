import logging

logger = logging.getLogger("application")
logger.setLevel(logging.INFO)


def return_parameter() -> dict:
    logger.info("In the app and returning something")
    return {"x": 1}


def get_parameter(x):
    print(x)

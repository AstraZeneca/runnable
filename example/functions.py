import logging

logger = logging.getLogger("application")
logger.setLevel(logging.DEBUG)


def return_parameter() -> dict:
    logger.info("#####################")
    return {"x": 1}


def get_parameter(x):
    print(x)

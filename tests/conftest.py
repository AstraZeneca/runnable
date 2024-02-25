from pytest import fixture


@fixture(scope="session", autouse=True)
def runnable_log():
    import logging

    logger = logging.getLogger("runnable")
    logger.setLevel(logging.WARNING)
    logger.propagate = True
    yield logger

from pytest import fixture


@fixture(scope="session", autouse=True)
def magnus_log():
    import logging

    logger = logging.getLogger("magnus")
    logger.setLevel(logging.WARNING)
    logger.propagate = True
    yield logger

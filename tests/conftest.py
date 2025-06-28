import sys
from pathlib import Path

from pytest import fixture

# Add the project root to the Python path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))


@fixture(scope="session", autouse=True)
def runnable_log():
    import logging

    logger = logging.getLogger("runnable")
    logger.setLevel(logging.WARNING)
    logger.propagate = True
    yield logger

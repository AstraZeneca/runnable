"""
You can execute this async pipeline by:

    python examples/async/async_sequential.py

This demonstrates a sequential async pipeline with multiple async tasks
that pass parameters between them.
"""

import asyncio

from examples.common.functions import async_hello, async_process
from runnable import AsyncPipeline, AsyncPythonTask


async def main():
    # First task produces a result
    hello_task = AsyncPythonTask(
        name="hello",
        function=async_hello,
        returns=["result"],
    )

    # Second task processes the result from the first task
    process_task = AsyncPythonTask(
        name="process",
        function=async_process,
        returns=["processed"],
    )

    # Chain them together in a sequential pipeline
    pipeline = AsyncPipeline(
        steps=[hello_task, process_task],
        name="async_sequential",
    )

    await pipeline.execute()
    return pipeline


if __name__ == "__main__":
    asyncio.run(main())

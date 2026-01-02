"""
You can execute this async pipeline by:

    python examples/async/async_tasks.py

This demonstrates basic async pipeline execution with AsyncPythonTask.
"""

import asyncio

from examples.common.functions import async_hello
from runnable import AsyncPipeline, AsyncPythonTask


async def main():
    # Create an async task which calls the async function "async_hello"
    hello_task = AsyncPythonTask(
        name="hello",
        function=async_hello,
        returns=["result"],
    )

    # The async pipeline has only one step.
    pipeline = AsyncPipeline(steps=[hello_task])

    await pipeline.execute()
    return pipeline


if __name__ == "__main__":
    asyncio.run(main())

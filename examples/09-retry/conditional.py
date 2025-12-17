import os
import random

from runnable import Conditional, Pipeline, PythonTask, Stub


def check_envvar():
    should_pass = os.environ.get("should_pass", "false").lower()

    if should_pass == "true":
        return
    raise ValueError("Environment variable check failed!")


def when_tails_function():
    print("when_tails_function called")


def toss_function():
    if "FIX_RANDOM_TOSS" in os.environ:
        # Use the fixed value for testing
        toss = os.environ["FIX_RANDOM_TOSS"]
        print(f"Using fixed toss result: {toss}")
        return toss

    # Simulate a coin toss
    toss = random.choice(["heads", "tails"])
    print(f"Toss result: {toss}")
    return toss


def main():
    when_heads_pipeline = PythonTask(  # [concept:branch-pipeline]
        name="check_envvar",
        function=check_envvar,
    ).as_pipeline()

    when_tails_pipeline = PythonTask(  # [concept:branch-pipeline]
        name="when_tails_task",
        function=when_tails_function,
    ).as_pipeline()

    conditional = Conditional(  # [concept:conditional]
        name="conditional",
        branches={
            "heads": when_heads_pipeline,
            "tails": when_tails_pipeline,
        },
        parameter="toss",
    )

    toss_task = PythonTask(  # [concept:task-with-returns]
        name="toss_task",
        function=toss_function,
        returns=["toss"],
    )

    continue_to = Stub(name="continue to")  # [concept:stub]
    pipeline = Pipeline(
        steps=[toss_task, conditional, continue_to]
    )  # [concept:pipeline]

    pipeline.execute()  # [concept:execution]

    return pipeline


if __name__ == "__main__":
    main()
    main()

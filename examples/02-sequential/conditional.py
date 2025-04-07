from runnable import Conditional, Pipeline, PythonTask


def when_heads_function():
    print("when_heads_function called")


def when_tails_function():
    print("when_tails_function called")


def toss_function():
    import random

    # Simulate a coin toss
    toss = random.choice(["heads", "tails"])
    print(f"Toss result: {toss}")
    return toss


def main():
    when_heads_pipeline = PythonTask(
        name="when_heads_task",
        function=when_heads_function,
    ).as_pipeline()

    when_tails_pipeline = PythonTask(
        name="when_tails_task",
        function=when_tails_function,
    ).as_pipeline()

    conditional = Conditional(
        name="conditional",
        branches={
            "heads": when_heads_pipeline,
            "tails": when_tails_pipeline,
        },
        parameter="toss",
    )

    toss_task = PythonTask(
        name="toss_task",
        function=toss_function,
        returns=["toss"],
    )
    pipeline = Pipeline(steps=[toss_task, conditional])

    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()

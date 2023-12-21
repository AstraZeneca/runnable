from magnus import Pipeline, Task


def simple_function():
    """
    A simple function that just prints "Hello World!".
    """
    print("Hello World!")


def main():
    simple_task = Task(
        name="simple",
        command="examples.concepts.simple_function",
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[simple_task],
        start_at=simple_task,
        add_terminal_nodes=True,
    )

    pipeline.execute()


if __name__ == "__main__":
    main()

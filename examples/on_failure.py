from magnus import Pipeline, Stub, Task


def main():
    step_1 = Task(name="step 1", command="exit 1", command_type="shell")
    step_2 = Stub(name="step 2")

    step_3 = Stub(name="step 3", terminate_with_success=True)

    step_1.on_failure = step_3.name

    step_1 >> step_2 >> step_3

    pipeline = Pipeline(
        steps=[step_1, step_2, step_3],
        start_at=step_1,
        add_terminal_nodes=True,
    )
    pipeline.execute()


if __name__ == "__main__":
    main()

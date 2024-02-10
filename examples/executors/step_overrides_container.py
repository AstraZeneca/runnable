"""
An example to demonstrate overriding global configuration for a step.

    step 1 runs in the docker image specified in the executor config and uses the environment
    value for key to be "value"

    step 2 overrides the config and executes the command in the configuration defined
    in overrides section of executor config.

    You can run this example using two steps:
        Generates yaml file:

        MAGNUS_CONFIGURATION_FILE=examples/executors/local-container-override.yaml \
        python examples/executors/step_overrides_container.py

        # Create the docker image with the pipeline magnus-pipeline.yaml as part of it.

        Execute the pipeline using the CLI:

        MAGNUS_VAR_default_docker_image=magnus:3.8 \
        MAGNUS_VAR_custom_docker_image=magnus:3.9 \
        magnus execute -f magnus-pipeline.yaml -c examples/executors/local-container-override.yaml

"""

from magnus import Pipeline, Task


def main():
    step1 = Task(
        name="step1",
        command="python --version && env | grep key",
        command_type="shell",
    )

    step2 = Task(
        name="step2",
        command="python --version && env | grep key",
        command_type="shell",
        terminate_with_success=True,
        overrides={"local-container": "custom_docker_image"},
    )

    step1 >> step2

    pipeline = Pipeline(
        start_at=step1,
        steps=[step1, step2],
        add_terminal_nodes=True,
    )

    pipeline.execute()


if __name__ == "__main__":
    main()

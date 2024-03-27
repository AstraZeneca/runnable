"""
An example to demonstrate overriding global configuration for a step.

    step 1 runs in the docker image specified in the executor config and uses the environment
    value for key to be "value"

    step 2 overrides the config and executes the command in the configuration defined
    in overrides section of executor config.

    You can run this example using two steps:
        Generates yaml file:

        runnable_CONFIGURATION_FILE=examples/executors/local-container-override.yaml \
        python examples/executors/step_overrides_container.py

        # Create the docker image with the pipeline runnable-pipeline.yaml as part of it.

        Execute the pipeline using the CLI:

        runnable_VAR_default_docker_image=runnable:3.8 \
        runnable_VAR_custom_docker_image=runnable:3.9 \
        runnable execute -f runnable-pipeline.yaml -c examples/executors/local-container-override.yaml

"""

from runnable import Pipeline, Task


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

    pipeline = Pipeline(
        steps=[step1, step2],
        add_terminal_nodes=True,
    )

    pipeline.execute()


if __name__ == "__main__":
    main()

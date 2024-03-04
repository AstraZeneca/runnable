"""
An example pipeline to demonstrate how to use the secrets manager.

Run this pipeline by:
    secret="secret_value" runnable_CONFIGURATION_FILE=examples/configs/secrets-env-default.yaml \
    python examples/secrets_env.py

"""

from runnable import get_secret


def show_secret():
    secret = get_secret("secret")

    assert secret == "secret_value"


def main():
    from runnable import Pipeline, Task

    show = Task(
        name="show secret",
        command="examples.secrets_env.show_secret",
        terminate_with_success=True,
    )

    pipeline = Pipeline(steps=[show], start_at=show, add_terminal_nodes=True)

    pipeline.execute()


if __name__ == "__main__":
    main()

"""
An example pipeline to demonstrate how to use the secrets manager.
"""

from magnus import get_secret


def show_secret():
    shell_variable = get_secret("shell_type")  # (1)
    key_value_type = get_secret("kv_style")

    assert shell_variable == "shell type secret"
    assert key_value_type == "value"


def main():
    from magnus import Pipeline, Task

    show = Task(
        name="show secret",
        command="examples.secrets.show_secret",
        terminate_with_success=True,
    )

    pipeline = Pipeline(steps=[show], start_at=show, add_terminal_nodes=True)

    pipeline.execute(configuration_file="examples/configs/dotenv.yaml", log_level="ERROR")


if __name__ == "__main__":
    main()

from runnable import Pipeline, TorchTask


def main():
    # If this step executes successfully, the pipeline will terminate with success
    hello_task = TorchTask(
        name="hello",
        args_to_torchrun={
            "nproc_per_node": 1,
            "nnodes": "1",
            "node_rank": "0",
        },
        script_to_call="examples/common/script.py",
    )

    # The pipeline has only one step.
    pipeline = Pipeline(steps=[hello_task])

    pipeline.execute(parameters_file="examples/common/initial_parameters.yaml")

    return pipeline


if __name__ == "__main__":
    main()

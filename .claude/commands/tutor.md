# Tutorial development

You are helping with tutorial of the Runnable framework.

# Working examples

There are plenty of examples in examples folder with the following structure.
They are layered as per increasing complexity. Focus only on .py files as yaml is being deprecated.

You can run any example as : ```uv run <python_file_name>```.
The resulting run is named by ```run_id```.

Any execution results in:

- a run log captured in .run_log_store against that run id
- a catalog folder against the run_id which has the files moved between tasks. It also captures the output from the
  function or script execution. In case of the notebook, the output notebook is stored.

├── 01-tasks - tells you how to run python functions, notebooks or scripts as pipelines
│   ├── notebook.py
│   ├── notebook.yaml
│   ├── python_task_as_pipeline.py
│   ├── python_tasks.py
│   ├── python_tasks.yaml
│   ├── scripts.py
│   ├── scripts.yaml
│   ├── stub.py
│   └── stub.yaml
├── 02-sequential - tell you how to stitch tasks into pipelines.
│   ├── conditional.py
│   ├── default_fail.py
│   ├── default_fail.yaml
│   ├── on_failure_fail.py
│   ├── on_failure_fail.yaml
│   ├── on_failure_succeed.py
│   ├── on_failure_succeed.yaml
│   ├── traversal.py
│   └── traversal.yaml
├── 03-parameters - shows the parameter flow between tasks and setting initial parameters.
                    Focus on how parameters are accessed and returned back. They are by names or argspace or kwargs.
│   ├── passing_parameters_notebook.py
│   ├── passing_parameters_notebook.yaml
│   ├── passing_parameters_python.py
│   ├── passing_parameters_python.yaml
│   ├── passing_parameters_shell.py
│   ├── passing_parameters_shell.yaml
│   ├── static_parameters_fail.py
│   ├── static_parameters_fail.yaml
│   ├── static_parameters_non_python.py
│   ├── static_parameters_non_python.yaml
│   ├── static_parameters_python.py
│   └── static_parameters_python.yaml
├── 04-catalog - Shows how to flow files between tasks. Focus on how get/put works and also how the user can chose not
                to store a copy in case if the file is too big.
│   ├── catalog_no_copy.py
│   ├── catalog_on_fail.py
│   ├── catalog_on_fail.yaml
│   ├── catalog_python.py
│   ├── catalog_python.yaml
│   └── catalog.py
├── 06-parallel - shows how to run parallel branches
│   ├── nesting.py
│   ├── nesting.yaml
│   ├── parallel_branch_fail.py
│   ├── parallel_branch_fail.yaml
│   ├── parallel.py
│   └── parallel.yaml
├── 07-map - shows how to run a branch looped over an iterable.
│   ├── custom_reducer.py
│   ├── custom_reducer.yaml
│   ├── map_fail.py
│   ├── map_fail.yaml
│   ├── map.py
│   └── map.yaml
├── 08-mocking - Useful for mocking/testing parts of the workflow.
│   ├── default.yaml
│   ├── mocked_map_parameters.yaml
│   ├── mocked-config-debug.yaml
│   ├── mocked-config-simple.yaml
│   ├── mocked-config-unittest.yaml
│   ├── mocked-config.yaml
│   └── patching.yaml
├── 11-jobs - shows how to run jobs.
│   ├── catalog_no_copy.py
│   ├── catalog.py
│   ├── emulate.yaml
│   ├── k8s-job.yaml
│   ├── local-container.yaml
│   ├── mini-k8s-job.yaml
│   ├── notebooks.py
│   ├── passing_parameters_python.py
│   ├── python_tasks.py
│   └── scripts.py

# Your role

Your role is to understand the current show case of capabilities and come up with missing examples.

You also need to help me with writing tutorials based on common ML workflows. There are some examples given in
examples/tutorials but it can be improved.

The same applies to examples provided in torch folder. They should be improved to make it easier to understand.

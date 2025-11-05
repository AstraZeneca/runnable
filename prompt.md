## Issue with parameters

In runnable, the parameters can be via environmental variables or via a yaml based file.
The environmental variables over ride the yaml based variables.

You can refer to the example in examples/03-parameters/static_parameters_python.py to understand it.
Make that clear in the docs


## Common patterns

In runnable, the order of concepts is something like this.

We have python functions, notebooks or scripts which are completely user code. We never ever touch them.
Lets call them as user_executables. User_Executables can:

- Accept and return parameters
- can ingest and process data
- Some times need to have access to secrets

Runnable makes these user_executables work in any environment by simply changing configuration.

The user_executables can be used, as per the intent of the user, in two scenarios.

One is the concept of a `job` while other being the concept of a `task` of a pipeline.

### Job pattern

look at the examples in examples/11-jobs/. Focus on python_tasks.py, notebooks.py or scripts.py to
understand how the user_executable is wrapped.

The jobs can take parameters defined in either yaml files or environmental variables like in the
example of examples/11-jobs/passing_parameters_python.py.

The jobs can also store processed data in local file system which we want to centrally catalog it like in the example
of catalog.py


### Pipeline pattern

The same user_executable can also be a single step in a workflow. To see this pattern, look at examples in
examples/01-tasks/. Focus on python_tasks.py, notebook.py or scripts.py or stub.py which is a mock.

These user_executables can be chained as a workflow as shown in examples/02-sequential, focus on traversal.py

The steps in the workflow can accept and return parameters just like their user_executables. To understand this,
look at the code in examples/03-parameters/. While the example static_parameters_* focuses on using
parameters from the start of the pipeline, the workflow in passing_parameters_* focus on parameter flow during the
execution. Especially focus on the returns of the PythonTask or NotebookTask or ShellTask to understand how the
names are assigned to return values of these executables which become as available parameters for the downstream
tasks of the workflow.

Similarly, the file also can be passed from one step to another using the catalog pattern. Look at the examples in
examples/04-catalog.


As you see there is a lot of overlap between the concepts in terms of parameters and catalog and user_executables.
Plan me a good way to document them to clearly show the similarities and purpose.


## Advanced patterns

Runnable allows to create complex patterns like

- parallel branches: example in 06-parallel
- map reduce patterns: example in 07-map. Note that the reducer can be customised
- -conditional branch: example in 02-sequential/conditional.py
- failure handling: example in 02-sequential/*fail* which can control the traversal.
- Infinite nesting: example in 08-parallel/nesting
- mocking: or stub based workflows example in 08-mocking/ . This requires no change in code, you can mock every step and also patch some code to  over-ride the user_Executable.

These are fairly advanced patterns. Refine the Advanced concepts section to reflect it.


## Example vs complete example

In the code snippets we are showing in getting started section, I see the value in showing only the relevant
bits of the code but it might be misleading to show only that as it is not perfectly executable.

Is there a better way to document them? Plan it and brainstorm with me.


## Run log

Runnable has a rich logging to make workflows reproducible.

For example, run any example from examples for example examples/04-catalog/catalog_python.py to understand the cataloging
logging or examples/03-parameters/passing_parameters_python.py you will see that a .catalog directory and .run_log_store
directory will be created with run logs and catalog stored.

The run_id is a unique identifier of the run and the logs are captured against it.

Run the above two examples like uv run script_name and analyse the output and the created files. Lets talk more about
that after you see the results.


## Comparison kedro

- Ensure that the return types and inputs are annotated in functions
- Runnable does not have or need int(x) in returns
- runnable cannot work with data that is not part of the catalog yet, so the first step which does a catalog get  \
  will fail. The alternate and recommended way is to either have them as part of the file structure which is less ideal \
  or have them in central storage like S3 and access it. Using s3 has an advantage as we can start to version it and \
  also we can use secrets to safely access them.
- The parameters which are used cannot be sent into runnable pipeline execute as ```parameters={"max_depth": 15}``` \
  They should be either defined by a parameters file in yaml or via environmental variables. Use defaults but override
  them before execution in the demonstration code.
- Highlight that the fact that the domain code can exist elsewhere and not part of the runnable wrappers
- If I am not wrong, kedro need not have data in such a heirarchial fashion, it is only a recommendation. Correct me \
  if I am wrong.


## Bugs

Not all return types are support by all jobs or tasks.

Python related jobs or tasks can return anything and accept anything.

Notebook jobs or tasks can only accept JSON friendly inputs (string, integer etc) but can return anything
Shell jobs or tasks can only accept JSON friend inputs and are exposed as environmental variables and can only return json friendly outputs from environmental variables

In superpowers/deploy anywhere - the kubernetes.yaml is specific to jobs.


In advanced patterns/parallel-execution - the infinite nesting is a mix and match between any of the complex pipelines, not just parallel

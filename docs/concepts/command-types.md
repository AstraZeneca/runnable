# Command types

## Python

By default, ```python``` is the command type. You can mention the python function that you want to invoke
using the ```command``` section.

For example, in the dag definition below, the command type is defaulted to python and magnus invokes
```my_module.my_function``` as part of the step.

```yaml
dag:
  steps:
    step1:
      command: my_module.my_function
    ...
```

Or via the python SDK:

```python
from magnus import Task

first = Task(name='step1', command='my_module.my_function')
```

The function arguments are dynamically introspected from the parameter space.

The return value of the function should always be a dictionary for parameters and are added as key-value pairs
into the parameter space. Non dictionary arguments are ignored with a warning.

More [examples](../../examples)

Any console output from the function is automatically uploaded to the catalog for future reference.


## Shell

You can have shell commands as part of magnus dag definition. The ```command``` provided in the config is
invoked as part of the step.

For example, in the dag definition below, ```step``` invokes the ```ls``` command as part of the pipeline.
You can use this ```command_type``` to have non-python executables as part of your pipeline.

```yaml
dag:
  steps:
    step1:
      command: ls
      command_type: shell
    ...
```

Or via the python SDK:

```python
from magnus import Task

first = Task(name='step1', command='ls', command_type='shell')
```


Please note that, magnus will be able to send in the existing parameters using environmental variables prefixed with
```MAGNUS_PRM_``` but would not be able to collect any return parameters. Similarly, the functionality of
secrets should be handled by the ```script``` and would not be done by magnus.

The cataloging functionality works as designed and can be used to sync data in and out the ```compute_data_folder```.

More [examples](../../examples)

## Python lambda expression

Using ```command_type: python-lambda```, you can provide a lambda expression as ```command```. For example:

```
lambda x : int(x) + 1
```

Or via the python SDK:

```python
from magnus import Task

first = Task(name='step1', command='lambda x : int(x) + 1', command_type='python-lambda')
```


is a valid lambda expression. Note that, you cannot have ```_```or ```__``` as part of your string. This is just a
security feature to
[avoid malicious code injections](https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html).

The parameters used as part of the lambda expression are introspected and provided dynamically from the parameter space.

This command type is designed to provide simpler ways to manipulate parameter space.

## Notebook

In magnus, you can execute Jupyter notebooks by ```command_type: notebook```. The ```command``` should be the path
to the notebook you want to execute.

---
!!! Note

    For ```command_type: notebook``` to work, you need to install optional packages by:

    pip install magnus[notebook]
---


Internally, we use [ploomber](https://ploomber.io/) for inspection and execution
of the notebook. Any ```parameters``` defined in the notebook would be introspected and dynamically provided at runtime
from the parameter space.

The path of the output of execution is obtained by post-fixing ```_out``` to the input notebook but can be configured by providing ```notebook_output_path```

```yaml
dag:
  steps:
    step1:
      command: notebooks/input.ipynb
      command_type: notebook
      notebook_output_path: notebooks/output.ipynb
    ...
```

Or via the python SDK:

```python
from magnus import Task

first = Task(name='step1', command='notebooks/input.ipynb', command_type='notebook',
        notebook_output_path='notebooks/output.ipynb')
```

Since the kernel used is the same as the execution environment via ploomber, anything that you can do via the python
function should be available via the notebook.

The output notebook is automatically uploaded to the catalog for future reference.

## Container

You can execute a container as part of the task too. Please note that this is different from [```local-container```
execution mode](../executor-implementations/local-container) as the container need not have magnus installed as part of
it.


---
!!! Note

    For ```command_type: container``` to work, you need to install optional packages by:

    pip install magnus[docker]
---

The complete configuration of the container is:

```yaml
image: str
context_path: str = "/opt/magnus"
command: str = ""  # Would be defaulted to the entrypoint of the container
data_folder: str = "data"  # Would be relative to the context_path
output_parameters_file: str = "parameters.json"  # would be relative to the context_path
secrets: List[str] = []
experiment_tracking_file: str = ""

```

### image
The name of the image that you want to execute as part of the pipeline. The image should be accessible to the docker
client on the machine, so either a local docker image or a authenticated docker registry will work.


### command
The command that you want to execute in the container as part of the pipeline. Empty string defaults to the CMD of the
image.

### context_path
Defaults to ```/opt/magnus```.

The base path where magnus would populate the catalog and parameters and also retrieve it back when the execution is
finished.

### data_folder
Defaults to ```data```.

The directory relative to the ```context_path``` where the data is synced in and out by the catalog service.

### output_parameters_file
Defaults to ```parameters.json```.

The JSON file containing the parameters that the container wants to return to the downstream steps.

### secrets
The list of secret key names that need to be exposed to the container.

### experiment_tracking_file
The JSON file containing the experiment tracking information that should be saved against the step.

## Extensions

You can extend and implement your ```command_types``` by extending the base class of the command type.

[Extensions are being actively developed and can be found here.](https://github.com/AstraZeneca/magnus-extensions)

```python
#Example implementations can be found in magnus/tasks.py
--8<-- "magnus/tasks.py:docs"

```

The custom extensions should be registered as part of the namespace: ```tasks``` for it to be
loaded.

```toml
# For example, as part of your pyproject.toml
[tool.poetry.plugins."tasks"]
"sql" = "YOUR_PACKAGE:SQLtaskType"

```

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

The path of the output of execution is obtained by post-fixing ```_out``` to the input notebook but can be configured
by ```command_config``` as shown below.

```yaml
dag:
  steps:
    step1:
      command: notebooks/input.ipynb
      command_type: notebook
      command_config:
        notebook_output_path: notebooks/output.ipynb
    ...
```

Or via the python SDK:

```python
from magnus import Task

first = Task(name='step1', command='notebooks/input.ipynb', command_type='notebook',
        command_config={'notebook_output_path': 'notebooks/output.ipynb'})
```

Since the kernel used is the same as the execution environment via ploomber, anything that you can do via the python
function should be available via the notebook.

The output notebook is automatically uploaded to the catalog for future reference.

## Extensions

You can extend and implement your ```command_types``` by extending the base class of the command type.

```python
#Example implementations can be found in magnus/tasks.py
class BaseTaskType:
    """
    A base task class which does the execution of command defined by the user
    """
    task_type = ''

    class Config(BaseModel):
        command: str

    def __init__(self, config: dict = None):
        config = config or {}
        self.config = self.Config(**config)

    @property
    def command(self):
        return self.config.command

    def _to_dict(self) -> dict:
        """
        Return a dictionary representation of the task
        """
        task = {}
        task['command'] = self.command
        task['config'] = self.config

        return task

    def _get_parameters(self, map_variable: dict = None, **kwargs) -> dict:
        """
        Return the parameters in scope for the execution

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        Returns:
            dict: The parameters dictionary in-scope for the task execution
        """
        return utils.get_user_set_parameters(remove=False)

    def execute_command(self, map_variable: dict = None, **kwargs):
        """
        The function to execute the command.

        And map_variable is sent in as an argument into the function.

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        Raises:
            NotImplementedError: Base class, not implemented
        """
        raise NotImplementedError()

    def _set_parameters(self, parameters: dict = None, **kwargs):
        """
        Set the parameters back to the environment variables.

        Args:
            parameters (dict, optional): The parameters to set back as env variables. Defaults to None.
        """
        # Nothing to do
        if not parameters:
            return

        if not isinstance(parameters, dict):
            msg = (
                f'call to function {self.command} returns of type: {type(parameters)}. '
                'Only dictionaries are supported as return values for functions as part part of magnus pipeline.')
            logger.warn(msg)
            return

        for key, value in parameters.items():
            logger.info(f'Setting User defined parameter {key} with value: {value}')
            os.environ[defaults.PARAMETER_PREFIX + key] = json.dumps(value)

```

The custom extensions should be registered as part of the namespace: ```tasks``` for it to be
loaded.

```toml
# For example, as part of your pyproject.toml
[tool.poetry.plugins."tasks"]
"sql" = "YOUR_PACKAGE:SQLtaskType"

```

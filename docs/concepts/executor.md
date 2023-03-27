# Executors

In magnus, executors essentially represent either the compute resource or the orchestration framework.

Conceptually, a executor can be one of two types:

- **Interactive executors**: In this mode, magnus does the work of executing the pipeline/function/notebook

```shell
magnus execute --file my-project.yaml --config-file config.yaml

magnus execute_notebook interesting.ipynb --config-file config.yaml

magnus execute_function my_module.my_function --config-file config.yaml
```

Magnus takes care of the execution of the pipeline/function or notebook in the compute you requested. Examples
of this executors or local, local container etc.

- **Orchestration executors**: In this mode, the dag definition is transpiled to your preferred orchestration language
of dag definition. To still achieve the capabilities of interactive executors, the orchestration language is
directed to call an internal method instead of your actual function.

Specifically, the orchestration is asked to call

```shell
magnus execute_single_node --file my-project.yaml --config-file config.yaml --step-name step-to-call
```

The branches of the original dag are also translated to the orchestrators language if its supported. If the
orchestration executors does not support a feature that magnus supports, you could still make it work by a mixed model.

Examples of orchestration modes are aws step functions, kubeflow job specification, argo job specification etc.

## Configuration

As with any system within magnus, configuration of an executor can be done by:

```yaml
executor:
  type:
  config:
```

### type

The type of mode provider you want. This should be one of the executors types already available.

Local executor is provided by default if nothing is provided.

### config

Any configuration parameters the execution provider accepts.

## Parameterized definition

As with any part of the magnus configuration, you can parameterize the configuration of executor to switch between
execution providers without changing the base definition.

Please follow the example provided [here](../dag/#parameterized_definition) for more information.

## Extensions

You can easily extend magnus to interpret the dag definition to a orchestration language of choice, if a default
implementation does not exist or you are not happy with the implementation.

[Extensions are being actively developed and can be found here.](https://github.com/AstraZeneca/magnus-extensions)

The ```BaseExecutor``` implementation is as follows:

```python
# The code can be found at magnus/executor.py
# The "_private" methods should not be touched without significant knowledge about the design

--8<-- "magnus/executor.py:docs"

```

The custom extensions should be registered as part of the namespace: ```executor``` for it to be
loaded.

```toml
# For example, as part of your pyproject.toml
[tool.poetry.plugins."executor"]
"kubeflow" = "YOUR_PACKAGE:Kubeflow"
```

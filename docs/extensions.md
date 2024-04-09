## General set up

runnable is built around the idea to decouple the pipeline definition and pipeline execution.

[All the concepts](concepts/the-big-picture.md/) are defined with this principle and therefore
are extendible as long as the API is satisfied.

We internally use [stevedore](https:/pypi.org/project/stevedore/) to manage extensions.
Our [pyproject.toml](https://github.com/AstraZeneca/runnable-core/blob/main/pyproject.toml) has
plugin space for all the concepts.

```toml
[tool.poetry.plugins."executor"]
"local" = "runnable.extensions.executor.local.implementation:LocalExecutor"
"local-container" = "runnable.extensions.executor.local_container.implementation:LocalContainerExecutor"
"argo" = "runnable.extensions.executor.argo.implementation:ArgoExecutor"

# Plugins for Catalog
[tool.poetry.plugins."catalog"]
"do-nothing" = "runnable.catalog:DoNothingCatalog"
"file-system" = "runnable.extensions.catalog.file_system.implementation:FileSystemCatalog"

# Plugins for Secrets
[tool.poetry.plugins."secrets"]
"do-nothing" = "runnable.secrets:DoNothingSecretManager"
"dotenv" = "runnable.extensions.secrets.dotenv.implementation:DotEnvSecrets"
"env-secrets-manager" = "runnable.extensions.secrets.env_secrets.implementation:EnvSecretsManager"

# Plugins for Run Log store
[tool.poetry.plugins."run_log_store"]
"buffered" = "runnable.datastore:BufferRunLogstore"
"file-system" = "runnable.extensions.run_log_store.file_system.implementation:FileSystemRunLogstore"
"chunked-fs" = "runnable.extensions.run_log_store.chunked_file_system.implementation:ChunkedFileSystemRunLogStore"

# Plugins for Experiment tracker
[tool.poetry.plugins."experiment_tracker"]
"do-nothing" = "runnable.experiment_tracker:DoNothingTracker"
"mlflow" = "runnable.extensions.experiment_tracker.mlflow.implementation:MLFlowExperimentTracker"

# Plugins for Pickler
[tool.poetry.plugins."pickler"]
"pickle" = "runnable.pickler:NativePickler"


# Plugins for Integration
[tool.poetry.plugins."integration"]
# Left empty for 3rd party integrations

# Plugins for Tasks
[tool.poetry.plugins."tasks"]
"python" = "runnable.tasks:PythonTaskType"
"shell" = "runnable.tasks:ShellTaskType"
"notebook" = "runnable.tasks:NotebookTaskType"


# Plugins for Nodes
[tool.poetry.plugins."nodes"]
"task" = "runnable.extensions.nodes:TaskNode"
"fail" = "runnable.extensions.nodes:FailNode"
"success" = "runnable.extensions.nodes:SuccessNode"
"parallel" = "runnable.extensions.nodes:ParallelNode"
"map" = "runnable.extensions.nodes:MapNode"
"stub" = "runnable.extensions.nodes:StubNode"
```


To submit extensions to this project (pretty please!!) submit a PR with plugin name
and implementation path inserted in *pyproject.toml*. We are happy to work with you to write
them, the complexity is mostly in having access to them.

To write extensions for your project and are not useful for wider audience, include the plugin
within your pyproject.toml or  [setuptools entry points](https://setuptools.pypa.io/en/latest/
pkg_resources.html#entry-points). During the execution of the pipeline,
runnable would automatically pick up the extension if it registered to the correct namespace.


The below section shows the base class implementation for all the concepts. All the base classes
are extended from pydantic BaseModel.


## Executor

Register to namespace: [tool.poetry.plugins."executor"]

Examples: [local](configurations/executors/local.md),
[local-container](configurations/executors/local-container.md),
[argo](configurations/executors/argo.md)

::: runnable.executor.BaseExecutor
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3


## Run Log

Register to namespace: [tool.poetry.plugins."run_log_store"]

Examples: [buffered](configurations/run-log.md/#buffered),
[file-system](configurations/run-log.md/#file-system),
 [chunked-fs](configurations/run-log.md/#chunked-fs)

::: runnable.datastore.BaseRunLogStore
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3

The ```RunLog``` is a nested pydantic model and is located in ```runnable.datastore.RunLog```.



## Catalog

Register to namespace: [tool.poetry.plugins."catalog"]

Example:
[do-nothing](configurations/catalog.md/#do-nothing),
 [file-system](configurations/catalog.md/#file-system)

::: runnable.catalog.BaseCatalog
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3


## Secrets

Register to namespace: [tool.poetry.plugins."secrets"]

Example:
[do-nothing](configurations/secrets.md/#do-nothing),
 [env-secrets-manager](configurations/secrets.md/#environment_secret_manager),
 [dotenv](configurations/secrets.md/#dotenv)

::: runnable.secrets.BaseSecrets
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3


## Experiment tracking

Register to namespace: [tool.poetry.plugins."experiment_tracker"]

Example:
[do-nothing](configurations/experiment-tracking.md), ```mlflow```

::: runnable.experiment_tracker.BaseExperimentTracker
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3

## Nodes

Register to namespace: [tool.poetry.plugins."nodes"]

Example:
[task](concepts/task.md),
[stub](concepts/stub.md),
[parallel](concepts/parallel.md),
[map](concepts/map.md)

::: runnable.nodes.BaseNode
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3



## Tasks

Register to namespace: [tool.poetry.plugins."tasks"]

Example:
[python](concepts/task.md/#python_functions),
[shell](concepts/task.md/#shell),
[notebook](concepts/task.md/#notebook)

::: runnable.tasks.BaseTaskType
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3

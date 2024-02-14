## General set up

Magnus is built around the idea to decouple the pipeline definition and pipeline execution.

[All the concepts](/concepts/the-big-picture/) are defined with this principle and therefore
are extendible as long as the API is satisfied.

We internally use [stevedore](https:/pypi.org/project/stevedore/) to manage extensions.
Our [pyproject.toml](https://github.com/AstraZeneca/magnus-core/blob/main/pyproject.toml) has
plugin space for all the concepts.

```toml
[tool.poetry.plugins."executor"]
"local" = "magnus.extensions.executor.local.implementation:LocalExecutor"
"local-container" = "magnus.extensions.executor.local_container.implementation:LocalContainerExecutor"
"argo" = "magnus.extensions.executor.argo.implementation:ArgoExecutor"

# Plugins for Catalog
[tool.poetry.plugins."catalog"]
"do-nothing" = "magnus.catalog:DoNothingCatalog"
"file-system" = "magnus.extensions.catalog.file_system.implementation:FileSystemCatalog"

# Plugins for Secrets
[tool.poetry.plugins."secrets"]
"do-nothing" = "magnus.secrets:DoNothingSecretManager"
"dotenv" = "magnus.extensions.secrets.dotenv.implementation:DotEnvSecrets"
"env-secrets-manager" = "magnus.extensions.secrets.env_secrets.implementation:EnvSecretsManager"

# Plugins for Run Log store
[tool.poetry.plugins."run_log_store"]
"buffered" = "magnus.datastore:BufferRunLogstore"
"file-system" = "magnus.extensions.run_log_store.file_system.implementation:FileSystemRunLogstore"
"chunked-fs" = "magnus.extensions.run_log_store.chunked_file_system.implementation:ChunkedFileSystemRunLogStore"

# Plugins for Experiment tracker
[tool.poetry.plugins."experiment_tracker"]
"do-nothing" = "magnus.experiment_tracker:DoNothingTracker"
"mlflow" = "magnus.extensions.experiment_tracker.mlflow.implementation:MLFlowExperimentTracker"

# Plugins for Pickler
[tool.poetry.plugins."pickler"]
"pickle" = "magnus.pickler:NativePickler"


# Plugins for Integration
[tool.poetry.plugins."integration"]
# Left empty for 3rd party integrations

# Plugins for Tasks
[tool.poetry.plugins."tasks"]
"python" = "magnus.tasks:PythonTaskType"
"shell" = "magnus.tasks:ShellTaskType"
"notebook" = "magnus.tasks:NotebookTaskType"


# Plugins for Nodes
[tool.poetry.plugins."nodes"]
"task" = "magnus.extensions.nodes:TaskNode"
"fail" = "magnus.extensions.nodes:FailNode"
"success" = "magnus.extensions.nodes:SuccessNode"
"parallel" = "magnus.extensions.nodes:ParallelNode"
"map" = "magnus.extensions.nodes:MapNode"
"stub" = "magnus.extensions.nodes:StubNode"
```


To submit extensions to this project (pretty please!!) submit a PR with plugin name
and implementation path inserted in *pyproject.toml*. We are happy to work with you to write
them, the complexity is mostly in having access to them.

To write extensions for your project and are not useful for wider audience, include the plugin
within your pyproject.toml or  [setuptools entry points](https://setuptools.pypa.io/en/latest/
pkg_resources.html#entry-points). During the execution of the pipeline,
magnus would automatically pick up the extension if it registered to the correct namespace.


The below section shows the base class implementation for all the concepts. All the base classes
are extended from pydantic BaseModel.


## Executor

Register to namespace: [tool.poetry.plugins."executor"]

Examples: [local](/configurations/executors/local),
[local-container](/configurations/executors/local-container),
[argo](/configurations/executors/argo)

::: magnus.executor.BaseExecutor
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3


## Run Log

Register to namespace: [tool.poetry.plugins."run_log_store"]

Examples: [buffered](/configurations/run-log/#buffered),
[file-system](/configurations/run-log/#file-system),
 [chunked-fs](/configurations/run-log/#chunked-fs)

::: magnus.datastore.BaseRunLogStore
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3

The ```RunLog``` is a nested pydantic model and is located in ```magnus.datastore.RunLog```.



## Catalog

Register to namespace: [tool.poetry.plugins."catalog"]

Example:
[do-nothing](/configurations/catalog/#do-nothing),
 [file-system](/configurations/catalog/#file-system)

::: magnus.catalog.BaseCatalog
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3


## Secrets

Register to namespace: [tool.poetry.plugins."secrets"]

Example:
[do-nothing](/configurations/secrets/#do-nothing),
 [env-secrets-manager](/configurations/secrets/#environment_secret_manager),
 [dotenv](/configurations/secrets/#dotenv)

::: magnus.secrets.BaseSecrets
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3


## Experiment tracking

Register to namespace: [tool.poetry.plugins."experiment_tracker"]

Example:
[do-nothing](/configurations/experiment-tracking), ```mlflow```

::: magnus.experiment_tracker.BaseExperimentTracker
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3

## Nodes

Register to namespace: [tool.poetry.plugins."nodes"]

Example:
[task](/concepts/task),
[stub](/concepts/stub),
[parallel](/concepts/parallel),
[map](/concepts/map)

::: magnus.nodes.BaseNode
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3



## Tasks

Register to namespace: [tool.poetry.plugins."tasks"]

Example:
[python](/concepts/task/#python_functions),
[shell](/concepts/task/#shell),
[notebook](/concepts/task/#notebook)

::: magnus.tasks.BaseTaskType
    options:
        show_root_heading: true
        show_source: true
        show_symbol_type_heading: true
        members: None
        heading_level: 3


## Roadmap

- AWS environments using Sagemaker pipelines or AWS step functions.
- HPC environment using SLURM executor.
- Database based Run log store.
- Better integrations with experiment tracking tools.
- Azure ML environments.

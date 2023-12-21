# Integration

Magnus at the core provides 5 services

- A computational execution plan or an Executor.
- A run log store to store metadata and run logs.
- A cataloging functionality to pass data between steps and audibility trace.
- A framework to handle secrets.
- A framework to interact with experiment tracking tools.

The executor plays the role of talking to other 4 service providers to process the graph, keep track of the status
of the run, pass data between steps and provide secrets.

Depending upon the stage of execution, the executor might do one of the two actions

- **traversing the graph**: For compute modes that just render instructions for other engines, the executor first
    traverses the graph to understand the plan but does not actually execute. For interactive modes, the executor
    traverses to set up the right environment for execution but defers the execution for later stage.

- **executing the node**: The executor is actually in the compute environment that it has to be and executes the task.

Magnus is designed to make the executor talk to the service providers at both these stages to understand the changes
needed for the config to make it happen via the ```BaseIntegration``` pattern.

```python
# Source code present at magnus/integration.py
--8<-- "magnus/integration.py:docs"
```

The custom extensions should be registered as part of the namespace: ```magnus.integration.BaseIntegration``` for it
to be loaded.

```toml
# For example, as part of your pyproject.toml
[tool.poetry.plugins."magnus.integration.BaseIntegration"]
# {executor.name}-{service}-{service.name}
"local-secrets-vault" = "YOUR_PACKAGE:LocalComputeSecretsVault"
```

All extensions need to be unique given a ```executor_type```, ```service_type``` and ```service_provider```.
Duplicate integrations will be raised as an exception.


## Example

Consider the example of S3 Run log store. For the execution engine of ```local```, the aws credentials file is available
on the local machine and we can store the run logs in the S3 bucket. But for the executor ```local-container```, the
aws credentials file has to be mounted in the container for the container to have access to S3.

This could be achieved by writing an integration pattern between S3 and ```local-container``` to do the same.

```python
class LocalContainerComputeS3Store(BaseIntegration):
    """
    Integration between local container and S3 run log store
    """
    executor_type = 'local-container'
    service_type = 'run-log-store'  # One of secret, catalog, datastore
    service_provider = 's3'  # The actual implementation of the service

    def configure_for_traversal(self, **kwargs):
        write_to = self.service.get_aws_credentials_file()
        self.executor.volumes[str(Path(write_to).resolve())] = {
            'bind': '/root/.aws/credentials',
            'mode': 'ro'
        }
```

We instruct the executor to mount the volumes containing the AWS credentials file as part of spinning the container to
make the credentials available to the running container.

# Integration

Magnus at the core provides 4 services

- A computational execution plan or an Executor.
- A run log store to store metadata and run logs.
- A cataloging functionality to pass data between steps and audibility trace.
- A framework to handle secrets.

The executor plays the role of talking to other 3 service providers to process the graph, keep track of the status
of the run, pass data between steps and provide secrets.

Depending upon the stage of execution, the executor might do one of the two actions

- **traversing the graph**: For compute modes that just render instructions for other engines, the executor first 
    traverses the graph to understand the plan but does not actually execute. For interactive modes, the executor 
    traverses to set up the right environment for execution but defers the execution for later stage.

- **executing the node**: The executor is actually in the compute environment that it has to be and executes the task.

Magnus is designed to make the executor talk to the service providers at both these stages to understand the changes
needed for the config to make it happen via the ```BaseIntegration``` pattern. 

```python
class BaseIntegration:
    """
    Base class for handling integration between Executor and one of Catalog, Secrets, RunLogStore.
    """
    mode_type = None
    service_type = None  # One of secret, catalog, datastore
    service_provider = None  # The actual implementation of the service

    def __init__(self, executor, integration_service):
        self.executor = executor
        self.service = integration_service

    def validate(self, **kwargs):
        """
        Raise an exception if the mode_type is not compatible with service provider.

        By default, it is considered as compatible.
        """

    def configure_for_traversal(self, **kwargs):
        """
        Do any changes needed to both executor and service provider during traversal of the graph.

        You are in the compute environment traversing the graph by this time.

        By default, no change is required.
        """

    def configure_for_execution(self,  **kwargs):
        """
        Do any changes needed to both executor and service provider during execution of a node.

        You are in the compute environment by this time.

        By default, no change is required.
        """
```

Consider the example of S3 Run log store. For the execution engine of ```local```, the aws credentials file is available
on the local machine and we can store the run logs in the S3 bucket. But for the mode ```local-container```, the 
aws credentials file has to be mounted in the container for the container to have access to S3. 

This could be achieved by writing an integration pattern between S3 and ```local-container``` to do the same.

```python
class LocalContainerComputeS3Store(BaseIntegration):
    """
    Integration between local container and S3 run log store
    """
    mode_type = 'local-container'
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
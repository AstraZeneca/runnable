Once a [pipeline is defined](../concepts/index.md), ```runnable``` can execute the pipeline in different environments
by changing a configuration. Neither the pipeline definition or the data science code needs to change at all.


## Concept

Consider the example:

```python linenums="1"
import os

def generate():
    ...
    # write some files, data.csv
    secret = os.environ["secret_key"]
    ...
    # return objects or simple python data types.
    return x, y

def consume(x, y):
    ...
    # read from data.csv
    # do some computation with x and y


# Stich the functions together
# This is the driver pattern.
x, y = generate()
consume(x, y)
```

To execute the functions, we need:

- Compute environment with defined resources (CPU, memory, GPU): configured by ```executor```.
- Mechanism to make variables, ```x``` and ```y```, available to functions: achieved by ```run_log_store```.
- Mechanism to recreate the file system structure for accessing ```data```:  achieved by ```catalog```.
- Populate secrets as environment variables: configured by ```secrets```.

<hr style="border:2px dotted orange">

By default, ```runnable``` uses:

- local compute to run the pipeline.
- local file system for storing the the run log.
- local file system for cataloging data flowing through the pipeline.
- wrapper around system environment variables for accessing secrets.

This can be over-ridden by ```configuration```. Configuration is now primarily done through the Python SDK, which provides more flexibility and type safety.

!!! warning "Legacy YAML Configuration"

    YAML-based configuration is being phased out in favor of Python-based configuration. The example below shows the legacy YAML format for reference only.

For example, a configuration that uses:

- argo workflows as execution engine
- mounted pvc for storing the run log
- mounted pvc for storing the catalog
- kubernetes secrets exposed to the container as secrets provider

Would be configured through the Python SDK configuration system (see individual configuration pages for Python examples).

For reference, the legacy YAML configuration would have looked like:

```yaml
executor:
  type: argo
  config:
    image: image_to_use
    persistent_volumes: # mount a pvc to every container as /mnt
      - name: runnable-volume
        mount_path: /mnt
    secrets_from_k8s: # expose postgres/connection string to container.
      - environment_variable: connection_string
        secret_name: postgres
        secret_key: connection_string

run_log_store:
  type: file-system
  config:
    log_folder: /mnt/run_log_store # /mnt is a pvc

catalog:
  type: file-system
  config:
   catalog_location: /mnt/catalog # /mnt is a pvc

secrets: # Kubernetes exposes secrets as environment variables
  type: env-secrets-manager
```

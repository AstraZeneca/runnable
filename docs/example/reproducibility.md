Magnus stores a lot of information about the current execution in ```run log```. The run log is internally used
for keeping track of the execution (status of different steps, parameters etc) but also has rich information
for reproducing the state at the time of pipeline execution.

The following are "invisibly" captured as part of the run log:

- Code: The ```git``` commit hash of the code used to run a pipeline is stored as part of the run log against
every step.
- Data hash: The data hash of the file passing through the catalog is stored as part of the run log. Since the
catalog itself is indexed against the execution id, it is easy to recreate the exact state of the data used
in the pipeline execution.
- Configuration: The configuration of the pipeline (dag definition, execution configuration) is also stored
as part of the run log.


=== "Example pipeline"

    ```python title="simple data passing pipeline"
    --8<-- "examples/catalog_api.py"
    ```

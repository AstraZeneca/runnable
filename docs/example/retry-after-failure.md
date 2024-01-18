Magnus allows you to debug and recover from a failure during the execution of pipeline. The pipeline can be
restarted in any suitable environment for debugging.


!!! example annotate

    A pipeline that is transpiled to argo workflows can be re-run on your local compute
    for debugging purposes. The only caveat is that, your local compute should have access to run log of the failed
    execution (1), generated catalog artifacts (2) from the the failed execution.

1. Access to the run log can be as simple as copy the json file to your local compute.
2. Generated catalog artifacts can be sourced from ```file-system``` which is your local folder.



Below is an example of retrying a pipeline that failed.


=== "Failed pipeline"

    !!! note

        You can run this pipeline on your local machine by

        ```magnus execute -f examples/retry-fail.yaml -c examples/configs/fs-catalog-run_log.yaml --run-id wrong-file-name```

        Note that we have specified the ```run_id``` to be something we can use later.
        The execution logs of the steps in the catalog will show the reason of the failure.

    ```yaml title="Pipeline that fails"
    --8<-- "examples/retry-fail.yaml"
    ```

    1. We make a data folder to store content.
    2. Puts a file in the data folder and catalogs it for downstream steps.
    3. It will fail here as there is no file called ```hello1.txt``` in the data folder.
    4. Get the file, ```hello.txt``` generated from previous steps into data folder.


=== "Fixed pipeline"

    !!! note

        You can run this pipeline on your local machine by

        ```magnus execute -f examples/retry-fail.yaml -c examples/configs/fs-catalog-run_log.yaml --use-cached wrong-file-name```

        Note that we have specified the run_id of the failed execution to be ```use-cached``` for the new execution.


    ```yaml title="Pipeline that restarts"
    --8<-- "examples/retry-fixed.yaml"
    ```

    1. We make a data folder to store content.
    2. Puts a file in the data folder and catalogs it for downstream steps.
    3. We fix the problem by pointing to the correct file in the data folder
    4. Get the file, ```hello.txt``` generated from previous steps into data folder.


=== "Run log"

## The problems we are trying to solve.

- feature:
    - Users should be able to record custom data to the logs.

- There is some data that need not be replicated but should be captured in logs.
    - DECISION: We would not play a role in sourcing the data and the user is expected to do it.
    - There should be a provision to identify the source data as a soft get.
    - This is true for large datasets. How true is that though?
    - This data could be sourced from a different location than the catalog.
        - This is a soft requirement and can be justified if it is not satisfied.
        - For simplicity lets assume that this is part of the catalog location.
    - This could be achieved by using a type of catalog that does not copy but records.


!!! note:
    Can this be simplified by using a "cached catalog"?

- cached catalog:
    - the run log will capture the catalog metadata.
    - The catalog will not be run id specific.

- Cached behavior: Given a previous run.
    - Users can refer to data generated from a previous run.
    - If the step by name is part of the pipeline and executed successfully, we want to skip execution of the step.
    - The logs should make it clear that it is a continuation of the previous run.
    - Its OK to assume that the run log is maintained in the same way.
    - Its OK to assume that the catalog is maintained in the same way.

    - Question about the recursive behavior.
        - What if the referenced run is a continuation of the previous run?
        - The desired behavior should be:
            - original run_id
            - continuation run_id
            - ...
        - The step will be skipped based on the status of penultimate run only.

    - What about many runs trying to write the same file all at once?
        - Do we keep versions of it or error out?
        - The desired behavior could be:
            - do the processing.
            - While saving, check for the existence of the file,
                - If it exists, write a versioned file and error out.
                - If it does not exist, continue successfully.

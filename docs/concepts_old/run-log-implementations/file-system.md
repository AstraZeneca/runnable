# File System Run Log store

This Run Log store stores the run logs on the file system as one JSON file.

The name of the json file is the ```run_id``` of the run.

When to use:

- When you want to compare logs between runs.
- During testing/debugging in local environments.


When not to use:

- This Run Log store is not compliant when the pipeline has parallel branches and enabled for a parallel runs.
    The results could be inconsistent and not reliable.
- Only Local and Local Container compute modes accept this as a Run Log Store.

## Configuration

The configuration is as follows:

```yaml
run_log:
  type: file-system
  config:
    log_folder:
```

### log_folder

The location of the folder where you want to write the run logs.

Defaults to .run_log_store

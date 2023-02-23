# File System Run Log store

This Run Log store stores the run logs on the file system as multiple thread safe files.

It creates a folder with ```run_id``` of the run and stores the contents of the run log in it.


When to use:

- When you want to compare logs between runs.
- During testing/debugging/developing in local environments.
- Especially useful when you have parallel processing of tasks.


When not to use:

- Only Local and Local Container compute modes accept this as a Run Log Store.

## Configuration

The configuration is as follows:

```yaml
run_log:
  type: chunked-fs
  config:
    log_folder:
```

### log_folder

The location of the folder where you want to write the run logs.

Defaults to .run_log_store

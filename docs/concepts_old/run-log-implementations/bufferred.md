# Buffered Run Log store

This Run Log store does not store the logs any where but in memory during the execution of the pipeline.

When to use:

- During development phase of the project and there is no need to compare outputs between runs.
- For a quick debug of a run.

When not to use:

- When you need to compare outputs between runs or experiments.
- Close to production runs or in production unless you do not want to store any run logs.
- Other than Local compute execution, no other compute modes accept this as a Run Log store.

## Configuration

Buffered Run Log store is the default if nothing was provided in the config.

The configuration is minimal and just needs:

```yaml
run_log:
  type: buffered
```

# Release history

## v0.4.1

- Bug fix with MLflow creeping in.
- Improved documentation.
- Improved CI process.
## v0.4.0 (2023-02-09)

- Added Experiment tracking interface
- Added Python SDK support
- Added configuration validation via Pydantic


## v0.3.11 (2022-07-12)

- Added a env-secrets-manager which gets its secrets from environment, Issue 58


## v0.3.10 (2022-06-30)

- Local container mode can accept a config parameter to allow it to run local system, Issue 52

## v0.3.9 (2022-06-29)

- Bug fix in catalog get function, Issue 54.
- Enhancement of sending log levels to different executing environments, Issue 53
- Map variable sent in as environment variable to non-shell tasks, Issue 51

## v0.3.8 (2022-06-29)

- Exposing secrets as environment variables if the user requested via secret_as_env in the config.
    The key of the secret_as_env should be the name of the environment variable while the value should be a secret name.

## v0.3.7 (2022-06-27)

- Fixing a bug with empty placeholders

## v0.3.6 (2022-06-25)

- Run ID is exposed to the user as interaction function or via environment variable MAGNUS_RUN_ID

## v0.3.5 (2022-05-24)

- Cataloging happens w.r.t to source location for flexibility

## v0.3.4 (2022-05-19)

- Bug fix with JSON decoding of parameters

## v0.3.3 (2022-04-23)

- Bug fix with executor interactions with run log store parameters.

## v0.3.2 (2022-04-23)

- Added the functionality of build_docker to the CLI.

## v0.3.1 (2022-04-23)

- Bug fix with executor interactions with run log store parameters.

## v0.3.0 (2022-03-30)

- Supporting execute_step which executes only a single node of the pipeline

## v0.2.4 (2022-03-28)

- Bug fix with optional git commit identifiers.


## v0.2.3 (2022-03-18)

- local container mode has a provision to send in environment variables to the container from the config.

## v0.2.2 (2022-02-23)

- Enabling configurations to have placeholders that individual nodes can use to over-ride.
- Provided API in the executor to resolve the effective config by using global and local config

## v0.2.1 (2022-02-22)

- Updated docs to clarify the extension capabilities of the CLI and nodes
- Removed demo-renderer argument parsing block as parameters come from parameters

## v0.2 (2022-02-22)

- Moved magnus CLI to click.
- magnus command group can be extended to add more commands by other packages.

Breaking changes:

- Contextualized config parameters for executors
- Parameters to be sent in via parameters file

## v0.1.2 (2022-02-08)

- Command config provided for all command types to pass in additional configuration.
- Moved to plugin based model using stevedore for nodes and tasks.
- Added support for notebooks as command types with optional install of papermill.

## v0.1.1 (2022-02-03)

- Bug fix with demo-renderer and as-is
- Moved to plugin based model using stevedore for executor, run log store, catalog, secrets and integrations

## v0.1.0 (2022-01-21)

- First release to open source.
- Compute: local, local-container, demo-renderer
- Run log store: local, buffered.
- Catalog: local, do-nothing.
- Secrets: dotenv, do-nothing.

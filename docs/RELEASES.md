# Release history


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

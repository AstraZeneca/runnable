# Release history

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

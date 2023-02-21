---
title: Welcome
sidebarDepth: 0
---


![logo](assets/logo1.png){ width="400" height="300" style="display: block; margin: 0 auto" }

---

**Magnus** is a *thin* layer of abstraction over the underlying infrastructure to enable data scientist and
machine learning engineers. It provides:

- A way to execute Jupyter notebooks/python functions in local or remote platforms.
- A framework to define complex pipelines via YAML or Python SDK.
- Robust and *automatic* logging to ensure maximum reproducibility of experiments.
- A framework to interact with secret managers ranging from environment variables to other vendors.
- Interactions with various experiment tracking tools.

## What does **thin** mean?

- We really have no say in what happens within your notebooks or python functions.
- We do not dictate how the infrastructure should be configured as long as it satisfies some *basic* criteria.
    - The underlying infrastructure should support container execution and an orchestration framework.
    - Some way to handle secrets either via environment variables or secrets manager.
    - A blob storage or some way to store your intermediate artifacts.
    - A database or blob storage to store logs.
- We have no opinion of how your structure your project.
- We do not creep into your CI/CD practices but it is your responsibility to provide the same environment where ever
the execution happens. This is usually via git, virtual environment manager and docker[^1].
- We transpile to the orchestration framework that is used by your teams to do the heavy lifting.

## What does it do?


![works](assets/work.png){ style="display: block; margin: 0 auto" }

### Shift Left

Magnus provides patterns typically used in production environments even in the development phase.

- Reduces the need for code refactoring during production phase of the project.
- Enables best practices and understanding of infrastructure patterns.
- Run the same code on your local machines or in production environments.


Please find the [github project here](https://github.com/AstraZeneca/magnus-core).
Supported extensions are available at: [magnus extensions](https://github.com/AstraZeneca/magnus-extensions)


Happy Experimenting!!


[^1]: We believe that for successful data science projects need good engineering practices and we enable data science
teams to follow them with ease.

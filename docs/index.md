---
title: Welcome
sidebarDepth: 0
---

<figure markdown>
  ![Image title](assets/logo1.png){ width="400" height="300"}
  <figcaption></figcaption>
</figure>

---

Magnus is a simplified workflow definition language that helps in:

- **Streamlined Design Process:** Magnus enables users to efficiently plan their pipelines with
[stubbed nodes](concepts/stub.md), along with offering support for various structures such as
[tasks](concepts/task.md), [parallel branches](concepts/parallel.md), and [loops or map branches](concepts/map.md)
in both [yaml](concepts/pipeline.md) or a [python SDK](sdk.md) for maximum flexibility.

- **Incremental Development:** Build your pipeline piece by piece with Magnus, which allows for the
implementation of tasks as [python functions](concepts/task.md/#python_functions),
[notebooks](concepts/task.md/#notebooks), or [shell scripts](concepts/task.md/#shell),
adapting to the developer's preferred tools and methods.

- **Robust Testing:** Ensure your pipeline performs as expected with the ability to test using sampled data. Magnus
also provides the capability to [mock and patch tasks](configurations/executors/mocked.md)
for thorough evaluation before full-scale deployment.

- **Seamless Deployment:** Transition from the development stage to production with ease.
Magnus simplifies the process by requiring
[only configuration changes](configurations/overview.md)
to adapt to different environments, including support for [argo workflows](configurations/executors/argo.md).

- **Efficient Debugging:** Quickly identify and resolve issues in pipeline execution with Magnus's local
debugging features. Retrieve data from failed tasks and [retry failures](concepts/run-log.md/#retrying_failures)
using your chosen debugging tools to maintain a smooth development experience.


Along with the developer friendly features, magnus also acts as an interface to production grade concepts
such as [data catalog](concepts/catalog.md), [reproducibility](concepts/run-log.md),
[experiment tracking](concepts/experiment-tracking.md)
and secure [access to secrets](concepts/secrets.md).

## Motivation

Successful data science projects require a varied set of skills from data scientists, ML engineers, and infrastructure
teams. Often, the roles and responsibilities of these personas are blurred leading to projects that are difficult to
maintain, test, reproduce or run at scale.

We build __**Magnus**__ to separate those concerns and create a clear boundary of the personas.

## Design principles

- [x] Code should not be mixed with implementation details of underlying platform.

**Example**: Data and parameters are often shared between different steps of the pipeline.
The platform implementation should not add additional code to make this happen.



- [x] Interactive development/debugging should be a first-class citizen.


**Example**: Data science teams thrive in environments with quick debug loop. Able to use their preferred tools
and iterate without constraints of the platform aids development/debugging.


- [x] Align the best practices even during development phase.

**Example**: All projects require secrets to access secure content. The concept of secret should be
available even during development phase and there should be no change in code when it is run in production set up.




## What does it do?

Magnus is a thin abstraction layer over the services typically provided by production grade infrastructures. Independent
of the provider, it exposes a consistent interface to those services, **this holds true even for the local environment**.

<figure markdown>
  ![Image title](assets/whatdo.png){ width="1200" height="800"}
  <figcaption></figcaption>
</figure>

The scope of magnus is intentionally limited to aid during the model development phase.
It does not boast of end to end development. The heavy lifting is always done by the providers.

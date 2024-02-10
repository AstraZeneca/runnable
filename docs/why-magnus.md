# Why Magnus

The scope of **magnus** is intentionally limited as an aid to author workflows for
production grade orchestrators like AWS Step functions or Argo Workflows. It is designed
to complement them, **NOT** replace them.

### Simplified flow of data

Production-grade orchestrators excel at managing a series of independent tasks, offering
straightforward implementation for task orchestration. Nonetheless, due to their general-purpose
design, orchestrating the flow of data—whether parameters or artifacts—can introduce complexity and
require careful handling.

Magnus simplifies this aspect by introducing an intuitive mechanism for data flow, thereby
streamlining data management. This approach allows the orchestrators to focus on their core
competency: allocating the necessary computational resources for task execution.

### Local first

In the context of the project's proof-of-concept (PoC) phase, the utilization of production-level
 orchestration systems is not optimal due to their complexity and potential constraints on rapid
 experimentation. Data scientists require an environment that aligns with their established workflows,
 which is most effectively achieved through the use of local development tools.

Magnus serves as an intermediary stage, simulating the production environment by offering local
versions of essential services—such as execution engines, data catalogs, secret management, and
experiment tracking—without necessitating intricate configuration. As the project transitions into the
production phase, these local stand-ins are replaced with their robust, production-grade counterparts.

### Remove refactoring

Transitioning from the proof of concept (PoC) phase to production often necessitates extensive code
refactoring, which presents significant challenges:

1. Refactoring demands considerable engineering resources to dissect the existing codebase and
reconstruct it in a form that is both maintainable and amenable to testing.

2. The engineered solutions that result from this process tend to exclude researchers from further
experimentation, thus impeding iterative research and development.


Magnus is engineered to minimize the need for such extensive refactoring when operationalizing
projects. It achieves this by allowing tasks to be defined as simple Python functions or Jupyter
notebooks. This means that the research-centric components of the code can remain unchanged, avoiding
the need for immediate refactoring and allowing for the postponement of these efforts until they
become necessary for the long-term maintenance of the product.

### Decouple implementations

In the rapidly evolving realm of technology, the solutions and tools selected today can
quickly become the technical debt of tomorrow. Magnus addresses this inevitability by
abstracting the implementation details from the underlying concepts. This decoupling
enables a seamless transition to new technical architectures, reducing the process to a
mere modification of configuration settings. Thus, Magnus facilitates adaptability
in the face of changing technological landscapes, ensuring that updates or replacements
of the technical stack can be implemented with minimal disruption.

### Non intrusive implementation

A lof of design aspect of magnus is to let the task definitions, python functions or notebooks,
remain agnostic of the orchestration process. Most often, the task orchestration can be
achieved by writing native "driver" functions. This allows the implementation to be completely
within the control of data scientists.

Most often, it should be possible to remove magnus from the tech stack if necessary.

<hr style="border:2px dotted orange">

## Alternatives

[Kedro](https://github.com/kedro-org/kedro) and [metaflow](https://metaflow.org/) are also
based on similar ideas and have established presence in this field. We took a lot of
inspiration from these excellent projects when writing magnus.

!!! note "Caveat"

    The scope of magnus is limited in comparison to metaflow. The below points are on
    the design philosophy rather that implementation specifics.

    The highlighted differences are subjective opinions and should be taken as preferences
    rather than criticisms.




### Infrastructure

Metaflow stipulates [infrastructure prerequisites](https://docs.metaflow.org/getting-started/infrastructure) that are established and validated across numerous scenarios.

In contrast, Magnus empowers engineering teams to define infrastructure specifications through a configuration file tailored to the stack they maintain. This versatility enables specialized teams to leverage their domain expertise, thereby enhancing the project's overall efficacy.

As magnus is mostly responsible for translating workflows to infrastructure patterns, it can
adapt to different environments.

### Project structure

Kedro and metaflow come with their own predefined project structures, which might be
appealing to some users while others might find them restrictive.

Magnus, on the other hand, offers a more flexible approach. It doesn't impose a specific
structure on your project. Whether you're working with Python functions, Jupyter notebooks,
or shell scripts, Magnus allows you to organize your work as you see fit. Even the location
of the data folder can be tailored for each step, avoiding a one-size-fits-all design and
providing the freedom to structure your project in a way that suits your preferences and
requirements.


### Notebook support

Both metaflow and kedro do not support notebooks as tasks. Notebooks are great during the iterative
phase of the project allowing for interactive development.

Magnus supports notebooks as tasks and has the ability to pass data/parameters between them
to allow orchestrating notebooks.

### Testing pipelines

Magnus supports patching tasks and mocking tasks to test the end to end execution of the
pipeline. It is not clear on how to achieve the same in kedro or metaflow.

### Learning curve

Magnus allows tasks to stand on their own, separate from the orchestration system. Explaining and
understanding these tasks is made easy through the use of simple "driver" functions. This approach
makes it easier for anyone working on the project to get up to speed and maintain it, as the
orchestration part of Magnus remains distinct and straightforward.

In contrast, learning to use Kedro and Metaflow can take more time because they have their own
specific ways of structuring projects and code that users need to learn.

### Language support

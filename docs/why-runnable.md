# Why runnable

**runnable** allows the data scientists/engineers to hook into production stack without
explicit knowledge of them. It offers a simpler abstraction of the concepts found in
production stack thereby aligning to the production standards even during development.

**runnable** is not a end to end deployment platform but limited to be an aid during
the developement phase without modifying the production stack or application code.

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Easy to pickup, its mostly your code__

    ---

    Adding **runnable** to your application is as simple as adding 2 files to your
    application without changing your application code.

    [:octicons-arrow-right-24: Getting started](#)

-   :fontawesome-brands-markdown:{ .lg .middle } __It's just Markdown__

    ---

    Focus on your content and generate a responsive and searchable static site

    [:octicons-arrow-right-24: Reference](#)

-   :material-format-font:{ .lg .middle } __Made to measure__

    ---

    Change the colors, fonts, language, icons, logo and more with a few lines

    [:octicons-arrow-right-24: Customization](#)

-   :material-scale-balance:{ .lg .middle } __Open Source, MIT__

    ---

    Material for MkDocs is licensed under MIT and available on [GitHub]

    [:octicons-arrow-right-24: License](#)

</div>

#### Simplified flow of data

Passing parameters or data artifacts between steps in airflow, argo or aws step step functions
is not trivial and requires code to be structured in a specific way. **runnable** handles
that for you in a fashion similar to code written without any orchestration detail.

#### Local first

Production stacks are ideal for the end state of the application. They are painful to use
during proof-of-concept (PoC) phase. **runnable** bridges the local setup to production stack
by adding just one file.


#### Bring your own stack

**runnable** can be used to bridge your application code into most commonly used production
stacks. It does not impose a change to either your production stack or application code.


#### Reduce refactoring

Transitioning from the proof of concept (PoC) phase to production often necessitates extensive code
refactoring, which presents significant challenges:

1. Refactoring demands considerable engineering resources to dissect the existing codebase and
reconstruct it in a form that is both maintainable and amenable to testing.

2. The engineered solutions that result from this process tend to exclude researchers from further
experimentation, thus impeding iterative research and development.


runnable is engineered to minimize the need for such extensive refactoring when operationalizing
projects. It achieves this by allowing tasks to be defined as [simple Python functions](concepts/task.md/#python_functions)
or [Jupyter notebooks](concepts/task.md/#notebook). This means that the research-centric components of the code
can remain unchanged, avoiding
the need for immediate refactoring and allowing for the postponement of these efforts until they
become necessary for the long-term maintenance of the product.

### Decouple implementations

In the rapidly evolving realm of technology, the solutions and tools selected today can
quickly become the technical debt of tomorrow. runnable addresses this inevitability by
abstracting the implementation details from the underlying concepts. This decoupling
enables a seamless transition to new technical architectures, reducing the process to a
mere modification of configuration settings. Thus, runnable facilitates adaptability
in the face of changing technological landscapes, ensuring that updates or replacements
of the technical stack can be implemented with minimal disruption.

### Non intrusive implementation

A lof of design aspect of runnable is to let the task definitions, python functions or notebooks,
remain agnostic of the orchestration process. Most often, the task orchestration can be
achieved by writing native "driver" functions. This allows the implementation to be completely
within the control of data scientists.

Most often, it should be possible to remove runnable from the tech stack if necessary.

<hr style="border:2px dotted orange">

## Alternatives

[Kedro](https://github.com/kedro-org/kedro) and [metaflow](https://metaflow.org/) are also
based on similar ideas and have established presence in this field. We took a lot of
inspiration from these excellent projects when writing runnable.

!!! note "Caveat"

    The scope of runnable is limited in comparison to metaflow. The below points are on
    the design philosophy rather that implementation specifics.

    The highlighted differences are subjective opinions and should be taken as preferences
    rather than criticisms.




### Infrastructure

Metaflow stipulates [infrastructure prerequisites](https://docs.metaflow.org/getting-started/infrastructure) that are established and validated across numerous scenarios.

In contrast, runnable empowers engineering teams to define infrastructure specifications through a configuration file tailored to the stack they maintain. This versatility enables specialized teams to leverage their domain expertise, thereby enhancing the project's overall efficacy.

As runnable is mostly responsible for translating workflows to infrastructure patterns, it can
adapt to different environments.

### Project structure

Kedro and metaflow come with their own predefined project structures, which might be
appealing to some users while others might find them restrictive.

runnable, on the other hand, offers a more flexible approach. It doesn't impose a specific
structure on your project. Whether you're working with Python functions, Jupyter notebooks,
or shell scripts, runnable allows you to organize your work as you see fit. Even the location
of the data folder can be tailored for each step, avoiding a one-size-fits-all design and
providing the freedom to structure your project in a way that suits your preferences and
requirements.


### Notebook support

Both metaflow and kedro do not support notebooks as tasks. Notebooks are great during the iterative
phase of the project allowing for interactive development.

runnable supports notebooks as tasks and has the ability to pass data/parameters between them
to allow orchestrating notebooks.

### Testing pipelines

runnable supports patching and mocking tasks to test the end to end execution of the
pipeline. It is not clear on how to achieve the same in kedro or metaflow.

### Learning curve

runnable allows tasks to stand on their own, separate from the orchestration system. Explaining and
understanding these tasks is made easy through the use of simple "driver" functions. This approach
makes it easier for anyone working on the project to get up to speed and maintain it, as the
orchestration part of runnable remains distinct and straightforward.

In contrast, learning to use Kedro and Metaflow can take more time because they have their own
specific ways of structuring projects and code that users need to learn.

### Language support

Kedro and metaflow only support python based pipeline definitions. It is possible to
run the non-python tasks as ```subprocesses``` in the pipeline tasks but the definition
is only possible using the python API.

runnable supports ```yaml``` based pipeline definitions and has ```shell``` tasks which
can be used for non-python tasks.

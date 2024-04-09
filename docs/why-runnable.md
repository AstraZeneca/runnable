# Why runnable

**runnable** allows the data scientists/engineers to hook into production stack without
knowledge of them. It offers a simpler abstraction of the concepts found in
production stack thereby aligning to the production standards even during development.

**runnable** is not a end to end deployment platform but limited to be an aid during
the development phase without modifying the production stack or application code.

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Easy to adopt, its mostly your code__

    ---

    Your application code remains as it is. Runnable exists outside of it.

    [:octicons-arrow-right-24: Getting started](concepts/the-big-picture.md)

-    :building_construction:{ .lg .middle } __Bring your infrastructure__

    ---

    Runnable can be adapted to your infrastructure stack instead of dictating it.

    [:octicons-arrow-right-24: Infrastructure](configurations/overview.md)

-   :memo:{ .lg .middle } __Reproducibility__

    ---

    Runnable tracks key information to reproduce the execution.

    [:octicons-arrow-right-24: Run Log](concepts/run-log.md)



-   :repeat:{ .lg .middle } __Retry failues__

    ---

    Debug any failure in your local development environment.

    [:octicons-arrow-right-24: Retry](#)

-   :microscope:{ .lg .middle } __Testing__

    ---

    Unit test your code and pipelines.

    [:octicons-arrow-right-24: Test](#)



-   :broken_heart:{ .lg .middle } __Move on__

    ---

    Moving away from runnable is as simple as deleting relevant files.


</div>


## Alternatives

**runnable** as an SDK competes with

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

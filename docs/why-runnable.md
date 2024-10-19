Obviously, there are a lot of orchestration tools. A well maintained and curated [list is
available here](https://github.com/EthicalML/awesome-production-machine-learning/).

Broadly, they could be classed into ```native``` or ```meta``` orchestrators.

<figure markdown>
  ![Image title](assets/work_light.png#only-light){ width="600" height="300"}
  ![Image title](assets/work_dark.png#only-dark){ width="600" height="300"}
</figure>


### __native orchestrators__

- Focus on resource management, job scheduling, robustness and scalability.
- Have less features on domain (data engineering, data science) activities.
- Difficult to run locally.
- Not ideal for quick experimentation or research activities.

### __meta orchestrators__

- An abstraction over native orchestrators.
- Oriented towards domain (data engineering, data science) features.
- Easy to get started and run locally.
- Ideal for quick experimentation or research activities.

```runnable``` is a _meta_ orchestrator with simple API, geared towards data engineering, data science projects.
It works in conjunction with _native_ orchestrators and an alternative to [kedro](https://docs.kedro.org/en/stable/index.html)
or [metaflow](https://metaflow.org/), in the design philosophy.

```runnable``` could also function as an SDK for _native_ orchestrators as it always compiles pipeline definitions
to _native_ orchestrators.

<br>

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Easy to adopt, its mostly your code__

    ---

    Your application code remains as it is. Runnable exists outside of it.

    - No API's or decorators or any imposed structure.

    [:octicons-arrow-right-24: Getting started](concepts/index.md)

-    :building_construction:{ .lg .middle } __Bring your infrastructure__

    ---

    ```runnable``` is not a platform. It works with your platforms.

    - ```runnable``` composes pipeline definitions suited to your infrastructure.

    [:octicons-arrow-right-24: Infrastructure](configurations/overview.md)

-   :memo:{ .lg .middle } __Reproducibility__

    ---

    Runnable tracks key information to reproduce the execution. All this happens without
    any additional code.

    [:octicons-arrow-right-24: Run Log](concepts/run-log.md)



-   :repeat:{ .lg .middle } __Retry failues__

    ---

    Debug any failure in your local development environment.

    [:octicons-arrow-right-24: Retry](#)

-   :microscope:{ .lg .middle } __Testing__

    ---

    Unit test your code and pipelines.

    - mock/patch the steps of the pipeline
    - test your functions as you normally do.

    [:octicons-arrow-right-24: Test](#)



-   :broken_heart:{ .lg .middle } __Move on__

    ---

    Moving away from runnable is as simple as deleting relevant files.

    - Your application code remains as it is.


</div>

<hr style="border:2px dotted orange">

## Comparisons

--8<-- "examples/comparisons/README.md"

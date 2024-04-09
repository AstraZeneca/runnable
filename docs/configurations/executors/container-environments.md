## Pipeline definition

Executing pipelines in containers needs a ```yaml``` based definition of the pipeline which is
referred during the [task execution](../../concepts/executor.md/#step_execution).


Any execution of the pipeline [defined by SDK](../../sdk.md) generates the pipeline
definition in```yaml``` format for all executors apart from the [```local``` executor](local.md).


Follow the below steps to execute the pipeline defined by SDK.


<div class="annotate" markdown>

1. Execute the pipeline by running the python script as you would normally do to generate
```yaml``` based definition.
2. Optionally (but highly recommended) version your code using git.
2. Build the docker image with the ```yaml``` file-based definition as part of the image. We recommend
tagging the docker image with the short git sha to uniquely identify the docker image (1).
3. Define a [variable to temporarily hold](https://docs.python.org/3/library/string.html#template-strings) the docker image name in the
pipeline definition, if the docker image name is not known.
4. Execute the pipeline using the [runnable CLI](../../usage.md/#usage).

</div>

1. Avoid using generic tags such as [```latest```](https://docs.docker.com/develop/dev-best-practices/).

## Dynamic name of the image


All containerized executors have a circular dependency problem.

- The docker image tag is only known after the creation of the image with the ```yaml``` based definition.
- But the ```yaml``` based definition needs the docker image tag as part of the definition.



!!! warning inline end

    Not providing the required environment variable will raise an exception.

To resolve this, runnable supports ```variables``` in the configuration of executors, both global and in step
overrides. Variables should follow the
[python template strings](https://docs.python.org/3/library/string.html#template-strings)
syntax and are replaced with environment variable prefixed by ```runnable_VAR_<identifier>```.

Concretely, ```$identifier``` is replaced by ```runnable_VAR_<identifier>```.


## Dockerfile

runnable should be installed in the docker image and available in the path. An example dockerfile is provided
below.

!!! note inline end "non-native orchestration"

    Having runnable to be part of the docker image adds additional dependencies for python to be present in the docker
    image. In that sense, runnable is technically non-native container orchestration tool.

    Facilitating native container orchestration, without runnable as part of the docker image, results in a complicated
    specification of files/parameters/experiment tracking losing the value of native interfaces to these essential
    orchestration concepts.

    With the improvements in python packaging ecosystem, it should be possible to distribute runnable as a
    self-contained binary and reducing the dependency on the docker image.

#### TODO: Change this to a proper example.
```dockerfile linenums="1"
--8<-- "examples/Dockerfile"
```

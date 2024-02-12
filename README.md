

# Hello from magnus


<p align="center">
  <img src="https://github.com/AstraZeneca/magnus-core/blob/main/assets/logo-readme.png?raw=true" alt="Logo"/>
</p>
---

<p align="center">
<a href="https://pypi.org/project/magnus/"><img alt="python:" src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg"></a>
<a href="https://pypi.org/project/magnus/"><img alt="Pypi" src="https://badge.fury.io/py/magnus.svg"></a>
<a href="https://github.com/AstraZeneca/magnus-core/blob/main/LICENSE"><img alt"License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/python/mypy"><img alt="MyPy Checked" src="https://www.mypy-lang.org/static/mypy_badge.svg"></a>
<a href="https://github.com/AstraZeneca/magnus-core/actions/workflows/release.yaml"><img alt="Tests:" src="https://github.com/AstraZeneca/magnus-core/actions/workflows/release.yaml/badge.svg">
<a href="https://github.com/AstraZeneca/magnus-core/actions/workflows/docs.yaml"><img alt="Docs:" src="https://github.com/AstraZeneca/magnus-core/actions/workflows/docs.yaml/badge.svg">
</p>
---

Magnus is a simplified workflow definition language that helps in:

- **Streamlined Design Process:** Magnus enables users to efficiently plan their pipelines with
[stubbed nodes](https://astrazeneca.github.io/magnus-core/concepts/stub), along with offering support for various structures such as
[tasks](https://astrazeneca.github.io/magnus-core/concepts/task), [parallel branches](https://astrazeneca.github.io/magnus-core/concepts/parallel), and [loops or map branches](https://astrazeneca.github.io/magnus-core/concepts/map)
in both [yaml](https://astrazeneca.github.io/magnus-core/concepts/pipeline) or a [python SDK](https://astrazeneca.github.io/magnus-core/sdk) for maximum flexibility.

- **Incremental Development:** Build your pipeline piece by piece with Magnus, which allows for the
implementation of tasks as [python functions](https://astrazeneca.github.io/magnus-core/concepts/task/#python_functions),
[notebooks](https://astrazeneca.github.io/magnus-core/concepts/task/#notebooks), or [shell scripts](https://astrazeneca.github.io/magnus-core/concepts/task/#shell),
adapting to the developer's preferred tools and methods.

- **Robust Testing:** Ensure your pipeline performs as expected with the ability to test using sampled data. Magnus
also provides the capability to [mock and patch tasks](https://astrazeneca.github.io/magnus-core/configurations/executors/mocked)
for thorough evaluation before full-scale deployment.

- **Seamless Deployment:** Transition from the development stage to production with ease.
Magnus simplifies the process by requiring [only configuration changes](https://astrazeneca.github.io/magnus-core/configurations/overview)
to adapt to different environments, including support for [argo workflows](https://astrazeneca.github.io/magnus-core/configurations/executors/argo).

- **Efficient Debugging:** Quickly identify and resolve issues in pipeline execution with Magnus's local
debugging features. Retrieve data from failed tasks and [retry failures](https://astrazeneca.github.io/magnus-core/concepts/run-log/#retrying_failures)
using your chosen debugging tools to maintain a smooth development experience.

Along with the developer friendly features, magnus also acts as an interface to production grade concepts
such as [data catalog](https://astrazeneca.github.io/magnus-core/concepts/catalog), [reproducibility](https://astrazeneca.github.io/magnus-core/concepts/run-log),
[experiment tracking](https://astrazeneca.github.io/magnus-core/concepts/experiment-tracking)
and secure [access to secrets](https://astrazeneca.github.io/magnus-core/concepts/secrets).


## What does it do?


![works](assets/work.png)


## Documentation

[More details about the project and how to use it available here](https://astrazeneca.github.io/magnus-core/).


## Installation

The minimum python version that magnus supports is 3.8

```shell
pip install magnus
```

Please look at the [installation guide](https://astrazeneca.github.io/magnus-core/usage)
for more information.


## Example

Your application code. Use pydantic models as DTO.

Assumed to be present at ```functions.py```
```python
from pydantic import BaseModel

class InnerModel(BaseModel):
    """
    A pydantic model representing a group of related parameters.
    """

    foo: int
    bar: str


class Parameter(BaseModel):
    """
    A pydantic model representing the parameters of the whole pipeline.
    """

    x: int
    y: InnerModel


def return_parameter() -> Parameter:
    """
    The annotation of the return type of the function is not mandatory
    but it is a good practice.

    Returns:
        Parameter: The parameters that should be used in downstream steps.
    """
    # Return type of a function should be a pydantic model
    return Parameter(x=1, y=InnerModel(foo=10, bar="hello world"))


def display_parameter(x: int, y: InnerModel):
    """
    Annotating the arguments of the function is important for
    magnus to understand the type of parameters you want.

    Input args can be a pydantic model or the individual attributes.
    """
    print(x)
    # >>> prints 1
    print(y)
    # >>> prints InnerModel(foo=10, bar="hello world")
```

### Application code without magnus using driver functions.

The code is runnable without any orchestration framework.

```python
from functions import return_parameter, display_parameter

my_param = return_parameter()
display_parameter(my_param.x, my_param.y)
```

### Orchestration using magnus

<table>
<tr>
    <th>python SDK</th>
    <th>yaml</th>
</tr>
<tr>
<td valign="top"><p>

Example present at: ```examples/python-tasks.py```

Run it as: ```python examples/python-tasks.py```

```python
from magnus import Pipeline, Task

def main():
    step1 = Task(
        name="step1",
        command="examples.functions.return_parameter",
    )
    step2 = Task(
        name="step2",
        command="examples.functions.display_parameter",
        terminate_with_success=True,
    )

    step1 >> step2

    pipeline = Pipeline(
        start_at=step1,
        steps=[step1, step2],
        add_terminal_nodes=True,
    )

    pipeline.execute()


if __name__ == "__main__":
    main()
```

</p></td>

<td valign="top"><p>

Example present at: ```examples/python-tasks.yaml```


Execute via the cli: ```magnus execute -f examples/python-tasks.yaml```

```yaml
dag:
  description: |
    This is a simple pipeline that does 3 steps in sequence.
    In this example:
      1. First step: returns a "parameter" x as a Pydantic model
      2. Second step: Consumes that parameter and prints it

    This pipeline demonstrates one way to pass small data from one step to another.

  start_at: step 1
  steps:
    step 1:
      type: task
      command_type: python # (2)
      command: examples.functions.return_parameter # (1)
      next: step 2
    step 2:
      type: task
      command_type: python
      command: examples.functions.display_parameter
      next: success
    success:
      type: success
    fail:
      type: fail
```

</p></td>

</tr>
</table>

### Transpile to argo workflows

No code change, just change the configuration.

```yaml
executor:
  type: "argo"
  config:
    image: magnus:demo
    persistent_volumes:
      - name: magnus-volume
        mount_path: /mnt

run_log_store:
  type: file-system
  config:
    log_folder: /mnt/run_log_store
```

More details can be found in [argo configuration](https://astrazeneca.github.io/magnus-core/configurations/executors/argo).

Execute the code as ```magnus execute -f examples/python-tasks.yaml -c examples/configs/argo-config.yam```

<details>
  <summary>Expand</summary>

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: magnus-dag-
  annotations: {}
  labels: {}
spec:
  activeDeadlineSeconds: 172800
  entrypoint: magnus-dag
  podGC:
    strategy: OnPodCompletion
  retryStrategy:
    limit: '0'
    retryPolicy: Always
    backoff:
      duration: '120'
      factor: 2
      maxDuration: '3600'
  serviceAccountName: default-editor
  templates:
    - name: magnus-dag
      failFast: true
      dag:
        tasks:
          - name: step-1-task-uvdp7h
            template: step-1-task-uvdp7h
            depends: ''
          - name: step-2-task-772vg3
            template: step-2-task-772vg3
            depends: step-1-task-uvdp7h.Succeeded
          - name: success-success-igzq2e
            template: success-success-igzq2e
            depends: step-2-task-772vg3.Succeeded
    - name: step-1-task-uvdp7h
      container:
        image: magnus:demo
        command:
          - magnus
          - execute_single_node
          - '{{workflow.parameters.run_id}}'
          - step%1
          - --log-level
          - WARNING
          - --file
          - examples/python-tasks.yaml
          - --config-file
          - examples/configs/argo-config.yaml
        volumeMounts:
          - name: executor-0
            mountPath: /mnt
        imagePullPolicy: ''
        resources:
          limits:
            memory: 1Gi
            cpu: 250m
          requests:
            memory: 1Gi
            cpu: 250m
    - name: step-2-task-772vg3
      container:
        image: magnus:demo
        command:
          - magnus
          - execute_single_node
          - '{{workflow.parameters.run_id}}'
          - step%2
          - --log-level
          - WARNING
          - --file
          - examples/python-tasks.yaml
          - --config-file
          - examples/configs/argo-config.yaml
        volumeMounts:
          - name: executor-0
            mountPath: /mnt
        imagePullPolicy: ''
        resources:
          limits:
            memory: 1Gi
            cpu: 250m
          requests:
            memory: 1Gi
            cpu: 250m
    - name: success-success-igzq2e
      container:
        image: magnus:demo
        command:
          - magnus
          - execute_single_node
          - '{{workflow.parameters.run_id}}'
          - success
          - --log-level
          - WARNING
          - --file
          - examples/python-tasks.yaml
          - --config-file
          - examples/configs/argo-config.yaml
        volumeMounts:
          - name: executor-0
            mountPath: /mnt
        imagePullPolicy: ''
        resources:
          limits:
            memory: 1Gi
            cpu: 250m
          requests:
            memory: 1Gi
            cpu: 250m
  templateDefaults:
    activeDeadlineSeconds: 7200
    timeout: 10800s
  arguments:
    parameters:
      - name: run_id
        value: '{{workflow.uid}}'
  volumes:
    - name: executor-0
      persistentVolumeClaim:
        claimName: magnus-volume

```

</details>

## Pipelines can be:

### Linear

A simple linear pipeline with tasks either a python functions, shell scripts or notebooks.

``` mermaid
%%{ init: { 'flowchart': { 'curve': 'linear' } } }%%
flowchart LR

    step1:::green
    step1([Step 1]) --> step2:::green
    step2([Step 2]) --> step3:::green
    step3([Step ...]) --> step4:::green
    step4([Step n]) --> suc([success]):::green

    classDef green stroke:#0f0

```

### Parallel branches

Execute branches in parallel.

```mermaid
flowchart TD

    getFeatures([Get Features]):::green
    trainStep(Train Models):::green
    ensembleModel([Ensemble Modelling]):::green
    inference([Run Inference]):::green
    success([Success]):::green

    prepareXG([Prepare for XGBoost]):::yellow
    trainXG([Train XGBoost]):::yellow
    successXG([XGBoost success]):::yellow
    prepareXG --> trainXG --> successXG

    trainRF([Train RF model]):::yellow
    successRF([RF Model success]):::yellow
    trainRF --> successRF


    getFeatures --> trainStep
    trainStep --> prepareXG
    trainStep --> trainRF
    successXG --> ensembleModel
    successRF --> ensembleModel
    ensembleModel --> inference
    inference --> success


    classDef yellow stroke:#FFFF00
    classDef green stroke:#0f0


```

### loops or map

Execute a branch over a parameter.

```mermaid
flowchart TD
chunkify([Chunk files]):::green
success([Success]):::green

subgraph one[Process Chunk]
    process_chunk1([Process Chunk]):::yellow
    success_chunk1([Success]):::yellow

    process_chunk1 --> success_chunk1
end

subgraph two[Process Chunk]
    process_chunk2([Process Chunk]):::yellow
    success_chunk2([Success]):::yellow

    process_chunk2 --> success_chunk2
end

subgraph three[Process Chunk]
    process_chunk3([Process Chunk]):::yellow
    success_chunk3([Success]):::yellow

    process_chunk3 --> success_chunk3
end

subgraph four[Process Chunk]
    process_chunk4([Process Chunk]):::yellow
    success_chunk4([Success]):::yellow

    process_chunk4 --> success_chunk4
end

subgraph five[Process Chunk]
    process_chunk5([Process Chunk]):::yellow
    success_chunk5([Success]):::yellow

    process_chunk5 --> success_chunk5
end



chunkify -- (stride=10, start_index=0)--> one --> success
chunkify -- (stride=10, start_index=10)--> two --> success
chunkify -- (stride=10, start_index=20)--> three --> success
chunkify -- (stride=10, start_index=30)--> four --> success
chunkify -- (stride=10, start_index=40)--> five --> success

classDef yellow stroke:#FFFF00
classDef green stroke:#0f0
```

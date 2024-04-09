



<p align="center">

                                                      ,////,
                                                      /// 6|
                                                      //  _|
                                                    _/_,-'
                                                _.-/'/   \   ,/;,
                                            ,-' /'  \_   \ / _/
                                            `\ /     _/\  ` /
                                               |     /,  `\_/
                                               |     \'
                                    /\_        /`      /\
                                  /' /_``--.__/\  `,. /  \
                                  |_/`  `-._     `\/  `\   `.
                                            `-.__/'     `\   |
                                                        `\  \
                                                          `\ \
                                                            \_\__
                                                              \___)

</p>
<hr style="border:2px dotted orange">

<p align="center">
<a href="https://pypi.org/project/runnable/"><img alt="python:" src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg"></a>
<a href="https://pypi.org/project/runnable/"><img alt="Pypi" src="https://badge.fury.io/py/runnable.svg"></a>
<a href="https://github.com/vijayvammi/runnable/blob/main/LICENSE"><img alt"License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/python/mypy"><img alt="MyPy Checked" src="https://www.mypy-lang.org/static/mypy_badge.svg"></a>
<a href="https://github.com/vijayvammi/runnable/actions/workflows/release.yaml"><img alt="Tests:" src="https://github.com/vijayvammi/runnable/actions/workflows/release.yaml/badge.svg">
<a href="https://github.com/vijayvammi/runnable/actions/workflows/docs.yaml"><img alt="Docs:" src="https://github.com/vijayvammi/runnable/actions/workflows/docs.yaml/badge.svg">
</p>
<hr style="border:2px dotted orange">

runnable is a simplified workflow definition language that helps in:

- **Streamlined Design Process:** runnable enables users to efficiently plan their pipelines with
[stubbed nodes](https://astrazeneca.github.io/runnable-core/concepts/stub), along with offering support for various structures such as
[tasks](https://astrazeneca.github.io/runnable-core/concepts/task), [parallel branches](https://astrazeneca.github.io/runnable-core/concepts/parallel), and [loops or map branches](https://astrazeneca.github.io/runnable-core/concepts/map)
in both [yaml](https://astrazeneca.github.io/runnable-core/concepts/pipeline) or a [python SDK](https://astrazeneca.github.io/runnable-core/sdk) for maximum flexibility.

- **Incremental Development:** Build your pipeline piece by piece with runnable, which allows for the
implementation of tasks as [python functions](https://astrazeneca.github.io/runnable-core/concepts/task/#python_functions),
[notebooks](https://astrazeneca.github.io/runnable-core/concepts/task/#notebooks), or [shell scripts](https://astrazeneca.github.io/runnable-core/concepts/task/#shell),
adapting to the developer's preferred tools and methods.

- **Robust Testing:** Ensure your pipeline performs as expected with the ability to test using sampled data. runnable
also provides the capability to [mock and patch tasks](https://astrazeneca.github.io/runnable-core/configurations/executors/mocked)
for thorough evaluation before full-scale deployment.

- **Seamless Deployment:** Transition from the development stage to production with ease.
runnable simplifies the process by requiring [only configuration changes](https://astrazeneca.github.io/runnable-core/configurations/overview)
to adapt to different environments, including support for [argo workflows](https://astrazeneca.github.io/runnable-core/configurations/executors/argo).

- **Efficient Debugging:** Quickly identify and resolve issues in pipeline execution with runnable's local
debugging features. Retrieve data from failed tasks and [retry failures](https://astrazeneca.github.io/runnable-core/concepts/run-log/#retrying_failures)
using your chosen debugging tools to maintain a smooth development experience.

Along with the developer friendly features, runnable also acts as an interface to production grade concepts
such as [data catalog](https://astrazeneca.github.io/runnable-core/concepts/catalog), [reproducibility](https://astrazeneca.github.io/runnable-core/concepts/run-log),
[experiment tracking](https://astrazeneca.github.io/runnable-core/concepts/experiment-tracking)
and secure [access to secrets](https://astrazeneca.github.io/runnable-core/concepts/secrets).

<hr style="border:2px dotted orange">

## What does it do?


![works](assets/work.png)

<hr style="border:2px dotted orange">

## Documentation

[More details about the project and how to use it available here](https://astrazeneca.github.io/runnable-core/).

<hr style="border:2px dotted orange">

## Installation

The minimum python version that runnable supports is 3.8

```shell
pip install runnable
```

Please look at the [installation guide](https://astrazeneca.github.io/runnable-core/usage)
for more information.

<hr style="border:2px dotted orange">

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
    runnable to understand the type of parameters you want.

    Input args can be a pydantic model or the individual attributes.
    """
    print(x)
    # >>> prints 1
    print(y)
    # >>> prints InnerModel(foo=10, bar="hello world")
```

### Application code using driver functions.

The code is runnable without any orchestration framework.

```python
from functions import return_parameter, display_parameter

my_param = return_parameter()
display_parameter(my_param.x, my_param.y)
```

### Orchestration using runnable

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
from runnable import Pipeline, Task

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


Execute via the cli: ```runnable execute -f examples/python-tasks.yaml```

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
    image: runnable:demo
    persistent_volumes:
      - name: runnable-volume
        mount_path: /mnt

run_log_store:
  type: file-system
  config:
    log_folder: /mnt/run_log_store
```

More details can be found in [argo configuration](https://astrazeneca.github.io/runnable-core/configurations/executors/argo).

Execute the code as ```runnable execute -f examples/python-tasks.yaml -c examples/configs/argo-config.yam```

<details>
  <summary>Expand</summary>

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: runnable-dag-
  annotations: {}
  labels: {}
spec:
  activeDeadlineSeconds: 172800
  entrypoint: runnable-dag
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
    - name: runnable-dag
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
        image: runnable:demo
        command:
          - runnable
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
        image: runnable:demo
        command:
          - runnable
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
        image: runnable:demo
        command:
          - runnable
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
        claimName: runnable-volume

```

</details>

## Pipelines can be:

### Linear

A simple linear pipeline with tasks either
[python functions](https://astrazeneca.github.io/runnable-core/concepts/task/#python_functions),
[notebooks](https://astrazeneca.github.io/runnable-core/concepts/task/#notebooks), or [shell scripts](https://astrazeneca.github.io/runnable-core/concepts/task/#shell)

[![](https://mermaid.ink/img/pako:eNpl0bFuwyAQBuBXQVdZTqTESpxMDJ0ytkszhgwnOCcoNo4OaFVZfvcSx20tGSQ4fn0wHB3o1hBIyLJOWGeDFJ3Iq7r90lfkkA9HHfmTUpnX1hFyLvrHzDLl_qB4-1BOOZGGD3TfSikvTDSNFqdj2sT2vBTr9euQlXNWjqycsN2c7UZWFMUE7udwP0L3y6JenNKiyfvz8t8_b-gavT9QJYY0PcDtjeTLptrAChriBq1JzeoeWkG4UkMKZCoN8k2Bcn1yGEN7_HYaZOBIK4h3g4EOFi-MDcgKa59SMja0_P7s_vAJ_Q_YOH6o?type=png)](https://mermaid.live/edit#pako:eNpl0bFuwyAQBuBXQVdZTqTESpxMDJ0ytkszhgwnOCcoNo4OaFVZfvcSx20tGSQ4fn0wHB3o1hBIyLJOWGeDFJ3Iq7r90lfkkA9HHfmTUpnX1hFyLvrHzDLl_qB4-1BOOZGGD3TfSikvTDSNFqdj2sT2vBTr9euQlXNWjqycsN2c7UZWFMUE7udwP0L3y6JenNKiyfvz8t8_b-gavT9QJYY0PcDtjeTLptrAChriBq1JzeoeWkG4UkMKZCoN8k2Bcn1yGEN7_HYaZOBIK4h3g4EOFi-MDcgKa59SMja0_P7s_vAJ_Q_YOH6o)

### [Parallel branches](https://astrazeneca.github.io/runnable-core/concepts/parallel)

Execute branches in parallel

[![](https://mermaid.ink/img/pako:eNp9k01rwzAMhv-K8S4ZtJCzDzuMLmWwwkh2KMQ7eImShiZ2sB1KKf3vs52PpsWNT7LySHqlyBeciRwwwUUtTtmBSY2-YsopR8MpQUfAdCdBBekWNBpvv6-EkFICzGAtWcUTDW3wYy20M7lr5QGBK2j-anBAkH4M1z6grnjpy17xAiTwDII07jj6HK8-VnVZBspITnpjztyoVkLLJOy3Qfrdm6gQEu2370Io7WLORo84PbRoA_oOl9BBg4UHbHR58UkMWq_fxjrOnhLRx1nH0SgkjlBjh7ekxNKGc0NelDLknhePI8qf7MVNr_31nm1wwNTeM2Ao6pmf-3y3Mp7WlqA7twOnXfKs17zt-6azmim1gQL1A0NKS3EE8hKZE4Yezm3chIVFiFe4AdmwKjdv7mIjKNYHaIBiYsycySPFlF8NxzotkjPPMNGygxXu2pxp2FSslKzBpGC1Ml7IKy3krn_E7i1f_wEayTcn?type=png)](https://mermaid.live/edit#pako:eNp9k01rwzAMhv-K8S4ZtJCzDzuMLmWwwkh2KMQ7eImShiZ2sB1KKf3vs52PpsWNT7LySHqlyBeciRwwwUUtTtmBSY2-YsopR8MpQUfAdCdBBekWNBpvv6-EkFICzGAtWcUTDW3wYy20M7lr5QGBK2j-anBAkH4M1z6grnjpy17xAiTwDII07jj6HK8-VnVZBspITnpjztyoVkLLJOy3Qfrdm6gQEu2370Io7WLORo84PbRoA_oOl9BBg4UHbHR58UkMWq_fxjrOnhLRx1nH0SgkjlBjh7ekxNKGc0NelDLknhePI8qf7MVNr_31nm1wwNTeM2Ao6pmf-3y3Mp7WlqA7twOnXfKs17zt-6azmim1gQL1A0NKS3EE8hKZE4Yezm3chIVFiFe4AdmwKjdv7mIjKNYHaIBiYsycySPFlF8NxzotkjPPMNGygxXu2pxp2FSslKzBpGC1Ml7IKy3krn_E7i1f_wEayTcn)

### [loops or map](https://astrazeneca.github.io/runnable-core/concepts/map)

Execute a pipeline over an iterable parameter.

[![](https://mermaid.ink/img/pako:eNqVlF1rwjAUhv9KyG4qKNR-3AS2m8nuBgN3Z0Sy5tQG20SSdE7E_76kVVEr2CY3Ied9Tx6Sk3PAmeKACc5LtcsKpi36nlGZFbXciHwfLN79CuWiBLMcEULWGkBSaeosA2OCxbxdXMd89Get2bZASsLiSyuvQE2mJZXIjW27t2rOmQZ3Gp9rD6UjatWnwy7q6zPPukd50WTydmemEiS_QbQ79RwxGoQY9UaMuojRA8TCXexzyHgQZNwbMu5Cxl3IXNX6OWMyiDHpzZh0GZMHjOK3xz2mgxjT3oxplzG9MPp5_nVOhwJjteDwOg3HyFj3L1dCcvh7DUc-iftX18n6Waet1xX8cG908vpKHO6OW7cvkeHm5GR2b3drdvaSGTODHLW37mxabYC8fLgRhlfxpjNdwmEets-Dx7gCXTHBXQc8-D2KbQEVUEzckjO9oZjKo9Ox2qr5XmaYWF3DGNdbzizMBHOVVWGSs9K4XeDCKv3ZttSmsx7_AYa341E?type=png)](https://mermaid.live/edit#pako:eNqVlF1rwjAUhv9KyG4qKNR-3AS2m8nuBgN3Z0Sy5tQG20SSdE7E_76kVVEr2CY3Ied9Tx6Sk3PAmeKACc5LtcsKpi36nlGZFbXciHwfLN79CuWiBLMcEULWGkBSaeosA2OCxbxdXMd89Get2bZASsLiSyuvQE2mJZXIjW27t2rOmQZ3Gp9rD6UjatWnwy7q6zPPukd50WTydmemEiS_QbQ79RwxGoQY9UaMuojRA8TCXexzyHgQZNwbMu5Cxl3IXNX6OWMyiDHpzZh0GZMHjOK3xz2mgxjT3oxplzG9MPp5_nVOhwJjteDwOg3HyFj3L1dCcvh7DUc-iftX18n6Waet1xX8cG908vpKHO6OW7cvkeHm5GR2b3drdvaSGTODHLW37mxabYC8fLgRhlfxpjNdwmEets-Dx7gCXTHBXQc8-D2KbQEVUEzckjO9oZjKo9Ox2qr5XmaYWF3DGNdbzizMBHOVVWGSs9K4XeDCKv3ZttSmsx7_AYa341E)

### [Arbitrary nesting](https://astrazeneca.github.io/runnable-core/concepts/nesting/)
Any nesting of parallel within map and so on.

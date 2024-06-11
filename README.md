




</p>
<hr style="border:2px dotted orange">

<p align="center">
<a href="https://pypi.org/project/runnable/"><img alt="python:" src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg"></a>
<a href="https://pypi.org/project/runnable/"><img alt="Pypi" src="https://badge.fury.io/py/runnable.svg"></a>
<a href="https://github.com/vijayvammi/runnable/blob/main/LICENSE"><img alt"License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/python/mypy"><img alt="MyPy Checked" src="https://www.mypy-lang.org/static/mypy_badge.svg"></a>
<a href="https://github.com/vijayvammi/runnable/actions/workflows/release.yaml"><img alt="Tests:" src="https://github.com/vijayvammi/runnable/actions/workflows/release.yaml/badge.svg">
</p>
<hr style="border:2px dotted orange">


[Please check here for complete documentation](https://astrazeneca.github.io/runnable/)

## Example

The below data science flavored code is a well-known
[iris example from scikit-learn](https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html).


```python
"""
Example of Logistic regression using scikit-learn
https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression


def load_data():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    Y = iris.target

    return X, Y


def model_fit(X: np.ndarray, Y: np.ndarray, C: float = 1e5):
    logreg = LogisticRegression(C=C)
    logreg.fit(X, Y)

    return logreg


def generate_plots(X: np.ndarray, Y: np.ndarray, logreg: LogisticRegression):
    _, ax = plt.subplots(figsize=(4, 3))
    DecisionBoundaryDisplay.from_estimator(
        logreg,
        X,
        cmap=plt.cm.Paired,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        shading="auto",
        xlabel="Sepal length",
        ylabel="Sepal width",
        eps=0.5,
    )

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors="k", cmap=plt.cm.Paired)

    plt.xticks(())
    plt.yticks(())

    plt.savefig("iris_logistic.png")

    # TODO: What is the right value?
    return 0.6


## Without any orchestration
def main():
    X, Y = load_data()
    logreg = model_fit(X, Y, C=1.0)
    generate_plots(X, Y, logreg)


## With runnable orchestration
def runnable_pipeline():
    # The below code can be anywhere
    from runnable import Catalog, Pipeline, PythonTask, metric, pickled

    # X, Y = load_data()
    load_data_task = PythonTask(
        function=load_data,
        name="load_data",
        returns=[pickled("X"), pickled("Y")],  # (1)
    )

    # logreg = model_fit(X, Y, C=1.0)
    model_fit_task = PythonTask(
        function=model_fit,
        name="model_fit",
        returns=[pickled("logreg")],
    )

    # generate_plots(X, Y, logreg)
    generate_plots_task = PythonTask(
        function=generate_plots,
        name="generate_plots",
        terminate_with_success=True,
        catalog=Catalog(put=["iris_logistic.png"]),  # (2)
        returns=[metric("score")],
    )

    pipeline = Pipeline(
        steps=[load_data_task, model_fit_task, generate_plots_task],
    )  # (4)

    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    # main()
    runnable_pipeline()

```


1. Return two serialized objects X and Y.
2. Store the file `iris_logistic.png` for future reference.
3. Define the sequence of tasks.
4. Define a pipeline with the tasks

The difference between native driver and runnable orchestration:

!!! tip inline end "Notebooks and Shell scripts"

    You can execute notebooks and shell scripts too!!

    They can be written just as you would want them, *plain old notebooks and scripts*.




<div class="annotate" markdown>

```diff

- X, Y = load_data()
+load_data_task = PythonTask(
+    function=load_data,
+     name="load_data",
+     returns=[pickled("X"), pickled("Y")], (1)
+    )

-logreg = model_fit(X, Y, C=1.0)
+model_fit_task = PythonTask(
+   function=model_fit,
+   name="model_fit",
+   returns=[pickled("logreg")],
+   )

-generate_plots(X, Y, logreg)
+generate_plots_task = PythonTask(
+   function=generate_plots,
+   name="generate_plots",
+   terminate_with_success=True,
+   catalog=Catalog(put=["iris_logistic.png"]), (2)
+   )


+pipeline = Pipeline(
+   steps=[load_data_task, model_fit_task, generate_plots_task], (3)

```
</div>


---

- [x] ```Domain``` code remains completely independent of ```driver``` code.
- [x] The ```driver``` function has an equivalent and intuitive runnable expression
- [x] Reproducible by default, runnable stores metadata about code/data/config for every execution.
- [x] The pipeline is `runnable` in any environment.


## Documentation

[More details about the project and how to use it available here](https://astrazeneca.github.io/runnable/).

<hr style="border:2px dotted orange">

## Installation

The minimum python version that runnable supports is 3.8

```shell
pip install runnable
```

Please look at the [installation guide](https://astrazeneca.github.io/runnable-core/usage)
for more information.


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

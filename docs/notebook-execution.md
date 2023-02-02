# Executing Notebooks

Jupyter notebook are tools of the trade for many data scientists due to their interactivity and easy of use. In magnus,
you can run a notebook in any remote environment just like your local environment.

## Pre-requisites
Other than ```local``` compute environment, where the notebook essentially runs in the same python environment you are
using to invoke, the notebook would be run in a containerized environment with the current notebook as part of the
container. The process of creating/storing the docker images is outside of scope of magnus.

This is a design choice as CI/CD processes vary between teams.

## Example Notebook

``` title="Example notebook"
--8<-- "input.html"
```

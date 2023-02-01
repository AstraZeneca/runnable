# Installation

The minimum python version that magnus supports is 3.8
## pip

magnus is a python package and should be installed as any other.

```shell
pip install magnus
```

We recommend that you install magnus in a virtual environment specific to the project and also poetry for your
application development.

The command to install in a poetry managed virtual environment

```
poetry add magnus
```

## Optional capabilities

### Docker

To run the pipelines in a container, you need to install magnus with docker functionality.

```shell
pip install magnus[docker]
```

or if you are using poetry

```shell
poetry add magnus[docker]
```

### Notebook

To use notebook functionality, you need to install magnus with notebook functionality.

```shell
pip install magnus[notebook]
```

or if you are using poetry

```shell
poetry add magnus[notebook]
```

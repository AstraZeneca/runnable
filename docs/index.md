---
title: Welcome
sidebarDepth: 0
---

![logo](../assets/logo1.png){ width="400" height="300" style="display: block; margin: 0 auto" }

---

**Magnus** is a *thin* layer of abstraction provided over underlying infrastructure to enable data scientist and
machine learning engineers. It provides:

- A way to execute Jupyter notebooks/python functions in local or remote platforms.
- A framework to define complex pipelines via YAML or Python SDK.
- Robust and *invisible* logging to ensure maximum reproducibility of experiments.
- A framework to interact with secret managers ranging from environment variables to other vendors.
- Interactions with various experiment tracking tools.


| Interaction | Infrastructure Patterns | Abstraction |
|:-------------:| :------------------------:| :------------:|
|Run a Jupyter notebook| Use platform dependent (AWS Sagemaker, Kubeflow) notebooks |  magnus execute_notebook <> |
|               | Run it as a job in K8's or AWS |
|Run a python function| Use platform dependent (AWS Sagemaker, Kubeflow) IDE's |  magnus execute_function <> |
|               | Run it as a job in K8's or AWS |



:sparkles::sparkles:Happy Experimenting!!:sparkles::sparkles:

Please find the [github project here](https://github.com/AstraZeneca/magnus-core).

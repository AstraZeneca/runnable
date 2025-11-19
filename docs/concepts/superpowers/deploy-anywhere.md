# 🌍 Deploy Anywhere

Here's the ultimate superpower: Same code runs everywhere. Just change the configuration.

## Your code never changes

```python
from runnable import Pipeline, PythonTask

def train_model():
    # Your model training logic
    print("Training model...")
    return "model_v1.pkl"

pipeline = Pipeline(steps=[
    PythonTask(function=train_model, name="training")
])
```

## Change where it runs

### 💻 Local development
```python
pipeline.execute()  # Runs on your laptop
```

### 🐳 Container execution
```python
pipeline.execute(config="container.yaml")  # Runs in Docker
```

### ☁️ Cloud platforms
```python
pipeline.execute(config="argo.yaml")  # Runs on Argo Workflows
```

## Configuration files define the environment

**Local config** (fast for development):
```yaml title="local.yaml"
pipeline-executor:
  type: local # (1)

run-log-store:
  type: file-system

catalog:
  type: file-system # (3)

secrets:
  type: env-secrets # (4)

```

**Argo config** (production scale):
```yaml title="kubernetes.yaml"
pipeline-executor:
  type: "argo" # (1)
  config:
    pvc_for_runnable: runnable
    defaults:
      image: $docker_image # (3)
      resources:
        limits:
          cpu: "1"
          memory: 1Gi
        requests:
          cpu: "0.5"
          memory: 500Mi
      env:
        - name: argo_env
          value: "argo"
    argoWorkflow:
      metadata:
        generateName: "argo-" # (2)
        namespace: my_namespace
      spec:
        serviceAccountName: "default-editor"


run-log-store: # (4)
  type: chunked-fs
  config:
    log_folder: /mnt/run_log_store

```

## The power of environment-agnostic code

**Development:**
```bash
# Quick local testing
uv run my_pipeline.py
```

**Production:**
```bash
# Same code, production environment
runnable execute my_pipeline.py --config prod-k8s.yaml
```

## Real-world example

From development to production:

```python
from runnable import Pipeline, PythonTask
from examples.common.functions import hello

# Same code, different environments
task = PythonTask(function=hello, name="say_hello")
pipeline = Pipeline(steps=[task])

# Development: just run it
pipeline.execute()

# Production: same code, different config
pipeline.execute(config="kubernetes.yaml")
```

??? example "See complete runnable code"
    ```python title="examples/01-tasks/python_tasks.py"
    --8<-- "examples/01-tasks/python_tasks.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/01-tasks/python_tasks.py
    ```

**Dev environment:** Runs in 2 seconds on your laptop
**Prod environment:** Runs on Kubernetes with monitoring, logging, and auto-scaling

**Zero code changes.**

!!! tip "Start simple, scale up"

    1. Develop locally with no config
    2. Test in containers with `container.yaml`
    3. Deploy to production with `kubernetes.yaml`

    Same pipeline, different environments.

You now know the core concepts! For advanced workflows, check out the reference documentation.

# 🌍 Deploy Anywhere

Here's the ultimate superpower: Same code runs everywhere. Just change the configuration.

## Your code never changes

```python
def train_model():
    # Your model training logic
    pass

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

### ☸️ Kubernetes cluster
```python
pipeline.execute(config="kubernetes.yaml")  # Runs on K8s
```

### ☁️ Cloud platforms
```python
pipeline.execute(config="argo.yaml")  # Runs on Argo Workflows
```

## Configuration files define the environment

**Local config** (fast for development):
```yaml title="local.yaml"
run_log_store:
  type: file-system
  config:
    log_folder: ".run_logs"

catalog:
  type: file-system
  config:
    compute_data_folder: ".catalog"
```

**Kubernetes config** (production scale):
```yaml title="kubernetes.yaml"
run_log_store:
  type: k8s-pvc
  config:
    persistent_volume_claim: runnable-pvc

catalog:
  type: s3
  config:
    bucket: my-production-bucket
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

```python linenums="1"
--8<-- "examples/01-tasks/python_tasks.py:7:17"
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

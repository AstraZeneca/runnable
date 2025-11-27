# Running Anywhere

You've built a great ML pipeline that works on your laptop. But what about production? Containers? Kubernetes? The good news: your code doesn't need to change.

## The Deployment Challenge

Traditional ML pipelines require code changes for different environments:

```python
# Development version
results = train_model(data, local=True)

# Production version
results = train_model(data, use_kubernetes=True, replicas=5)

# Container version
results = train_model(data, docker=True, image="my-model:latest")
```

**Problems:**

- Different code for different environments
- Hard to test production code locally
- Risk of bugs when deploying
- Code becomes cluttered with infrastructure logic

## The Runnable Way: Configuration Over Code

With Runnable, your code stays the same. Only the configuration changes:

```python title="examples/tutorials/getting-started/07_running_anywhere.py"
# This exact same code runs everywhere!
pipeline = Pipeline(steps=[
    PythonTask(function=load_data, name="load_data", returns=[pickled("df")]),
    PythonTask(function=preprocess_data, name="preprocess", returns=[pickled("preprocessed_data")]),
    PythonTask(function=train_model, name="train", returns=[pickled("model_data")]),
    PythonTask(function=evaluate_model, name="evaluate", returns=[pickled("evaluation_results")])
])

pipeline.execute()  # Environment determined by config, not code
```

**Try it:**

```bash
uv run examples/tutorials/getting-started/07_running_anywhere.py
```

## Same Code, Different Environments

### 1. Local Execution (Development)

Run on your laptop with default settings:

```bash
uv run examples/tutorials/getting-started/07_running_anywhere.py
```

**What happens:**

- Runs directly on your machine
- Uses local file system for storage
- Fast iteration during development
- No infrastructure required

### 2. Container Execution (Testing)

Run in containers for isolated testing:

```bash
uv run examples/tutorials/getting-started/07_running_anywhere.py \
  --config examples/configs/local-container.yaml
```

```yaml title="examples/configs/local-container.yaml"
pipeline-executor:
  type: "local-container"
  config:
    docker_image: runnable-m1:latest
    enable_parallel: true
```

**What changes:**

- Each step runs in a Docker container
- Same local file system access
- Tests containerized behavior locally
- **Your code: unchanged**

### 3. Cloud Storage (Production-like)

Use cloud storage for data:

```bash
uv run examples/tutorials/getting-started/07_running_anywhere.py \
  --config examples/configs/s3-storage.yaml
```

```yaml title="examples/configs/s3-storage.yaml"
catalog:
  type: "s3"
  config:
    bucket: "my-ml-artifacts"
    region: "us-west-2"
```

**What changes:**

- Artifacts stored in S3
- Team can access results
- Production-ready storage
- **Your code: unchanged**

### 4. Kubernetes Execution (Production)

Run on Kubernetes cluster:

```bash
uv run examples/tutorials/getting-started/07_running_anywhere.py \
  --config examples/configs/kubernetes.yaml
```

```yaml title="examples/configs/kubernetes.yaml"
pipeline-executor:
  type: "kubernetes"
  config:
    namespace: "ml-pipelines"
    image: "my-registry/ml-pipeline:v1"

catalog:
  type: "s3"
  config:
    bucket: "production-ml-artifacts"
```

**What changes:**

- Runs on Kubernetes pods
- Scales automatically
- Production-grade execution
- **Your code: unchanged**

## The Power of Configuration

All these configurations are external to your code:

```python
# Your pipeline code (never changes)
from functions import load_data, train_model
from runnable import Pipeline, PythonTask, pickled

pipeline = Pipeline(steps=[
    PythonTask(function=load_data, returns=[pickled("df")]),
    PythonTask(function=train_model, returns=[pickled("model")])
])

# Environment determined at runtime by config
pipeline.execute()
```

## Development to Production Workflow

### Step 1: Develop Locally

```bash
# Fast iteration on your laptop
uv run 07_running_anywhere.py
```

### Step 2: Test in Containers

```bash
# Verify containerized behavior
uv run 07_running_anywhere.py --config local-container.yaml
```

### Step 3: Deploy to Staging

```bash
# Run on staging cluster with cloud storage
uv run 07_running_anywhere.py --config staging.yaml
```

### Step 4: Deploy to Production

```bash
# Same code, production configuration
uv run 07_running_anywhere.py --config production.yaml
```

**At no point did you change your pipeline code!**

## Configuration Options

Runnable supports many deployment scenarios through configuration:

### Execution Environments

- **local**: Run directly on your machine
- **local-container**: Run in Docker containers locally
- **kubernetes**: Run on Kubernetes cluster
- **argo**: Use Argo Workflows for complex DAGs

### Storage Options

- **file-system**: Local file storage
- **s3**: AWS S3 buckets
- **minio**: Self-hosted S3-compatible storage
- **azure-blob**: Azure Blob Storage

### Run Log Storage

- **file-system**: Local JSON files
- **chunked-fs**: Optimized local storage
- **database**: PostgreSQL, MySQL
- **cloud**: S3, Azure, GCS

### Secret Management

- **env-secrets**: Environment variables
- **dotenv**: .env files
- **aws-secrets**: AWS Secrets Manager
- **azure-secrets**: Azure Key Vault

## Complete Example Pipeline

Here's the complete pipeline that runs anywhere:

```python title="examples/tutorials/getting-started/07_running_anywhere.py"
--8<-- "examples/tutorials/getting-started/07_running_anywhere.py:12:77"
```

## What You've Achieved

### 💻 **Code Portability**

Your pipeline code works everywhere:

- Local laptop for development
- Docker containers for testing
- Kubernetes for production
- Cloud platforms without changes

### 🔧 **Configuration-Driven**

Change behavior without code changes:

- Switch storage backends
- Change execution environments
- Scale up or down
- All through configuration

### 🎯 **Develop Locally, Deploy Anywhere**

The best development experience:

1. Write code locally with fast feedback
2. Test in containers for isolation
3. Deploy to production with confidence
4. No code changes between environments

### 🚀 **Production Ready**

Built-in support for:

- Distributed execution
- Cloud storage
- Secret management
- Monitoring and logging

## Real-World Example

A typical ML team workflow:

```bash
# Data scientist develops locally
python train.py

# CI/CD tests in containers
python train.py --config ci-container.yaml

# Model engineer validates on staging
python train.py --config staging-k8s.yaml

# Production deployment
python train.py --config production-k8s.yaml
```

**Same Python file. Four different environments. Zero code changes.**

## Tutorial Complete!

Congratulations! You've learned how to:

1. ✅ **Start simple** - Transform a basic ML function into a pipeline
2. ✅ **Make it reproducible** - Automatic tracking of all runs and results
3. ✅ **Add flexibility** - Configure experiments without code changes
4. ✅ **Connect workflow** - Multi-step pipelines with automatic data flow
5. ✅ **Handle large datasets** - Efficient file-based storage
6. ✅ **Share results** - Persistent models and metrics
7. ✅ **Run anywhere** - Same code, different environments

## What's Next?

Explore more advanced features:

- **[Parallel Execution](../advanced-patterns/parallel-execution.md)** - Run independent steps concurrently
- **[Conditional Workflows](../advanced-patterns/conditional-workflows.md)** - Dynamic workflow decisions
- **[Map Patterns](../advanced-patterns/map-patterns.md)** - Process items in parallel
- **[Deploy Anywhere](../production/deploy-anywhere.md)** - Production deployment strategies

## Summary

The key insight of this tutorial:

> **Separate your ML logic from infrastructure concerns. Your functions stay pure and simple. Runnable handles the orchestration, storage, tracking, and deployment.**

This separation enables:

- Faster development (test locally)
- Easier testing (same code everywhere)
- Confident deployment (proven code)
- Better collaboration (shared understanding)

Ready to build production ML pipelines? You now have all the foundations!

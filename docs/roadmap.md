## AWS environments

Bring in native AWS services to orchestrate workflows. The stack should be:

- AWS step functions.
- Sagemaker jobs - Since they can take dynamic image name, AWS batch needs job definition and can be tricky.
- S3 for Run log and Catalog: Already tested and working prototype.
- AWS secrets manager: Access to AWS secrets manager via the RBAC of the execution role.


## HPC environment using SLURM executor.

- Without native orchestration tools, the preferred way is to run it as local but use SLURM to schedule jobs.

## Database based Run log store.

## Better integrations with experiment tracking tools.

Currently, the implementation of experiment tracking tools within magnus is limited. It might be better to
choose a good open source implementation and stick with it.


## Model registry service

Could be interesting to bring in a model registry to catalog models.

# Why Magnus

Magnus is never set out to replace  production grade orchestrators like AWS Step functions or argo. These
orchestrators are proven to be robust and are constantly improved to align to best practices. We agree that, we should
always use these tools for production grade deployments.

But the same tools, seem to over-engineered and extremely complex for experiments and local development where the actual
data science teams thrive. The farther the operational world is from the developers, the longer it takes to
operationalize projects - lesson learnt from DevOps. Magnus was developed to bring the data science team closer to the
production infrastructure and practices while abstracting a lof of underlying complexity.


Magnus treats the *dag* definition as a contract between the data science team and the engineering team. While the dag
could be run on local computers or in cloud by the data science team during the development/experiment phase, the dag
is translated to chosen orchestrators language during deployment by the engineering team. This also enables the data
science team to think along the lines of pipelines and orchestration without infrastructure complexities.

We also found that, a few implementations in magnus to be more convenient than the counterparts it tries to
emulate. For example: passing variables between steps in AWS Step function is complex and not even possible when
using containers as one of the steps. The same step when wrapped around magnus before step function makes it easier.


Here are some of the key points on choosing magnus.

## Reproducibility of Experiments

Data science experiments and projects are notorious for being difficult to replicate. In our opinion, a data science
experiment is [code](../../concepts/run-log/#code_identity) + [data](../../concepts/run-log/#data_catalog) +
configuration.
Magnus tracks all three of them in the run logs and makes it easier to
reproduce/highlight differences between experiments.

If the default tracking provided by Magnus is not suitable, you can easily integrate your application with one of
your liking or extend magnus to fit your needs.

## Easy re-run

Along the same lines as reproducibility, a pipeline run with magnus can be re-run on any other environment as long as
the run log/catalog are available. Magnus would skip the steps that were successfully executed in the older
run and start execution from the point of failure.


## Extensibility

A lot of design principles while writing magnus was to promote [extensibility](../../extensions/extensions).
Its easy to write extensions to include
new

- [compute environments](../../concepts/modes-implementations/extensions/) (k8s, on-prem clusters etc)
- [run log store](../../concepts/run-log-implementations/extensions/) (databases, file systems etc)
- [data cataloging](../../concepts/catalog-implementations/extensions/) (feature stores, object storage etc)
- [secret managers](../../concepts/secrets-implementations/extensions/) (vault, azure secrets)

## Near Zero code change from local to production

Magnus was designed to make data science teams closer to operational world. The code and orchestration are ready to
be productionized as soon as you are ready. The only change to enable that would be a
[config](../../concepts/configurations/).

## Easy switch

The technological decisions made today for your project may not be correct one in a few months for a lot of varied
reasons. You might want to change your cloud provider or orchestrating tools or secrets manager and it should be easy
to do so. With magnus, you can easily switch without touching your project code/practices. Since the configuration
could also be parameterized, switching might be as simple as changing one file. For more details check here.

And one of the design principles in magnus was to limit needing to ```import magnus``` to achieve functionality.
This also means that you can move away from magnus if its no longer supporting you. :-)

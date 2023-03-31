# How do I

## Pass parameters between steps?

--8<--
docs/concepts/nodes.md:how-do-i-pass-simple
--8<--

## Pass data files between steps?

In magnus, data files are passed to downstream steps using the concept of [catalog](../concepts/catalog). The catalog
settings and behavior can be completely controlled by the pipeline definition but can also be controlled via code if
its convenient.

--8<--
docs/concepts/catalog.md:how-do-i-pass-data
--8<--


## Pass data objects between steps?

In magnus, data are passed to downstream steps using the concept of [catalog](../concepts/catalog). While this is
good for files, it is inconvenient to dump and load the object into files for the cataloging to happen. Magnus provides
utility functions to make it easier.

--8<--
docs/concepts/catalog.md:how-do-i-pass-objects
--8<--

## Define variables?

--8<--
docs/concepts/dag.md:how-do-i-parameterize
--8<--


## Track experiments?

--8<--
docs/concepts/experiment-tracking.md:how-do-i-track
--8<--

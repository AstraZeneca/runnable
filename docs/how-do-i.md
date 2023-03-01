# How do I

## Pass parameters between steps?

 In magnus, we classify 2 kinds of data sets that can be passed around to down stream steps.

- Data: Processed files by an upstream step should be available for downstream steps when required.
[Catalog](../concepts/catalog) provides the way to do this.

- Parameters: Any JSON serializable data can be passed to down stream steps.

### Parameters from command line


Initial parameters to the application can be sent in via a parameters file.

Example:

```shell
magnus execute --file getting-started.yaml --parameters-file parameters.yaml
```

```yaml
# in parameters.yaml
arg1 : test
arg2: dev
```

Or via environmental variables: Any environmental variable with prefix ```MAGNUS_PRM_``` is considered as a magnus
parameter. Eg: ```MAGNUS_PRM_arg1=test``` or ```MAGNUS_PRM_arg2=dev```.

---
!!! Note

    Parameters via environmental variables over-ride the parameters defined via parameters file.
---



In this case, arg1 and arg2 are available as parameters to downstream steps.

### Storing parameters

Any JSON serializable dictionary returned from a task node is available as parameters to downstream steps.

Example:

```python

def my_cool_function():
  return {'arg1': 'hello', 'arg2': {'hello', 'world'} }

```

Or

```python

from magnus import store_parameter

def my_cool_function():
  store_parameter(arg1='hello', 'arg2'={'hello', 'world'})

```

Or

```python
import os
import json

def my_cool_function():
  os.environ['MAGNUS_PRM_' + 'arg1'] = 'hello'
  os.environ['MAGNUS_PRM_' + 'arg2'] = json.dumps({'hello', 'world'})
```

All the three above ways store arg1 and arg2 for downstream steps.

### Accessing parameters

Any parameters set either at command line or by upstream nodes can be accessed by:


``` python
def my_cool_function(arg1, arg2=None):
  pass

```
The function is inspected to find all *named* args and provided value if the key exists in the parameters.

or

``` python

import os

def my_cool_function():
  arg1 = os.environ['MAGNUS_PRM_arg1']
  arg2 = os.environ['MAGNUS_PRM_arg2']

```
or

``` python

from magnus import get_parameter

def my_cool_function():
  arg1 = get_parameter('arg1')
  arg2 = get_parameter('arg2')

```

or
``` python
from magnus import get_parameter

def my_cool_function():
  args = get_parameter()
  arg1 = args['arg1']
  arg2 = args['arg2']

```
Calling ```get_parameter``` with no key returns all parameters.


## Pass data files between steps?

In magnus, data files are passed to downstream steps using the concept of [catalog](../concepts/catalog). The catalog
settings and behavior can be completely controlled by the pipeline definition but can also be controlled via code if
its convenient.

### Get from catalog

To get a file from the catalog, use ```get_from_catalog``` from magnus.

For example, the below code gets the file ```interesting_data.csv``` from the catalog into ```data/``` folder.


```python
from magnus import get_from_catalog

def my_function():
  get_from_catalog('interesting.csv', destination_folder='data/')

```

### Put in catalog

To put a file into the catalog, use ```put_in_catalog``` from magnus.

For example, the below code puts the file ```data/interesting_data.csv``` from the data folder into catalog.


```python
from magnus import put_in_catalog

def my_function():
  put_in_catalog('data/interesting.csv')

```

---
!!! Note

    Unlike ```put``` phase of the cataloging process, put_in_catalog does not check if the cataloging object has
    changed and does a blind update.

---

## Pass data objects between steps?

In magnus, data are passed to downstream steps using the concept of [catalog](../concepts/catalog). While this is
good for files, it is inconvenient to dump and load the object into files for the cataloging to happen. Magnus provides
utility functions to make it easier.

### Get object from catalog

To get a object from the catalog, use ```get_object``` from magnus.

For example, the below code gets a pandas dataframe from previous steps, called ```interesting_data``` from the catalog.


```python
from magnus import get_object

def my_function():
  df = get_object("interesting_data")

```

Be aware that, the function would raise an exception if ```interesting_data``` was not added to catalog before.

### Put object in catalog

To put a object into the catalog, use ```put_object``` from magnus.

For example, the below code puts the dataframe ```interesting_data``` into the catalog as ```interesting_data.pickle```.


```python
from magnus import put_object

def my_function():
  put_object(data=interesting_data, name="interesting_data")

```

---
!!! Note

    We internally use pickle for the serialization and deserialization. Please raise a feature request if you need
    other kind of serializers.

---

## Define variables?

Magnus allows dag definitions and configurations to be parameterized by using placeholders.
We use [python String templates](https://docs.python.org/3.7/library/string.html#template-strings)
to enable parameter substitution. As we use,
[safe_substitution](https://docs.python.org/3.7/library/string.html#string.Template.safe_substitute)
it means that we silently ignore any parameter that is not found.
You should make sure that the parameters are properly defined.

### Example of variables
Assuming this is in dag-variable.yaml
```yaml
dag:
  start_at: Hello
  steps:
    Hello:
      type: task
      command: ${module_name}
      next: Success
    Success:
      type: success
    Fail:
      type: fail
```

```python
# in my_module.py

def say_hello(name=world):
    print(f'Hello {name}')
```

Magnus variables can be defined by environmental variables, any string with a prefix ```MAGNUS_VAR_``` is considered a
variable.

```shell
export MAGNUS_VAR_module_name=my_module.say_hello
export MAGNUS_PRM_name="universe"
```
and with the python code shown, we can achieve the same result by:
```shell
magnus execute --file dag-variable.yaml
```

Magnus would resolve the placeholders at the load of the dag definition.

## Track experiments?

In magnus, experiment tracking is defined per step of the pipeline. You can instruct magnus to track a metric by:

```python
# In some step of the pipeline or function or notebook
from magnus import track_this

track_this(answer=42, step=0)
```

This would result in a corresponding entry in ```user_defined_metrics``` of the step log by default and also would be
passed to the integrated experiment tracking tool.

---
!!! Note

    Magnus allows the value pair of the metric to be any JSON friendly datatype. The underlying experiment tracker
    should have mechanisms to handle the same.
---


### step parameter

Many implementations of experiment tracking tools support a step parameter to define the history of the parameter.
Any value other than ```0``` as step parameter would create a ```user_defined_metric``` of ```key_{step}```.

### Environmental variables

You can also track metrics via environmental variables. Any environmental variable with prefix ```MAGNUS_TRACK_``` is
added to the step log.


```python
import os

os.environ['MAGNUS_TRACK_answer']=42
```

---
!!! Note

    Setting metrics via environmental variables would not invoke the underlying experiment tracking tool.
---


### Client context

You can also get a initialized context of the experiment tracking tool to completely control the behavior.

```python
from magnus import get_experiment_tracker_context
context = get_experiment_tracker_context()

with context as ctx:
    # do something
```

### Grouping of experiments

With experiment tracking tools that support grouping of experiments under a logical group, you can use ```tag``` of
magnus.

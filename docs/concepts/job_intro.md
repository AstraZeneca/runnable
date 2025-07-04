Jobs are isolated unit of work which can be python functions, jupyter notebooks or shell scripts.



Considering a simple function:

```python
def add_numbers(x: int, y: int):
    # save some data in data.csv
    return x + y
```

The runnable representation of it is:

```python
from functions import add_numbers
from runnable import PythonJob, Catalog

write_catalog = Catalog(put=["data.csv"])
job = PythonJob(function=add_numbers,
                returns["sum_of_numbers"],
                catalog=write_catalog,
            )

```

```PythonJob``` requires a function to call. The input parameters are passed in
 from the parameters provided at the time of execution.

The return parameters are stored for future reference. Any data object generated in the
process can be saved to the catalog.

<hr style="border:2px dotted orange">


## Python functions

You can use Python functions as jobs in a pipeline, enabling flexible encapsulation of logic, parameter passing, result capturing, and cataloging of outputs.

=== "Basic Python Function as a Job"
    ```python
    --8<-- "examples/11-jobs/python_tasks.py"
    ```

    The stdout (e.g., "Hello World!") and logs are captured and stored in the catalog for traceability.

=== "Writing Data to the Catalog"
    ```python
    --8<-- "examples/11-jobs/catalog.py"
    ```

    The `Catalog` object specifies which files or data should be saved after job execution.

=== "Passing and Returning Parameters"

    ```python
    --8<-- "examples/11-jobs/passing_parameters_python.py"
    ```

    Parameters can be passed at execution time, and returned values can be automatically handled, serialized, and tracked as metrics.

---

## Notebooks

You can also use Jupyter notebooks as jobs in your pipeline. This allows you to encapsulate notebook logic, capture outputs, and integrate notebooks seamlessly into your workflow.

=== "Notebook as a Job"
    ```python
    --8<-- "examples/11-jobs/notebooks.py"
    ```
    The output of the notebook will be captured as execution log
    along with the actual notebook and stored in the catalog for traceability.

---

## Shell script

You can also use shell scripts or commands as jobs in your pipeline. This allows you to execute any shell command, capture its output, and integrate it into your workflow.

=== "Shell Script"
    ```python
    --8<-- "examples/11-jobs/scripts.py"
    ```
    The stdout and stderr of the shell command are captured as execution log and stored in the catalog for traceability.

For more advanced examples, see the files in `examples/11-jobs/`.

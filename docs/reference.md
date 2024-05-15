## PythonTask

=== "sdk"

    ::: runnable.PythonTask
        options:
            show_root_heading: true
            show_bases: false
            show_docstring_description: true
            heading_level: 3

=== "yaml"

    Attributes:

    - ```name```: the name of the task
    - ```command```: the dotted path reference to the function.
    - ```next```: the next node to call if the function succeeds. Use ```success``` to terminate
    the pipeline successfully or ```fail``` to terminate with fail.
    - ```on_failure```: The next node in case of failure.
    - ```catalog```: mapping of cataloging items
    - ```overrides```: mapping of step overrides from global configuration.

    ```yaml
    dag:
      steps:
        name: <>
          type: task
          command: <>
          next: <>
          on_failure: <>
          catalog: # Any cataloging to be done.
          overrides: # mapping of overrides of global configuration
    ```

<hr style="border:2px dotted orange">


## NotebookTask

=== "sdk"

    ::: runnable.NotebookTask
        options:
            show_root_heading: true
            show_bases: false
            show_docstring_description: true
            heading_level: 3

=== "yaml"

    Attributes:

    - ```name```: the name of the task
    - ```command```: the path to the notebook relative to the project root.
    - ```next```: the next node to call if the function succeeds. Use ```success``` to terminate
    the pipeline successfully or ```fail``` to terminate with fail.
    - ```on_failure```: The next node in case of failure.
    - ```catalog```: mapping of cataloging items
    - ```overrides```: mapping of step overrides from global configuration.

    ```yaml
    dag:
      steps:
        name: <>
          type: task
          command: <>
          next: <>
          on_failure: <>
          catalog: # Any cataloging to be done.
          overrides: # mapping of overrides of global configuration
    ```


<hr style="border:2px dotted orange">


## Catalog

=== "sdk"

    ::: runnable.Catalog
        options:
            show_root_heading: true
            show_bases: false
            heading_level: 3

=== "yaml"



<hr style="border:2px dotted orange">

## Stub

=== "sdk"

    ::: runnable.Stub
        options:
            show_root_heading: true
            show_bases: false
            heading_level: 3

=== "yaml"



<hr style="border:2px dotted orange">



## ShellTask

=== "sdk"

    ::: runnable.ShellTask
        options:
            show_root_heading: true
            show_bases: false
            show_docstring_description: true
            heading_level: 3

=== "yaml"



<hr style="border:2px dotted orange">




## Parallel


=== "sdk"

    ::: runnable.Parallel
        options:
            show_root_heading: true
            show_bases: false
            show_docstring_description: true
            heading_level: 3

=== "yaml"



<hr style="border:2px dotted orange">

## Map

=== "sdk"

    ::: runnable.Map
        options:
            show_root_heading: true
            show_bases: false
            show_docstring_description: true
            heading_level: 3

=== "yaml"

<hr style="border:2px dotted orange">



::: runnable.Success
    options:
        show_root_heading: true
        show_bases: false
        show_docstring_description: true

<hr style="border:2px dotted orange">

::: runnable.Fail
    options:
        show_root_heading: true
        show_bases: false
        show_docstring_description: true

<hr style="border:2px dotted orange">

## Pipeline

=== "sdk"

    ::: runnable.Pipeline
        options:
            show_root_heading: true
            show_bases: false
            show_docstring_description: true
            heading_level: 3

=== "yaml"

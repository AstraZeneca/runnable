# Visualization Development

You are helping with visualization features in the Runnable framework. Focus on:

## Context
We want to visualise a run log from a run. The run logs are present in .run_log_store.

The executed pipelines are ranging from simple to composite workflows which have branches that are themselves pipelines.

### tasks of a pipeline.

An example of the run log is: worn-curie-1045

In the schema of the run log, there is metadata and the execution details of every step in the key steps.
The ```step_type``` of task is a simple task that could be python, notebook, shell.

For example:


```json
"steps": {
    "hello stub": {
        "name": "hello stub",
        "internal_name": "hello stub",
        "status": "SUCCESS",
        "step_type": "stub",
        "message": "",
        "mock": false,
        "code_identities": [
            {
                "code_identifier": "fa5a33f87c86d22340026073d0f814b226ede090",
                "code_identifier_type": "git",
                "code_identifier_dependable": false,
                "code_identifier_url": "https://github.com/AstraZeneca/runnable.git",
                "code_identifier_message": "changes found in .claude/commands/viz.md, df.csv"
            }
        ],
        "attempts": [
            {
                "attempt_number": 1,
                "start_time": "2025-11-10 10:45:53.169046",
                "end_time": "2025-11-10 10:45:53.169067",
                "status": "SUCCESS",
                "message": "",
                "input_parameters": {},
                "output_parameters": {},
                "user_defined_metrics": {}
            }
        ],
        "branches": {},
        "data_catalog": []
    },
    ...
```


This is one task with metadata which shows the code identities, attempts and duration of it. In every attempt, you
will see the input and output parameters and the data flow for files in get or put. Only parameters of ```kind``` json or metric are
human readable. The parameter of ```kind``` object are binary. REMEMBER this when trying to display the metadata. This only applies to ```step_type``` task.

In the same run_log, you will find a ```dag``` key which has the definition of the same task.

```json
"dag": {
    "start_at": "hello stub",
    "name": "",
    "description": "",
    "nodes": {
        "hello stub": {
            "node_type": "stub",
            "name": "hello stub",
            "next_node": "hello python",
            "on_failure": "",
            "overrides": {},
            "catalog": null,
            "max_attempts": 1
        },
    ...
```
You will find a one-to-one relation between ```nodes``` in the ```dag``` and ```steps```.

The steps are executed in chained as you can see in the dag definition - especially the ```next_node``` field in ```toss_task``` tells us the next node in the pipeline.


### Composite nodes

There are three node types that are composite.

- parallel - which has fixed branches. Look for the detail in the subsection parallel.
- map - which is one brach that is looped over an iterable. Look for the detail in map.
- conditional - which has many branches but only one of them gets executed based on a flag. Look for the detail in condition.

#### Parallel

Look for the example: hoary-brattain-1506

The parallel node is defined as this, it has a ```step_type``` parallel and it has branches in ```branches```.

```json
{
    "run_id": "hoary-brattain-1506",
    "dag_hash": "8ac293d7d4f6a25fe7e99d8f9a62e9c12fc7b6e4",
    "tag": "",
    "status": "SUCCESS",
    "steps": {
        "parallel_step": {
            "name": "parallel_step",
            "internal_name": "parallel_step",
            "status": "SUCCESS",
            "step_type": "parallel",
            "message": "",
            "mock": false,
            "code_identities": [
                {
                    "code_identifier": "30251b5ceb7dac5272f4776ba8401bbb27bda120",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": false,
                    "code_identifier_url": "https://github.com/AstraZeneca/runnable.git",
                    "code_identifier_message": "changes found in df.csv, runnable/gantt.py"
                }
            ],
            "attempts": [],
            "branches": {
                ...
            }
```

The corresponding section of the dag is:

```json
"nodes": {
    "parallel_step": {
        "node_type": "parallel",
        "name": "parallel_step",
        "is_composite": true,
        "next_node": "continue to",
        "on_failure": "",
        "overrides": {},
        "branches": {
            ...
        }
        ...
```

The branches are themselves pipelines with the similar structure to what you see in the tasks of the pipeline.

Here is the IMPORTANT detail, the ```internal_name``` that you see in a step of ```steps``` of the run_log is formatted as this:
```<composite_step_name>.<branch_name>.<step_name>```.

For example, here is the a snippet of one of the branch1 in ```dag```

```json
"branches": {
    "branch1": {
        "start_at": "hello stub",
        "name": "",
        "description": "",
        "nodes": {
            "hello stub": {
                "node_type": "stub",
                "name": "hello stub",
                "next_node": "hello python",
                "on_failure": "",
                "overrides": {},
                "catalog": null,
                "max_attempts": 1
            },
            ...
    }
```

and here is the snippet from ```steps```:

```json
"branches": {
    "parallel_step.branch1": {
        "internal_name": "parallel_step.branch1",
        "status": "SUCCESS",
        "steps": {
            "parallel_step.branch1.hello stub": {
                "name": "hello stub",
                "internal_name": "parallel_step.branch1.hello stub",
                "status": "SUCCESS",
                "step_type": "stub",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "30251b5ceb7dac5272f4776ba8401bbb27bda120",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": false,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable.git",
                        "code_identifier_message": "changes found in df.csv, runnable/gantt.py"
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2025-11-09 15:06:08.964979",
                        "end_time": "2025-11-09 15:06:08.964993",
                        "status": "SUCCESS",
                        "message": "",
                        "input_parameters": {},
                        "output_parameters": {},
                        "user_defined_metrics": {}
                    }
                ],
                "branches": {},
                "data_catalog": []
            },
```

Note that branches are named as ```<composite_node>.<branch_name>```, for example: ```parllel_step.branch1```
The steps in the branch are ```<composite_step_name>.<branch_name>.<step_name>```, for example ```internal_name``` is "parallel_step.branch1.hello stub"

The same logic applies for any kind of nesting.

Any branch is called: ```<composite_step_name>.<branch_name>.<step_name>.<branch_name>.<step_name> ``` and so on.


#### Map

Look into the example: strong-jepsen-1507

It has similar structure to parallel but the branch and step naming convention is a bit different.

Since we are looping over a iterable parameter: for example [a, b, c], or [1, 2, 3] etc
For a ```dag``` structure as below:

```json
"dag": {
    "start_at": "map_state",
    "name": "",
    "description": "",
    "nodes": {
        "map_state": {
            "node_type": "map",
            "name": "map_state",
            "is_composite": true,
            "next_node": "collect",
            "on_failure": "",
            "overrides": {},
            "iterate_on": "chunks",
            "iterate_as": "chunk",
            "reducer": null,
            "branch": {
                ...
            }

```

The steps would be:

```json
"steps": {
"map_state": {
    "name": "map_state",
    "internal_name": "map_state",
    "status": "SUCCESS",
    "step_type": "map",
    "message": "",
    "mock": false,
    "code_identities": [
        {
            "code_identifier": "30251b5ceb7dac5272f4776ba8401bbb27bda120",
            "code_identifier_type": "git",
            "code_identifier_dependable": false,
            "code_identifier_url": "https://github.com/AstraZeneca/runnable.git",
            "code_identifier_message": "changes found in df.csv, runnable/gantt.py"
        }
    ],
    "attempts": [],
    "branches": {
        ...
    }
...

```

And ```chunks``` is ```[1, 2, 3]```. As we are iterating over ```chunks```, the branches will be called:
```"map_state.1"```, ```"map_state.2``` , ```"map_state.3"```

Exactly like the parallel branch naming convention.

Here is a snippet of a step in the branch, ```"map_state.1"```:

```json
"steps": {
    "map_state.1.execute_python": {
        "name": "execute_python",
        "internal_name": "map_state.1.execute_python",
        "status": "SUCCESS",
        "step_type": "task",
        "message": "",
        "mock": false,
        "code_identities": [
            {
                "code_identifier": "30251b5ceb7dac5272f4776ba8401bbb27bda120",
                "code_identifier_type": "git",
                "code_identifier_dependable": false,
                "code_identifier_url": "https://github.com/AstraZeneca/runnable.git",
                "code_identifier_message": "changes found in df.csv, runnable/gantt.py"
            }
        ],
        "attempts": [
            {
                "attempt_number": 1,
                "start_time": "2025-11-09 15:07:59.754528",
                "end_time": "2025-11-09 15:07:59.762077",
                "status": "SUCCESS",
                "message": "",
            }
    },
    ...
```

The corresponding ```dag``` is:

```json
"branch": {
    "start_at": "execute_python",
    "name": "",
    "description": "",
    "nodes": {
        "execute_python": {
            "node_type": "task",
            "name": "execute_python",
            "next_node": "execute_notebook",
            "on_failure": "",
            "overrides": {},
            "catalog": null,
            "max_attempts": 1,
            "returns": [
                {
                    "name": "processed_python",
                    "kind": "json"
                }
            ],
            "secrets": [],
            "command_type": "python",
            "command": "examples.common.functions.process_chunk"
        },
        ...
```

Notice that the ```internal_name``` of the  step is: ```<composite_step_name>.<branch_name>.<step_name>```
where ```branch_name``` is one of the values of the iterable.



#### Conditional

Look into the example: augmenting-torvalds-1505

The ```dag``` is

```json
"nodes": {
    "toss_task": {
        "node_type": "task",
        "name": "toss_task",
        "next_node": "conditional",
        "on_failure": "",
        "overrides": {},
        "catalog": null,
        "max_attempts": 1,
        "returns": [
            {
                "name": "toss",
                "kind": "json"
            }
        ],
        "secrets": [],
        "command_type": "python",
        "command": "examples.02-sequential.conditional.toss_function"
    },
    "conditional": {
        "node_type": "conditional",
        "name": "conditional",
        "is_composite": true,
        "next_node": "continue to",
        "on_failure": "",
        "overrides": {},
        "parameter": "toss",
        "default": null,
        "branches": {
            "heads": {
                "start_at": "when_heads_task",
                "name": "",
                "description": "",
                "nodes": {
                    "when_heads_task": {
                        "node_type": "task",
                        "name": "when_heads_task",
                        "next_node": "success",
                        "on_failure": "",
                        "overrides": {},
                        "catalog": null,
                        "max_attempts": 1,
                        "returns": [],
                        "secrets": [],
                        "command_type": "python",
                        "command": "examples.02-sequential.conditional.when_heads_function"
                    },
                    "success": {
                        "node_type": "success",
                        "name": "success"
                    },
                    "fail": {
                        "node_type": "fail",
                        "name": "fail"
                    }
                }
            },
            "tails": {
                "start_at": "when_tails_task",
                "name": "",
                "description": "",
                "nodes": {
                    "when_tails_task": {
                        "node_type": "task",
                        "name": "when_tails_task",
                        "next_node": "success",
                        "on_failure": "",
                        "overrides": {},
                        "catalog": null,
                        "max_attempts": 1,
                        "returns": [],
                        "secrets": [],
                        "command_type": "python",
                        "command": "examples.02-sequential.conditional.when_tails_function"
                    },
                    "success": {
                        "node_type": "success",
                        "name": "success"
                    },
                    "fail": {
                        "node_type": "fail",
                        "name": "fail"
                    }
                }
            }
```

Only one of the branches is executed based on the parameter value of ```toss```. The toss could be something that the previous task has returned or available from the start of
the pipeline. If the ```toss``` value matches to the keys of the branch in ```branches``` it would invoke only that branch.

The steps of the same ```dag``` would be this:

```json
"conditional": {
    "name": "conditional",
    "internal_name": "conditional",
    "status": "SUCCESS",
    "step_type": "conditional",
    "message": "",
    "mock": false,
    "code_identities": [
        {
            "code_identifier": "30251b5ceb7dac5272f4776ba8401bbb27bda120",
            "code_identifier_type": "git",
            "code_identifier_dependable": false,
            "code_identifier_url": "https://github.com/AstraZeneca/runnable.git",
            "code_identifier_message": "changes found in df.csv, runnable/gantt.py"
        }
    ],
    "attempts": [],
    "branches": {
        "conditional.heads": {
            "internal_name": "conditional.heads",
            "status": "SUCCESS",
            "steps": {
                "conditional.heads.when_heads_task": {
```

Similar pattern from map or parallel, the branches are always: ```<composite_node>.<branch_name>```
And the steps are: ```<composite_node>.<branch_name>.<step_name>```


## Key Guidelines
- Design simple, lightweight visualization solutions
- Use Python API examples (not YAML) unless specifically requested
- Integrate with the core Pipeline and Task APIs from `runnable/`
- Leverage gant.py
- Avoid over-engineering - keep solutions minimal and focused

## Development Approach
1. Consider CLI-first solutions (text output, simple SVG generation)
2. Minimize dependencies - prefer Python standard library
3. Focus on developer experience and quick insights
4. Avoid complex web frameworks for simple visualization needs

## Documentation
- Update docs in `docs/` folder using mkdocs patterns
- Include code snippets from `examples/` directory
- Show contextual examples first, then detailed working examples
- Remember to add empty lines before markdown lists


## graph execution

Any runnable pipelines defined in examples folder, specifically in

├── 01-tasks
│   ├── notebook.py
│   ├── notebook.yaml
│   ├── python_task_as_pipeline.py
│   ├── python_tasks.py
│   ├── python_tasks.yaml
│   ├── scripts.py
│   ├── scripts.yaml
│   ├── stub.py
│   └── stub.yaml
├── 02-sequential
│   ├── conditional.py
│   ├── default_fail.py
│   ├── default_fail.yaml
│   ├── on_failure_fail.py
│   ├── on_failure_fail.yaml
│   ├── on_failure_succeed.py
│   ├── on_failure_succeed.yaml
│   ├── traversal.py
│   └── traversal.yaml
├── 03-parameters
│   ├── passing_parameters_notebook.py
│   ├── passing_parameters_notebook.yaml
│   ├── passing_parameters_python.py
│   ├── passing_parameters_python.yaml
│   ├── passing_parameters_shell.py
│   ├── passing_parameters_shell.yaml
│   ├── static_parameters_fail.py
│   ├── static_parameters_fail.yaml
│   ├── static_parameters_non_python.py
│   ├── static_parameters_non_python.yaml
│   ├── static_parameters_python.py
│   └── static_parameters_python.yaml
├── 04-catalog
│   ├── catalog_no_copy.py
│   ├── catalog_on_fail.py
│   ├── catalog_on_fail.yaml
│   ├── catalog_python.py
│   ├── catalog_python.yaml
│   └── catalog.py
├── 06-parallel
│   ├── nesting.py
│   ├── nesting.yaml
│   ├── parallel_branch_fail.py
│   ├── parallel_branch_fail.yaml
│   ├── parallel.py
│   └── parallel.yaml
├── 07-map
│   ├── custom_reducer.py
│   ├── custom_reducer.yaml
│   ├── map_fail.py
│   ├── map_fail.yaml
│   ├── map.py
│   └── map.yaml

can be executed and they produce run logs in .run_log_store.

Try to run a pipeline, go progressively from simple to complicated as marked by the number 01,02 etc
Inspect the run log. It should give you an idea of what happens in a runnable execution.

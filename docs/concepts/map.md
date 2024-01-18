```map``` nodes in magnus allows you to execute a sequence of nodes (i.e a pipeline) for all the items in a list. This is similar to
[Map state of AWS Step functions](https://docs.aws.amazon.com/step-functions/latest/dg/amazon-states-language-map-state.html) or
[loops in Argo workflows](https://argo-workflows.readthedocs.io/en/latest/walk-through/loops/).

Conceptually, map node can be represented in python like below.

```python
for i in iterable_parameter:
    # a pipeline of steps
    execute_first_step(i)
    execute_second_step(i)
    ...
```

You can control the parallelism by configuration of the executor.

## Example

Below is an example of processing a inventory of files (50) in parallel batches of 10 files per batch.
The ```stride``` parameter controls the chunk size and every batch is given the start index
of the files to process.

=== "visualization"

    The step "chunk files" identifies the number of files to process and computes the start index of every
    batch of files to process for a chunk size of 10, the stride.

    "Process Chunk" pipelines are then triggered in parallel to process the chunk of files between ```start index```
    and ```start index + stride```

    ```mermaid
    flowchart TD
    chunkify([Chunk files]):::green
    success([Success]):::green

    subgraph one[Process Chunk]
        process_chunk1([Process Chunk]):::yellow
        success_chunk1([Success]):::yellow

        process_chunk1 --> success_chunk1
    end

    subgraph two[Process Chunk]
        process_chunk2([Process Chunk]):::yellow
        success_chunk2([Success]):::yellow

        process_chunk2 --> success_chunk2
    end

    subgraph three[Process Chunk]
        process_chunk3([Process Chunk]):::yellow
        success_chunk3([Success]):::yellow

        process_chunk3 --> success_chunk3
    end

    subgraph four[Process Chunk]
        process_chunk4([Process Chunk]):::yellow
        success_chunk4([Success]):::yellow

        process_chunk4 --> success_chunk4
    end

    subgraph five[Process Chunk]
        process_chunk5([Process Chunk]):::yellow
        success_chunk5([Success]):::yellow

        process_chunk5 --> success_chunk5
    end



    chunkify -- (stride=10, start_index=0)--> one --> success
    chunkify -- (stride=10, start_index=10)--> two --> success
    chunkify -- (stride=10, start_index=20)--> three --> success
    chunkify -- (stride=10, start_index=30)--> four --> success
    chunkify -- (stride=10, start_index=40)--> five --> success

    classDef yellow stroke:#FFFF00
    classDef green stroke:#0f0
    ```

=== "python sdk"

    The ```start_index``` argument for the function ```process_chunk``` is dynamically set by iterating
    over ```chunks```.

    This instruction is set while defining the map node.

    ```python linenums="1" hl_lines="21 52-58"
    --8<-- "examples/concepts/map.py"
    ```


=== "pipeline in yaml"

    The ```start_index``` argument for the function ```process_chunk``` is dynamically set by iterating
    over ```chunks```.

    This instruction is set while defining the map node.
    Note that the ```branch``` of the map node has a similar schema of the pipeline.

    ```yaml linenums="1" hl_lines="22-23 25-36"
    --8<-- "examples/concepts/map.yaml"
    ```

=== "pipeline with shell tasks"

    The task ```chunk files``` sets the parameters ```stride``` and ```chunks``` similar to the python
    functions.

    The map branch "iterate and execute" iterates over chunks and exposes the current start_index of
    as environment variable ```MAGNUS_MAP_VARIABLE```.

    The environment variable ```MAGNUS_MAP_VARIABLE``` is a json string with keys of the ```iterate_as```.

    ```yaml linenums="1" hl_lines="23-24 38-40"
    --8<-- "examples/concepts/map_shell.yaml"
    ```

=== "Run log"

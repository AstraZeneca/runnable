# Visualization Development

You are helping with visualization features in the Runnable framework. Focus on:

## Context
- The previous complex web-based visualization system has been removed
- Current branch is `tracking-viz` (visualization tracking feature branch)
- Always use `uv run` for Python execution

## Key Guidelines
- Design simple, lightweight visualization solutions
- Use Python API examples (not YAML) unless specifically requested
- Integrate with the core Pipeline and Task APIs from `runnable/`
- Leverage the existing `graph.get_visualization_data()` function in `runnable/graph.py`
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


Remember: Keep visualization features lightweight, simple, and focused on core developer needs.

"""
Demonstration of Runnable's lightweight visualization capabilities.

This example shows how to use the visualization tools to:
1. Create and execute a pipeline
2. Visualize the pipeline structure
3. Generate execution reports
4. Export SVG diagrams
"""

from pathlib import Path
from examples.common.functions import hello, write_parameter, read_parameter
from runnable import Pipeline, PythonTask, pickled
from runnable.viz import PipelineVisualizer, visualize_pipeline, analyze_run_logs


def create_sample_pipeline():
    """Create a sample pipeline for visualization demonstration."""

    # Task 1: Generate some data
    write_task = PythonTask(
        name="generate_data",
        function=write_parameter,
        returns=[
            pickled("df"),
            pickled("integer"),
            pickled("floater"),
            pickled("stringer"),
            pickled("pydantic_param"),
            pickled("score")
        ]
    )

    # Task 2: Process the generated data
    read_task = PythonTask(
        name="process_data",
        function=read_parameter,
    )

    # Task 3: Say hello
    hello_task = PythonTask(
        name="greet",
        function=hello,
    )

    # Create pipeline with sequential execution
    pipeline = Pipeline(
        name="Visualization Demo Pipeline",
        description="A sample pipeline to demonstrate visualization features",
        steps=[write_task, read_task, hello_task]
    )

    return pipeline


def main():
    print("🎨 Runnable Visualization Demo")
    print("=" * 50)

    # Create and execute pipeline
    print("\n1. Creating and executing pipeline...")
    pipeline = create_sample_pipeline()

    # Show pipeline structure before execution
    print("\n2. Pipeline structure (before execution):")
    visualize_pipeline(pipeline.dag, output_format="console")

    # Execute the pipeline
    print("\n3. Executing pipeline...")
    result = pipeline.execute()

    # Get the run log path
    run_id = result.run_id
    run_log_path = Path(f".run_log_store/{run_id}.json")

    print(f"\n4. Pipeline executed! Run ID: {run_id}")

    # Show detailed visualization with execution data
    print("\n5. Pipeline visualization with execution results:")
    visualize_pipeline(pipeline.dag, run_log_path=run_log_path, output_format="console")

    # Generate SVG diagram
    print("\n6. Generating SVG diagram...")
    viz = PipelineVisualizer(graph=pipeline.dag, run_log_path=run_log_path)
    svg_output_path = Path("pipeline_diagram.svg")
    svg_content = viz.generate_simple_svg(output_path=svg_output_path)

    print(f"   SVG diagram saved to: {svg_output_path.absolute()}")

    # Analyze recent runs
    print("\n7. Recent run analysis:")
    analyze_run_logs(limit=5)

    print("\n✨ Visualization demo completed!")
    print("\nNext steps:")
    print("- Open pipeline_diagram.svg in a web browser to view the diagram")
    print("- Use 'uv run python -m runnable.viz analyze' to analyze run logs")
    print("- Integrate visualization into your own pipelines")


if __name__ == "__main__":
    main()

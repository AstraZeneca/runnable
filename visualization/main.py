import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rich import print

from runnable import context, graph

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="visualization/static"), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="visualization/templates")


# --- Simulate an API for graph data ---
def get_example_graph_data() -> Dict[str, List[Dict[str, Any]]]:
    """
    Generates example graph data with task types, metadata, and different node/edge types.
    """
    nodes = [
        {"id": "Start", "label": "parallel"},
        {
            "id": "Task1",
            "label": "task",
            "task_type": "python",
            "metadata": {
                "func": "process_data",
                "params": ["input_file"],
                "returns": ["clean_data"],
            },
        },
        {
            "id": "Task2",
            "label": "task",
            "task_type": "notebook",
            "metadata": {"path": "analysis.ipynb", "cell_count": 10},
        },
        {
            "id": "Task3",
            "label": "task",
            "task_type": "shell",
            "metadata": {"cmd": "ls -l /data", "user": "admin"},
        },
        {
            "id": "Task4",
            "label": "task",
            "task_type": "python",
            "metadata": {
                "func": "model_predict",
                "params": ["model_path", "test_data"],
                "returns": ["predictions"],
            },
        },
        {"id": "End", "label": "success"},
    ]

    links = [
        {"source": "Start", "target": "Task1"},
        {"source": "Start", "target": "Task2"},  # Parallel execution from Start
        {"source": "Task1", "target": "Task3"},
        {"source": "Task2", "target": "Task3"},
        {"source": "Task3", "target": "Task4"},
        {"source": "Task4", "target": "End"},
    ]
    return {"nodes": nodes, "links": links}


# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main D3 graph visualization page.
    """
    graph_data = get_graph_json_data()
    return templates.TemplateResponse(
        "index.html", {"request": request, "graph_data": graph_data}
    )


@app.get("/graph-data", response_model=Dict[str, List[Dict[str, Any]]])
async def get_graph_json_data():
    """
    API endpoint to return the raw graph data as JSON.
    (Optional, could be used for client-side fetching if not using Jinja2 for data injection)
    """
    source_file = "examples/02-sequential/traversal.py"
    function = "main"
    # return get_example_graph_data()
    return convert_graph_to_d3(source_file, function)


def convert_graph_to_d3(
    source_file: str, function: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Converts a graph representation to a format suitable for D3.js visualization.

    Args:
        source_file: The file path containing the graph definition.
        function: The function to visualize.

    Returns:
        A dictionary containing the nodes and links for D3.js.
    """

    context.run_context = "dummy"

    # Convert to Path object and resolve to absolute path
    file_path = Path(source_file).resolve()

    # Get module name from file name
    module_name = file_path.stem

    # Load module from file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    dag = getattr(module, function)().return_dag()
    d3_graph = graph.get_visualization_data(dag)

    print(d3_graph)
    return d3_graph

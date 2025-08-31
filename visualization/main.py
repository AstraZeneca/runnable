from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
        {"id": "Start", "label": "parallel", "group": "control"},
        {
            "id": "Task1",
            "label": "task",
            "task_type": "python",
            "metadata": {
                "func": "process_data",
                "params": ["input_file"],
                "returns": ["clean_data"],
            },
            "group": "task-python",
        },
        {
            "id": "Task2",
            "label": "task",
            "task_type": "notebook",
            "metadata": {"path": "analysis.ipynb", "cell_count": 10},
            "group": "task-notebook",
        },
        {
            "id": "Task3",
            "label": "task",
            "task_type": "shell",
            "metadata": {"cmd": "ls -l /data", "user": "admin"},
            "group": "task-shell",
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
            "group": "task-python",
        },
        {"id": "End", "label": "success", "group": "control"},
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
    graph_data = get_example_graph_data()
    return templates.TemplateResponse(
        "index.html", {"request": request, "graph_data": graph_data}
    )


@app.get("/graph-data", response_model=Dict[str, List[Dict[str, Any]]])
async def get_graph_json_data():
    """
    API endpoint to return the raw graph data as JSON.
    (Optional, could be used for client-side fetching if not using Jinja2 for data injection)
    """
    return get_example_graph_data()

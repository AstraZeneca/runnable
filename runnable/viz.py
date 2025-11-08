"""
Lightweight visualization utilities for Runnable pipelines.

This module provides simple, CLI-friendly visualization tools that avoid
complex web frameworks and focus on developer experience.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from runnable.graph import Graph


def _open_in_browser(file_path: Path) -> None:
    """
    Open a file in the user's default web browser.

    Args:
        file_path: Path to the file to open
    """
    import webbrowser

    try:
        file_url = file_path.as_uri()
        print(f"🌐 Opening in default browser: {file_path.name}")
        success = webbrowser.open(file_url)

        if not success:
            print("⚠️  Could not open browser automatically")
            print(f"🔗 Please open manually: {file_url}")

    except Exception as e:
        print(f"⚠️  Failed to open browser: {e}")
        print(f"🔗 Please open manually: file://{file_path}")


class PipelineVisualizer:
    """
    Simple visualizer for Runnable pipelines focusing on CLI output and SVG generation.
    """

    def __init__(
        self,
        graph: Optional[Graph] = None,
        run_log_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize visualizer with graph and/or run log data.

        Args:
            graph: Graph object representing pipeline structure
            run_log_path: Path to JSON run log file
        """
        self.graph = graph
        self.run_log_data = None

        if run_log_path:
            self.load_run_log(run_log_path)

    def load_run_log(self, run_log_path: Union[str, Path]) -> None:
        """Load run log data from JSON file."""
        path = Path(run_log_path)
        if path.exists():
            with open(path, "r") as f:
                self.run_log_data = json.load(f)
        else:
            raise FileNotFoundError(f"Run log not found: {run_log_path}")

    def print_ascii_dag(self) -> None:
        """Print ASCII representation of the pipeline DAG."""
        if not self.graph:
            print("No graph data available")
            return

        print("\n🔄 Pipeline Structure")
        print("=" * 50)

        # Get graph summary
        summary = self.graph.get_summary()
        nodes = {node["name"]: node for node in summary["nodes"]}

        # Start from the start_at node and traverse
        visited = set()
        self._print_node_tree(summary["start_at"], nodes, visited, indent=0)

        print("=" * 50)

    def print_dag_from_runlog(self) -> None:
        """Print ASCII representation of the pipeline DAG from run log data."""
        if not self.run_log_data:
            print("No run log data available")
            return

        print("\n🔄 Pipeline Structure (from execution log)")
        print("=" * 50)

        # Extract DAG info from run log
        dag_info = self.run_log_data.get("run_config", {}).get("dag", {})
        if not dag_info:
            print("No DAG information found in run log")
            return

        start_at = dag_info.get("start_at")
        nodes = dag_info.get("nodes", {})

        if not start_at or not nodes:
            print("Incomplete DAG information in run log")
            return

        # Build execution flow visualization
        visited = set()
        self._print_runlog_node_tree(start_at, nodes, visited, indent=0)

        print("=" * 50)

    def _print_runlog_node_tree(
        self, node_name: str, nodes: Dict[str, Any], visited: set, indent: int = 0
    ) -> None:
        """Print node tree from run log DAG data."""
        if node_name in visited or node_name not in nodes:
            return

        visited.add(node_name)
        node = nodes[node_name]

        # Create indent string
        prefix = "  " * indent

        # Choose icon based on node type
        icons = {
            "task": "⚙️ ",
            "stub": "📝",
            "success": "✅",
            "fail": "❌",
            "parallel": "🔀",
            "map": "🔁",
            "dag": "📊",
        }

        node_type = node.get("node_type", "task")
        icon = icons.get(node_type, "⚙️ ")

        # Get execution status and timing
        status_info = ""
        timing_info = ""
        if node_name in self.run_log_data.get("steps", {}):
            step_data = self.run_log_data["steps"][node_name]
            status = step_data.get("status", "UNKNOWN")

            # Status emoji
            status_emoji = (
                "✅" if status == "SUCCESS" else "❌" if status == "FAIL" else "⏸️ "
            )
            status_info = f" {status_emoji}"

            # Calculate duration
            attempts = step_data.get("attempts", [])
            if attempts:
                attempt = attempts[0]
                if "start_time" in attempt and "end_time" in attempt:
                    start = datetime.fromisoformat(attempt["start_time"])
                    end = datetime.fromisoformat(attempt["end_time"])
                    duration_ms = (end - start).total_seconds() * 1000
                    timing_info = f" ({duration_ms:.0f}ms)"

        # Print node with execution info
        print(f"{prefix}{icon} {node_name}{status_info}{timing_info}")

        # Determine actual next node based on execution status
        next_node = node.get("next_node")

        # Check if this step failed and should go to fail node
        if node_name in self.run_log_data.get("steps", {}):
            step_data = self.run_log_data["steps"][node_name]
            status = step_data.get("status", "UNKNOWN")

            if status == "FAIL":
                # Check if there's a custom on_failure handler
                on_failure = node.get("on_failure", "")
                if on_failure:
                    next_node = on_failure
                else:
                    # Default to fail node if no custom handler
                    next_node = "fail"

        # Continue traversal based on actual execution flow
        if next_node and next_node not in ["success", "fail"]:
            self._print_runlog_node_tree(next_node, nodes, visited, indent + 1)
        elif next_node in ["success", "fail"] and next_node in nodes:
            self._print_runlog_node_tree(next_node, nodes, visited, indent + 1)

    def _print_node_tree(
        self, node_name: str, nodes: Dict[str, Any], visited: set, indent: int = 0
    ) -> None:
        """Recursively print node tree structure."""
        if node_name in visited or node_name not in nodes:
            return

        visited.add(node_name)
        node = nodes[node_name]

        # Create indent string
        prefix = "  " * indent

        # Choose icon based on node type
        icons = {
            "task": "⚙️ ",
            "stub": "📝",
            "success": "✅",
            "fail": "❌",
            "parallel": "🔀",
            "map": "🔁",
            "dag": "📊",
        }

        icon = icons.get(node.get("node_type", "task"), "⚙️ ")

        # Print node with status if available
        status_indicator = ""
        if self.run_log_data and node_name in self.run_log_data.get("steps", {}):
            step_status = self.run_log_data["steps"][node_name]["status"]
            status_indicator = f" [{step_status}]"

        print(f"{prefix}{icon} {node_name}{status_indicator}")

        # Find next nodes (this is simplified - in real implementation would need to handle all connection types)
        next_node = node.get("next_node")
        if next_node and next_node != "success" and next_node != "fail":
            self._print_node_tree(next_node, nodes, visited, indent + 1)

    def print_execution_summary(self) -> None:
        """Print execution summary table."""
        if not self.run_log_data:
            print("No run log data available")
            return

        print(f"\n📊 Execution Summary - Run ID: {self.run_log_data['run_id']}")
        print("=" * 80)

        # Table header
        print(
            f"{'Step Name':<25} {'Status':<10} {'Duration':<12} {'Attempts':<8} {'Catalog':<12}"
        )
        print("-" * 92)

        for step_name, step_data in self.run_log_data.get("steps", {}).items():
            status = step_data["status"]
            attempts = len(step_data.get("attempts", []))

            # Calculate duration if attempt data is available
            duration = "N/A"
            if step_data.get("attempts"):
                attempt = step_data["attempts"][0]  # Use first attempt
                if "start_time" in attempt and "end_time" in attempt:
                    start = datetime.fromisoformat(attempt["start_time"])
                    end = datetime.fromisoformat(attempt["end_time"])
                    duration_ms = (end - start).total_seconds() * 1000
                    duration = f"{duration_ms:.1f}ms"

            # Status emoji
            status_emoji = (
                "✅" if status == "SUCCESS" else "❌" if status == "FAIL" else "⏸️ "
            )

            # Catalog operations summary
            catalog_data = step_data.get("data_catalog", [])
            put_ops = len([item for item in catalog_data if item.get("stage") == "put"])
            get_ops = len([item for item in catalog_data if item.get("stage") == "get"])
            catalog_info = ""
            if put_ops > 0 or get_ops > 0:
                catalog_info = f"📤{put_ops} 📥{get_ops}"
            else:
                catalog_info = "-"

            print(
                f"{step_name:<25} {status_emoji}{status:<9} {duration:<12} {attempts:<8} {catalog_info:<12}"
            )

        print("-" * 92)
        overall_status = self.run_log_data.get("status", "UNKNOWN")
        overall_emoji = "✅" if overall_status == "SUCCESS" else "❌"
        print(f"Overall Status: {overall_emoji} {overall_status}")

    def generate_simple_svg(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate a simple SVG representation of the pipeline.

        Args:
            output_path: Optional path to save SVG file

        Returns:
            SVG content as string
        """
        if not self.graph:
            return "<svg>No graph data available</svg>"

        summary = self.graph.get_summary()
        nodes = {node["name"]: node for node in summary["nodes"]}

        # Simple vertical layout
        svg_width = 400
        svg_height = len(nodes) * 80 + 100

        svg_content = f"""<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
    <style>
        .node {{ fill: #e1f5fe; stroke: #01579b; stroke-width: 2; }}
        .success {{ fill: #c8e6c9; stroke: #1b5e20; }}
        .fail {{ fill: #ffcdd2; stroke: #b71c1c; }}
        .text {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }}
        .title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }}
        .arrow {{ stroke: #666; stroke-width: 2; marker-end: url(#arrowhead); }}
    </style>

    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7"
                refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
        </marker>
    </defs>

    <text x="{svg_width//2}" y="30" class="title">Pipeline: {summary.get('name', 'Unnamed')}</text>
"""

        # Draw nodes and connections
        y_pos = 60
        node_positions = {}

        # First pass: position nodes
        for i, (node_name, node_data) in enumerate(nodes.items()):
            if node_data.get("node_type") in ["success", "fail"]:
                continue  # Handle terminal nodes separately

            node_positions[node_name] = {"x": svg_width // 2, "y": y_pos}
            y_pos += 80

        # Add terminal nodes at the bottom
        terminal_y = y_pos + 20
        if "success" in nodes:
            node_positions["success"] = {"x": svg_width // 2 - 80, "y": terminal_y}
        if "fail" in nodes:
            node_positions["fail"] = {"x": svg_width // 2 + 80, "y": terminal_y}

        # Second pass: draw nodes
        for node_name, pos in node_positions.items():
            node_data = nodes[node_name]
            node_type = node_data.get("node_type", "task")

            # Choose class based on type
            css_class = (
                "success"
                if node_type == "success"
                else "fail"
                if node_type == "fail"
                else "node"
            )

            # Add status info if available
            status_text = ""
            if self.run_log_data and node_name in self.run_log_data.get("steps", {}):
                status = self.run_log_data["steps"][node_name]["status"]
                status_text = f" ({status})"

            svg_content += f"""
    <rect x="{pos['x'] - 60}" y="{pos['y'] - 15}" width="120" height="30"
          rx="5" class="{css_class}" />
    <text x="{pos['x']}" y="{pos['y'] + 5}" class="text">{node_name}{status_text}</text>"""

        # Third pass: draw connections
        for node_name, node_data in nodes.items():
            if node_name not in node_positions:
                continue

            next_node = node_data.get("next_node")
            if next_node and next_node in node_positions:
                start_pos = node_positions[node_name]
                end_pos = node_positions[next_node]

                svg_content += f"""
    <line x1="{start_pos['x']}" y1="{start_pos['y'] + 15}"
          x2="{end_pos['x']}" y2="{end_pos['y'] - 15}" class="arrow" />"""

        svg_content += "\n</svg>"

        # Save to file if path provided
        if output_path:
            Path(output_path).write_text(svg_content)
            print(f"SVG saved to: {output_path}")

        return svg_content

    def generate_svg_from_runlog(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate SVG representation from run log data.

        Args:
            output_path: Optional path to save SVG file

        Returns:
            SVG content as string
        """
        if not self.run_log_data:
            return "<svg>No run log data available</svg>"

        # Extract DAG info from run log
        dag_info = self.run_log_data.get("run_config", {}).get("dag", {})
        if not dag_info:
            return "<svg>No DAG information found in run log</svg>"

        start_at = dag_info.get("start_at")
        nodes = dag_info.get("nodes", {})

        if not start_at or not nodes:
            return "<svg>Incomplete DAG information in run log</svg>"

        # Filter out success/fail nodes from main layout but count them
        main_nodes = {
            name: node
            for name, node in nodes.items()
            if node.get("node_type") not in ["success", "fail"]
        }
        terminal_nodes = {
            name: node
            for name, node in nodes.items()
            if node.get("node_type") in ["success", "fail"]
        }

        # Calculate SVG dimensions
        svg_width = 500
        svg_height = max(len(main_nodes) * 80 + 140, 300)

        # Get run info for title
        run_id = self.run_log_data.get("run_id", "unknown")
        pipeline_name = dag_info.get("name", "") or "Unnamed Pipeline"

        svg_content = f"""<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
    <style>
        .node {{ fill: #e1f5fe; stroke: #01579b; stroke-width: 2; }}
        .success {{ fill: #c8e6c9; stroke: #1b5e20; }}
        .fail {{ fill: #ffcdd2; stroke: #b71c1c; }}
        .executed {{ fill: #a5d6a7; stroke: #2e7d32; stroke-width: 3; }}
        .failed {{ fill: #ef9a9a; stroke: #c62828; stroke-width: 3; }}
        .text {{ font-family: Arial, sans-serif; font-size: 11px; text-anchor: middle; }}
        .status-text {{ font-family: Arial, sans-serif; font-size: 9px; text-anchor: middle; fill: #555; }}
        .title {{ font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; text-anchor: middle; }}
        .subtitle {{ font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; fill: #666; }}
        .arrow {{ stroke: #666; stroke-width: 2; marker-end: url(#arrowhead); }}
    </style>

    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7"
                refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
        </marker>
    </defs>

    <text x="{svg_width//2}" y="25" class="title">{pipeline_name}</text>
    <text x="{svg_width//2}" y="40" class="subtitle">Run ID: {run_id}</text>
"""

        # Position nodes in execution order
        y_pos = 70
        node_positions = {}

        # Order nodes by execution (start from start_at and follow next_node chain)
        ordered_nodes = []
        current = start_at
        visited = set()

        while current and current not in visited and current in nodes:
            if nodes[current].get("node_type") not in ["success", "fail"]:
                ordered_nodes.append(current)
            visited.add(current)
            current = nodes[current].get("next_node")

        # Position main execution nodes
        for node_name in ordered_nodes:
            node_positions[node_name] = {"x": svg_width // 2, "y": y_pos}
            y_pos += 70

        # Position terminal nodes at the bottom
        terminal_y = y_pos + 30
        terminal_x_positions = [svg_width // 2 - 80, svg_width // 2 + 80]
        for i, (node_name, node_data) in enumerate(terminal_nodes.items()):
            if i < len(terminal_x_positions):
                node_positions[node_name] = {
                    "x": terminal_x_positions[i],
                    "y": terminal_y,
                }

        # Draw nodes
        for node_name, pos in node_positions.items():
            node_data = nodes[node_name]
            node_type = node_data.get("node_type", "task")

            # Determine node style based on execution status
            css_class = "node"
            if node_type == "success":
                css_class = "success"
            elif node_type == "fail":
                css_class = "fail"
            elif node_name in self.run_log_data.get("steps", {}):
                step_status = self.run_log_data["steps"][node_name].get("status")
                if step_status == "SUCCESS":
                    css_class = "executed"
                elif step_status == "FAIL":
                    css_class = "failed"

            # Get timing info
            timing_text = ""
            if node_name in self.run_log_data.get("steps", {}):
                step_data = self.run_log_data["steps"][node_name]
                attempts = step_data.get("attempts", [])
                if attempts:
                    attempt = attempts[0]
                    if "start_time" in attempt and "end_time" in attempt:
                        start = datetime.fromisoformat(attempt["start_time"])
                        end = datetime.fromisoformat(attempt["end_time"])
                        duration_ms = (end - start).total_seconds() * 1000
                        timing_text = f"{duration_ms:.0f}ms"

            # Draw node rectangle
            svg_content += f"""
    <rect x="{pos['x'] - 70}" y="{pos['y'] - 20}" width="140" height="40"
          rx="5" class="{css_class}" />
    <text x="{pos['x']}" y="{pos['y'] - 2}" class="text">{node_name}</text>"""

            # Add timing info if available
            if timing_text:
                svg_content += f"""
    <text x="{pos['x']}" y="{pos['y'] + 12}" class="status-text">{timing_text}</text>"""

        # Draw connections
        for node_name, node_data in nodes.items():
            if node_name not in node_positions:
                continue

            next_node = node_data.get("next_node")

            # Check if this step failed and should go to fail node
            if node_name in self.run_log_data.get("steps", {}):
                step_data = self.run_log_data["steps"][node_name]
                status = step_data.get("status", "UNKNOWN")

                if status == "FAIL":
                    # Check if there's a custom on_failure handler
                    on_failure = node_data.get("on_failure", "")
                    if on_failure:
                        next_node = on_failure
                    else:
                        # Default to fail node if no custom handler
                        next_node = "fail"

            if next_node and next_node in node_positions:
                start_pos = node_positions[node_name]
                end_pos = node_positions[next_node]

                svg_content += f"""
    <line x1="{start_pos['x']}" y1="{start_pos['y'] + 20}"
          x2="{end_pos['x']}" y2="{end_pos['y'] - 20}" class="arrow" />"""

        svg_content += "\n</svg>"

        # Save to file if path provided
        if output_path:
            Path(output_path).write_text(svg_content)
            print(f"SVG saved to: {output_path}")

        return svg_content

    def generate_interactive_svg(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate an interactive SVG with hover tooltips and click details.

        Args:
            output_path: Optional path to save SVG file

        Returns:
            Interactive SVG content as string
        """
        if not self.run_log_data:
            return "<svg>No run log data available</svg>"

        # Use the basic SVG generation but with enhanced interactivity

        # Extract key information for enhanced features
        dag_info = self.run_log_data.get("run_config", {}).get("dag", {})
        nodes = dag_info.get("nodes", {})
        run_id = self.run_log_data.get("run_id", "unknown")

        # Replace the basic SVG with an enhanced interactive version
        svg_width = 600
        svg_height = max(len(nodes) * 100 + 200, 400)

        interactive_svg = f"""<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
    <style>
        .node {{
            fill: #e1f5fe;
            stroke: #01579b;
            stroke-width: 2;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .node:hover {{
            fill: #bbdefb;
            stroke-width: 3;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
        }}
        .success {{
            fill: #c8e6c9;
            stroke: #1b5e20;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .success:hover {{
            fill: #a5d6a7;
            stroke-width: 3;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
        }}
        .fail {{
            fill: #ffcdd2;
            stroke: #b71c1c;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .fail:hover {{
            fill: #ef9a9a;
            stroke-width: 3;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
        }}
        .executed {{
            fill: #a5d6a7;
            stroke: #2e7d32;
            stroke-width: 3;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .executed:hover {{
            fill: #81c784;
            stroke-width: 4;
            filter: drop-shadow(0 3px 6px rgba(0,0,0,0.4));
        }}
        .failed {{
            fill: #ef9a9a;
            stroke: #c62828;
            stroke-width: 3;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .failed:hover {{
            fill: #e57373;
            stroke-width: 4;
            filter: drop-shadow(0 3px 6px rgba(0,0,0,0.4));
        }}
        .text {{
            font-family: Arial, sans-serif;
            font-size: 11px;
            text-anchor: middle;
            pointer-events: none;
            fill: #333;
        }}
        .status-text {{
            font-family: Arial, sans-serif;
            font-size: 9px;
            text-anchor: middle;
            fill: #555;
            pointer-events: none;
        }}
        .title {{
            font-family: Arial, sans-serif;
            font-size: 16px;
            font-weight: bold;
            text-anchor: middle;
            fill: #1976d2;
        }}
        .subtitle {{
            font-family: Arial, sans-serif;
            font-size: 11px;
            text-anchor: middle;
            fill: #666;
        }}
        .arrow {{
            stroke: #666;
            stroke-width: 2;
            marker-end: url(#arrowhead);
        }}

        /* Info panel styles */
        .info-panel {{
            fill: #ffffff;
            stroke: #ddd;
            stroke-width: 2;
            opacity: 0;
            transition: opacity 0.3s ease;
            filter: drop-shadow(0 2px 8px rgba(0,0,0,0.15));
        }}
        .info-panel.show {{ opacity: 1; }}

        .info-text {{
            font-family: monospace;
            font-size: 10px;
            fill: #333;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        .info-text.show {{ opacity: 1; }}

        .info-title {{
            font-family: Arial, sans-serif;
            font-size: 12px;
            font-weight: bold;
            fill: #1976d2;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        .info-title.show {{ opacity: 1; }}

        .close-btn {{
            fill: #f44336;
            stroke: #d32f2f;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        .close-btn.show {{ opacity: 1; }}
        .close-btn:hover {{ fill: #e53935; }}
    </style>

    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7"
                refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
        </marker>
    </defs>

    <text x="{svg_width//2}" y="25" class="title">Interactive Pipeline Visualization</text>
    <text x="{svg_width//2}" y="45" class="subtitle">Run ID: {run_id} • Click nodes for detailed metadata</text>

    <!-- Info Panel (initially hidden) -->
    <rect id="info-panel" x="{svg_width - 320}" y="70" width="310" height="300" rx="8" class="info-panel" />
    <circle id="close-btn" cx="{svg_width - 25}" cy="85" r="8" class="close-btn" onclick="hideInfo()" />
    <text x="{svg_width - 25}" y="89" text-anchor="middle" font-family="Arial" font-size="10" fill="white" style="pointer-events: none;">×</text>
    <text id="info-title" x="{svg_width - 165}" y="95" class="info-title" text-anchor="middle">Node Details</text>
    <foreignObject id="info-content" x="{svg_width - 310}" y="105" width="300" height="250" class="info-text">
        <div xmlns="http://www.w3.org/1999/xhtml" style="padding: 5px; font-family: monospace; font-size: 10px; line-height: 1.4; color: #333;">
            Click a node to see detailed execution metadata, parameters, and timing information.
        </div>
    </foreignObject>
"""

        # Generate nodes with metadata - use the existing SVG generation approach
        dag_info = self.run_log_data.get("run_config", {}).get("dag", {})
        start_at = dag_info.get("start_at")
        nodes = dag_info.get("nodes", {})

        # Position nodes
        {
            name: node
            for name, node in nodes.items()
            if node.get("node_type") not in ["success", "fail"]
        }
        terminal_nodes = {
            name: node
            for name, node in nodes.items()
            if node.get("node_type") in ["success", "fail"]
        }

        y_pos = 80
        node_positions = {}

        # Order nodes by execution
        ordered_nodes = []
        current = start_at
        visited = set()

        while current and current not in visited and current in nodes:
            if nodes[current].get("node_type") not in ["success", "fail"]:
                ordered_nodes.append(current)
            visited.add(current)
            current = nodes[current].get("next_node")

        # Position main execution nodes
        for node_name in ordered_nodes:
            node_positions[node_name] = {"x": svg_width // 2, "y": y_pos}
            y_pos += 90

        # Position terminal nodes
        terminal_y = y_pos + 40
        terminal_x_positions = [svg_width // 2 - 100, svg_width // 2 + 100]
        for i, (node_name, node_data) in enumerate(terminal_nodes.items()):
            if i < len(terminal_x_positions):
                node_positions[node_name] = {
                    "x": terminal_x_positions[i],
                    "y": terminal_y,
                }

        # Draw interactive nodes
        for node_name, pos in node_positions.items():
            node_data = nodes[node_name]
            node_type = node_data.get("node_type", "task")

            # Determine node style based on execution status
            css_class = "node"
            if node_type == "success":
                css_class = "success"
            elif node_type == "fail":
                css_class = "fail"
            elif node_name in self.run_log_data.get("steps", {}):
                step_status = self.run_log_data["steps"][node_name].get("status")
                if step_status == "SUCCESS":
                    css_class = "executed"
                elif step_status == "FAIL":
                    css_class = "failed"

            # Get timing info
            timing_text = ""
            if node_name in self.run_log_data.get("steps", {}):
                step_data = self.run_log_data["steps"][node_name]
                attempts = step_data.get("attempts", [])
                if attempts:
                    attempt = attempts[0]
                    if "start_time" in attempt and "end_time" in attempt:
                        start = datetime.fromisoformat(attempt["start_time"])
                        end = datetime.fromisoformat(attempt["end_time"])
                        duration_ms = (end - start).total_seconds() * 1000
                        timing_text = f"{duration_ms:.0f}ms"

            # Draw interactive node
            interactive_svg += f"""
    <g class="node-group" data-node="{node_name}">
        <rect x="{pos['x'] - 90}" y="{pos['y'] - 30}" width="180" height="60"
              rx="10" class="{css_class}"
              onclick="showNodeInfo('{node_name}')" />
        <text x="{pos['x']}" y="{pos['y'] - 8}" class="text">{node_name}</text>"""

            if timing_text:
                interactive_svg += f"""
        <text x="{pos['x']}" y="{pos['y'] + 8}" class="status-text">{timing_text}</text>"""

            interactive_svg += """
    </g>"""

        # Draw connections
        for node_name, node_data in nodes.items():
            if node_name not in node_positions:
                continue

            next_node = node_data.get("next_node")

            # Check if this step failed and should go to fail node
            if node_name in self.run_log_data.get("steps", {}):
                step_data = self.run_log_data["steps"][node_name]
                status = step_data.get("status", "UNKNOWN")

                if status == "FAIL":
                    # Check if there's a custom on_failure handler
                    on_failure = node_data.get("on_failure", "")
                    if on_failure:
                        next_node = on_failure
                    else:
                        # Default to fail node if no custom handler
                        next_node = "fail"

            if next_node and next_node in node_positions:
                start_pos = node_positions[node_name]
                end_pos = node_positions[next_node]

                interactive_svg += f"""
    <line x1="{start_pos['x']}" y1="{start_pos['y'] + 30}"
          x2="{end_pos['x']}" y2="{end_pos['y'] - 30}" class="arrow" />"""

        interactive_svg += (
            """
    <script type="text/javascript">
    <![CDATA[
        window.nodeMetadata = {"""
            + self._generate_js_metadata()
            + """};

        function showNodeInfo(nodeName) {
            const panel = document.getElementById('info-panel');
            const title = document.getElementById('info-title');
            const content = document.getElementById('info-content');
            const closeBtn = document.getElementById('close-btn');

            if (panel && title && content && closeBtn) {
                panel.classList.add('show');
                title.classList.add('show');
                content.classList.add('show');
                closeBtn.classList.add('show');

                title.textContent = nodeName + ' Details';

                const metadata = window.nodeMetadata[nodeName] || {};
                const html = formatMetadata(metadata);
                content.firstElementChild.innerHTML = html;
            }
        }

        function hideInfo() {
            const elements = ['info-panel', 'info-title', 'info-content', 'close-btn'];
            elements.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.remove('show');
            });
        }

        function formatMetadata(data) {
            let html = '<div style="margin-bottom: 8px;"><strong>Execution Details:</strong></div>';

            if (data.status) {
                const statusColor = data.status === 'SUCCESS' ? '#4caf50' : data.status === 'FAIL' ? '#f44336' : '#ff9800';
                html += `<div>Status: <span style="color: ${statusColor}; font-weight: bold;">${data.status}</span></div>`;
            }

            if (data.duration_ms) {
                html += `<div>Duration: <strong>${data.duration_ms}</strong></div>`;
            }

            if (data.start_time) {
                html += `<div>Started: ${data.start_time}</div>`;
            }

            if (data.command) {
                html += '<div style="margin-top: 8px;"><strong>Command:</strong></div>';
                html += `<div style="background: #f5f5f5; padding: 4px; font-family: monospace; word-break: break-all; font-size: 9px;">${data.command}</div>`;
            }

            if (data.attempts > 1) {
                html += `<div style="margin-top: 4px;">Attempts: <strong>${data.attempts}</strong></div>`;
            }

            if (data.input_parameters && data.input_parameters.length > 0) {
                html += '<div style="margin-top: 8px;"><strong>Input Parameters:</strong></div>';
                html += '<div style="font-size: 9px;">' + data.input_parameters.join(', ') + '</div>';
            }

            if (data.output_parameters && data.output_parameters.length > 0) {
                html += '<div style="margin-top: 8px;"><strong>Output Parameters:</strong></div>';
                html += '<div style="font-size: 9px;">' + data.output_parameters.join(', ') + '</div>';
            }

            if (data.catalog_operations) {
                const catalog = data.catalog_operations;
                if (catalog.put_count > 0 || catalog.get_count > 0) {
                    html += '<div style="margin-top: 8px;"><strong>Data Catalog:</strong></div>';

                    if (catalog.put_count > 0) {
                        html += `<div style="font-size: 9px; margin-top: 4px;">📤 PUT (${catalog.put_count}): ${catalog.put_files.join(', ')}</div>`;
                    }

                    if (catalog.get_count > 0) {
                        html += `<div style="font-size: 9px; margin-top: 4px;">📥 GET (${catalog.get_count}): ${catalog.get_files.join(', ')}</div>`;
                    }
                }
            }

            return html;
        }

        // Click outside to close
        document.addEventListener('click', function(e) {
            if (!e.target.closest('#info-panel') && !e.target.closest('[onclick*="showNodeInfo"]')) {
                hideInfo();
            }
        });
    ]]>
    </script>
</svg>"""
        )

        # Save to file if path provided
        if output_path:
            Path(output_path).write_text(interactive_svg)
            print(f"Interactive SVG saved to: {output_path}")
            print("💡 Open in a web browser for full interactivity")

        return interactive_svg

    def generate_html_dashboard(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate a rich HTML dashboard with advanced interactivity.

        This method creates a comprehensive HTML report with embedded visualizations,
        timeline charts, and detailed execution analysis.

        Args:
            output_path: Optional path to save HTML file

        Returns:
            HTML content as string
        """
        if not self.run_log_data:
            return "<html><body>No run log data available</body></html>"

        # Extract run information
        run_id = self.run_log_data.get("run_id", "unknown")
        status = self.run_log_data.get("status", "UNKNOWN")
        self.run_log_data.get("run_config", {}).get("dag", {})
        steps = self.run_log_data.get("steps", {})

        # Generate timeline data
        self._generate_timeline_data()
        metadata_json = self._generate_js_metadata()

        # Create HTML dashboard
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Dashboard - {run_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #f8fafc;
            color: #1e293b;
            line-height: 1.6;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 0;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }}

        .dashboard {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin: 2rem auto;
            max-width: 1200px;
            padding: 0 1rem;
        }}

        .card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
        }}

        .card h3 {{
            color: #1e293b;
            margin-bottom: 1rem;
            font-size: 1.25rem;
            font-weight: 600;
        }}

        .pipeline-viz {{
            grid-column: 1 / -1;
            padding: 2rem;
        }}

        .node {{
            fill: #e1f5fe;
            stroke: #01579b;
            stroke-width: 2;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .node:hover {{
            fill: #bbdefb;
            stroke-width: 3;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
        }}

        .executed {{ fill: #a5d6a7; stroke: #2e7d32; stroke-width: 3; }}
        .failed {{ fill: #ef9a9a; stroke: #c62828; stroke-width: 3; }}
        .success {{ fill: #c8e6c9; stroke: #1b5e20; }}
        .fail {{ fill: #ffcdd2; stroke: #b71c1c; }}

        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid #f1f5f9;
        }}

        .metric:last-child {{ border-bottom: none; }}

        .metric-label {{
            font-weight: 500;
            color: #475569;
        }}

        .metric-value {{
            font-weight: 600;
            color: #1e293b;
        }}

        .status-success {{ color: #16a34a; }}
        .status-failed {{ color: #dc2626; }}
        .status-unknown {{ color: #f59e0b; }}

        /* Metadata Styles */
        .metadata-item {{
            margin: 1rem 0;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.2s ease;
        }}

        .metadata-item:hover {{
            border-color: #cbd5e1;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .metadata-item.success {{ border-left: 4px solid #16a34a; }}
        .metadata-item.failed {{ border-left: 4px solid #dc2626; }}
        .metadata-item.default {{ border-left: 4px solid #3b82f6; }}

        .metadata-header {{
            display: flex;
            align-items: center;
            padding: 1rem;
            background: #f8fafc;
            border-bottom: 1px solid #e2e8f0;
        }}

        .metadata-icon {{
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1rem;
            font-weight: bold;
            color: white;
            font-size: 14px;
        }}

        .metadata-icon.success {{ background: #16a34a; }}
        .metadata-icon.failed {{ background: #dc2626; }}
        .metadata-icon.default {{ background: #3b82f6; }}

        .metadata-title h4 {{
            margin: 0;
            color: #1e293b;
            font-size: 1.1rem;
        }}

        .metadata-subtitle {{
            font-size: 0.875rem;
            color: #64748b;
            margin-top: 0.25rem;
        }}

        .metadata-content {{
            padding: 1rem;
        }}

        .metadata-section {{
            margin-bottom: 1rem;
        }}

        .metadata-section:last-child {{
            margin-bottom: 0;
        }}

        .command-code {{
            background: #f1f5f9;
            padding: 0.5rem;
            border-radius: 4px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.875rem;
            display: block;
            margin-top: 0.5rem;
            word-break: break-all;
        }}

        .params-list {{
            margin-top: 0.5rem;
        }}

        .param-item {{
            display: flex;
            align-items: center;
            padding: 0.5rem;
            margin: 0.25rem 0;
            background: #f8fafc;
            border-radius: 4px;
            font-size: 0.875rem;
        }}

        .param-name {{
            font-weight: 600;
            color: #1e293b;
            margin-right: 0.5rem;
            min-width: 100px;
        }}

        .param-type {{
            background: #e0e7ff;
            color: #3730a3;
            padding: 0.125rem 0.375rem;
            border-radius: 3px;
            font-size: 0.75rem;
            margin-right: 0.5rem;
            font-weight: 500;
        }}

        .param-value {{
            color: #475569;
            flex: 1;
            word-break: break-word;
        }}

        .no-params {{
            color: #9ca3af;
            font-style: italic;
        }}

        /* Catalog Styles */
        .catalog-list {{
            margin-top: 0.5rem;
        }}

        .catalog-group {{
            margin: 0.75rem 0 0.25rem 0;
            font-size: 0.9rem;
            color: #374151;
        }}

        .catalog-item {{
            display: flex;
            align-items: center;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 4px;
            font-size: 0.875rem;
            border-left: 3px solid;
        }}

        .catalog-item.put-operation {{
            background: #fef3c7;
            border-left-color: #f59e0b;
        }}

        .catalog-item.get-operation {{
            background: #dbeafe;
            border-left-color: #3b82f6;
        }}

        .catalog-name {{
            font-weight: 600;
            color: #1e293b;
            margin-right: 0.5rem;
            flex: 1;
        }}

        .catalog-hash {{
            background: #f3f4f6;
            color: #6b7280;
            padding: 0.125rem 0.375rem;
            border-radius: 3px;
            font-size: 0.75rem;
            font-family: monospace;
            margin-right: 0.5rem;
        }}

        .catalog-path {{
            color: #64748b;
            font-size: 0.75rem;
            font-family: monospace;
            opacity: 0.8;
        }}

        .metadata-technical {{
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
        }}

        .tech-detail {{
            font-size: 0.875rem;
            color: #64748b;
        }}

        /* Parameters Flow Styles */
        .flow-section {{
            margin-bottom: 2rem;
        }}

        .flow-section h4 {{
            color: #1e293b;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }}

        .final-params {{
            display: grid;
            gap: 0.5rem;
        }}

        .final-param-item {{
            display: flex;
            align-items: center;
            padding: 0.75rem;
            background: #f0f9ff;
            border: 1px solid #e0f2fe;
            border-radius: 6px;
        }}

        .catalog-items {{
            display: grid;
            gap: 0.5rem;
        }}

        .catalog-item {{
            display: flex;
            align-items: center;
            padding: 0.75rem;
            background: #fefce8;
            border: 1px solid #fef3c7;
            border-radius: 6px;
        }}

        .catalog-name {{
            font-weight: 600;
            margin-right: 0.5rem;
        }}

        .catalog-step {{
            color: #64748b;
            font-size: 0.875rem;
            margin-right: 0.5rem;
        }}

        .catalog-type {{
            background: #fbbf24;
            color: #92400e;
            padding: 0.125rem 0.375rem;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: 500;
        }}

        .expandable {{
            cursor: pointer;
            user-select: none;
        }}

        .expandable-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}

        .expandable-content.expanded {{
            max-height: 500px;
        }}

        @media (max-width: 768px) {{
            .dashboard {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>Pipeline Execution Dashboard</h1>
            <p>Run ID: {run_id} • Status: <span class="status-{status.lower()}">{status}</span></p>
        </div>
    </div>

    <div class="dashboard">
        <!-- Pipeline Visualization -->
        <div class="card pipeline-viz">
            <h3>Pipeline Structure</h3>
            <div id="pipeline-svg">
                {self._generate_embedded_svg()}
            </div>
        </div>

        <!-- Execution Metrics -->
        <div class="card">
            <h3>Execution Metrics</h3>
            <div class="metric">
                <span class="metric-label">Total Steps</span>
                <span class="metric-value">{len(steps)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Success Rate</span>
                <span class="metric-value">{self._calculate_success_rate()}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Duration</span>
                <span class="metric-value">{self._calculate_total_duration()}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Pipeline Status</span>
                <span class="metric-value status-{status.lower()}">{status}</span>
            </div>
        </div>

        <!-- Step Details with Rich Metadata -->
        <div class="card">
            <h3>Step Details & Metadata</h3>
            <div id="step-details">
                {self._generate_rich_metadata_html()}
            </div>
        </div>

        <!-- Parameters & Data Flow -->
        <div class="card">
            <h3>Parameters & Data Flow</h3>
            <div id="parameters-flow">
                {self._generate_parameters_flow_html()}
            </div>
        </div>
    </div>

    <script>
        // Node metadata for interactivity
        window.nodeMetadata = {metadata_json};

        // Enhanced node interaction
        function showNodeInfo(nodeName) {{
            const metadata = window.nodeMetadata[nodeName] || {{}};
            const detailsDiv = document.getElementById('step-details');

            // Highlight selected step
            const items = detailsDiv.querySelectorAll('.timeline-item');
            items.forEach(item => item.style.background = '#f8fafc');

            const targetItem = Array.from(items).find(item =>
                item.textContent.includes(nodeName)
            );
            if (targetItem) {{
                targetItem.style.background = '#dbeafe';
                targetItem.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}

        // Highlight step from embedded SVG
        function highlightStep(nodeName) {{
            console.log('Highlighting step:', nodeName);

            // Reset all metadata items
            const metadataItems = document.querySelectorAll('.metadata-item');
            metadataItems.forEach(item => {{
                item.style.background = '';
                item.style.border = '';
            }});

            // Find and highlight the target step
            const targetStep = document.getElementById(`step-${{nodeName}}`);
            if (targetStep) {{
                targetStep.style.background = '#f0f9ff';
                targetStep.style.border = '2px solid #0ea5e9';
                targetStep.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}

            // Also highlight in parameters flow if present
            const flowItems = document.querySelectorAll('.final-param-item, .catalog-item');
            flowItems.forEach(item => {{
                if (item.textContent.includes(nodeName)) {{
                    item.style.background = '#f0f9ff';
                    item.style.border = '1px solid #0ea5e9';
                }}
            }});
        }}

        // Expandable sections
        document.addEventListener('click', function(e) {{
            if (e.target.classList.contains('expandable')) {{
                const content = e.target.nextElementSibling;
                content.classList.toggle('expanded');
                e.target.textContent = content.classList.contains('expanded')
                    ? e.target.textContent.replace('▶', '▼')
                    : e.target.textContent.replace('▼', '▶');
            }}
        }});
    </script>
</body>
</html>"""

        # Save to file if path provided
        if output_path:
            Path(output_path).write_text(html_content)
            print(f"HTML dashboard saved to: {output_path}")
            print("💡 Open in a web browser for full interactivity")

        return html_content

    def _generate_timeline_data(self) -> List[Dict[str, Any]]:
        """Generate timeline data for visualization."""
        timeline = []
        if not self.run_log_data:
            return timeline

        steps = self.run_log_data.get("steps", {})
        for step_name, step_data in steps.items():
            attempts = step_data.get("attempts", [])
            if attempts:
                attempt = attempts[0]
                timeline.append(
                    {
                        "name": step_name,
                        "status": step_data.get("status", "UNKNOWN"),
                        "start_time": attempt.get("start_time", ""),
                        "end_time": attempt.get("end_time", ""),
                        "duration": self._calculate_step_duration(attempt),
                    }
                )

        # Sort by start time
        timeline.sort(key=lambda x: x.get("start_time", ""))
        return timeline

    def _calculate_success_rate(self) -> int:
        """Calculate success rate of steps."""
        if not self.run_log_data:
            return 0

        steps = self.run_log_data.get("steps", {})
        if not steps:
            return 0

        successful = sum(
            1 for step in steps.values() if step.get("status") == "SUCCESS"
        )
        return int((successful / len(steps)) * 100)

    def _calculate_total_duration(self) -> str:
        """Calculate total pipeline duration."""
        if not self.run_log_data:
            return "N/A"

        steps = self.run_log_data.get("steps", {})
        all_times = []

        for step_data in steps.values():
            for attempt in step_data.get("attempts", []):
                if "start_time" in attempt:
                    all_times.append(datetime.fromisoformat(attempt["start_time"]))
                if "end_time" in attempt:
                    all_times.append(datetime.fromisoformat(attempt["end_time"]))

        if len(all_times) >= 2:
            total_duration_ms = (max(all_times) - min(all_times)).total_seconds() * 1000
            return f"{total_duration_ms:.0f}ms"

        return "N/A"

    def _calculate_step_duration(self, attempt: Dict[str, Any]) -> str:
        """Calculate duration for a single step attempt."""
        if "start_time" in attempt and "end_time" in attempt:
            start = datetime.fromisoformat(attempt["start_time"])
            end = datetime.fromisoformat(attempt["end_time"])
            duration_ms = (end - start).total_seconds() * 1000
            return f"{duration_ms:.1f}ms"
        return "N/A"

    def _generate_rich_metadata_html(self) -> str:
        """Generate HTML for rich step metadata display."""
        if not self.run_log_data:
            return "<p>No step data available</p>"

        steps = self.run_log_data.get("steps", {})
        dag_nodes = (
            self.run_log_data.get("run_config", {}).get("dag", {}).get("nodes", {})
        )
        html_parts = []

        for step_name, step_data in steps.items():
            if step_data.get("step_type") in ["success", "fail"]:
                continue  # Skip terminal nodes

            status = step_data.get("status", "UNKNOWN")
            status_class = (
                "success"
                if status == "SUCCESS"
                else "failed"
                if status == "FAIL"
                else "default"
            )
            step_type = step_data.get("step_type", "unknown")

            # Get command info from DAG
            dag_node = dag_nodes.get(step_name, {})
            command = dag_node.get("command", "")
            command_type = dag_node.get("command_type", "")

            # Get execution info
            attempts = step_data.get("attempts", [])
            duration = "N/A"
            if attempts:
                duration = self._calculate_step_duration(attempts[0])

            # Get parameter info
            input_params = {}
            output_params = {}
            if attempts:
                attempt = attempts[0]
                input_params = attempt.get("input_parameters", {})
                output_params = attempt.get("output_parameters", {})

            # Get catalog info
            catalog_data = step_data.get("data_catalog", [])

            html_parts.append(f"""
                <div class="metadata-item {status_class}" id="step-{step_name}">
                    <div class="metadata-header">
                        <div class="metadata-icon {status_class}">
                            {'✓' if status == 'SUCCESS' else '✗' if status == 'FAIL' else '?'}
                        </div>
                        <div class="metadata-title">
                            <h4>{step_name}</h4>
                            <div class="metadata-subtitle">
                                {command_type.upper()} • {duration} • {status}
                            </div>
                        </div>
                    </div>

                    <div class="metadata-content">
                        <div class="metadata-section">
                            <strong>Command:</strong>
                            <code class="command-code">{command}</code>
                        </div>

                        {self._format_parameters_section("Input Parameters", input_params)}
                        {self._format_parameters_section("Output Parameters", output_params)}
                        {self._format_catalog_section(catalog_data)}

                        <div class="metadata-technical">
                            <span class="tech-detail">Type: {step_type}</span>
                            <span class="tech-detail">Attempts: {len(attempts)}</span>
                            <span class="tech-detail">Duration: {duration}</span>
                        </div>
                    </div>
                </div>
            """)

        return "".join(html_parts)

    def _format_parameters_section(self, title: str, params: dict) -> str:
        """Format parameters section with proper styling."""
        if not params:
            return f"""
                <div class="metadata-section">
                    <strong>{title}:</strong>
                    <span class="no-params">None</span>
                </div>
            """

        param_items = []
        for name, param_info in params.items():
            if isinstance(param_info, dict):
                param_type = param_info.get("kind", "unknown")
                param_value = param_info.get("value", "")
                param_desc = param_info.get("description", "")

                # Format value for display
                if param_type == "object":
                    display_value = f"Object: {param_desc}"
                elif param_type == "json":
                    display_value = str(param_value)
                elif param_type == "metric":
                    display_value = f"Metric: {param_value}"
                else:
                    display_value = str(param_value)

                param_items.append(f"""
                    <div class="param-item">
                        <span class="param-name">{name}</span>
                        <span class="param-type">[{param_type}]</span>
                        <span class="param-value">{display_value}</span>
                    </div>
                """)
            else:
                param_items.append(f"""
                    <div class="param-item">
                        <span class="param-name">{name}</span>
                        <span class="param-value">{str(param_info)}</span>
                    </div>
                """)

        return f"""
            <div class="metadata-section">
                <strong>{title}:</strong>
                <div class="params-list">
                    {''.join(param_items)}
                </div>
            </div>
        """

    def _format_catalog_section(self, catalog_data: list) -> str:
        """Format catalog section with get/put operations."""
        if not catalog_data:
            return """
                <div class="metadata-section">
                    <strong>Data Catalog:</strong>
                    <span class="no-params">No catalog operations</span>
                </div>
            """

        # Group by operation type
        put_operations = [item for item in catalog_data if item.get("stage") == "put"]
        get_operations = [item for item in catalog_data if item.get("stage") == "get"]

        catalog_items = []

        if put_operations:
            catalog_items.append(
                '<div class="catalog-group"><strong>📤 PUT Operations:</strong></div>'
            )
            for item in put_operations:
                name = item.get("name", "unknown")
                data_hash = item.get("data_hash", "")[:8]
                catalog_path = item.get("catalog_relative_path", "")

                catalog_items.append(f"""
                    <div class="catalog-item put-operation">
                        <span class="catalog-name">{name}</span>
                        <span class="catalog-hash">#{data_hash}...</span>
                        <span class="catalog-path">{catalog_path}</span>
                    </div>
                """)

        if get_operations:
            catalog_items.append(
                '<div class="catalog-group"><strong>📥 GET Operations:</strong></div>'
            )
            for item in get_operations:
                name = item.get("name", "unknown")
                data_hash = item.get("data_hash", "")[:8]
                catalog_path = item.get("catalog_relative_path", "")

                catalog_items.append(f"""
                    <div class="catalog-item get-operation">
                        <span class="catalog-name">{name}</span>
                        <span class="catalog-hash">#{data_hash}...</span>
                        <span class="catalog-path">{catalog_path}</span>
                    </div>
                """)

        return f"""
            <div class="metadata-section">
                <strong>Data Catalog:</strong>
                <div class="catalog-list">
                    {''.join(catalog_items)}
                </div>
            </div>
        """

    def _generate_parameters_flow_html(self) -> str:
        """Generate HTML showing parameter flow between steps."""
        if not self.run_log_data:
            return "<p>No parameter flow data available</p>"

        steps = self.run_log_data.get("steps", {})
        final_params = self.run_log_data.get("parameters", {})

        # Create flow visualization
        html_parts = []

        # Show final parameters at pipeline level
        if final_params:
            html_parts.append("""
                <div class="flow-section">
                    <h4>Final Pipeline Parameters</h4>
                    <div class="final-params">
            """)

            for name, param_info in final_params.items():
                if isinstance(param_info, dict):
                    param_type = param_info.get("kind", "unknown")
                    param_value = param_info.get(
                        "description", param_info.get("value", "")
                    )

                    html_parts.append(f"""
                        <div class="final-param-item">
                            <span class="param-name">{name}</span>
                            <span class="param-type">[{param_type}]</span>
                            <span class="param-value">{param_value}</span>
                        </div>
                    """)

            html_parts.append("</div></div>")

        # Show data catalog if available
        catalog_items = []
        for step_name, step_data in steps.items():
            data_catalog = step_data.get("data_catalog", [])
            if data_catalog:
                for item in data_catalog:
                    catalog_items.append(
                        {
                            "step": step_name,
                            "name": item.get("name", ""),
                            "type": item.get("type", ""),
                            "location": item.get("location", ""),
                        }
                    )

        if catalog_items:
            html_parts.append("""
                <div class="flow-section">
                    <h4>Data Catalog</h4>
                    <div class="catalog-items">
            """)

            for item in catalog_items:
                html_parts.append(f"""
                    <div class="catalog-item">
                        <span class="catalog-name">{item["name"]}</span>
                        <span class="catalog-step">from {item["step"]}</span>
                        <span class="catalog-type">[{item["type"]}]</span>
                    </div>
                """)

            html_parts.append("</div></div>")

        if not html_parts:
            return "<p>No parameter flow or catalog data found</p>"

        return "".join(html_parts)

    def _generate_embedded_svg(self) -> str:
        """Generate a simplified SVG for embedding in HTML dashboard."""
        if not self.run_log_data:
            return "<p>No pipeline data available</p>"

        # Extract DAG info from run log
        dag_info = self.run_log_data.get("run_config", {}).get("dag", {})
        if not dag_info:
            return "<p>No DAG information found</p>"

        start_at = dag_info.get("start_at")
        nodes = dag_info.get("nodes", {})

        if not start_at or not nodes:
            return "<p>Incomplete DAG information</p>"

        # Create a simpler SVG for HTML embedding
        svg_width = 500
        svg_height = max(len(nodes) * 80 + 100, 300)

        svg_content = f"""<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
    <style>
        .embedded-node {{
            fill: #e3f2fd;
            stroke: #1976d2;
            stroke-width: 2;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .embedded-node:hover {{
            fill: #bbdefb;
            stroke-width: 3;
        }}
        .embedded-executed {{
            fill: #c8e6c9;
            stroke: #388e3c;
            stroke-width: 2;
        }}
        .embedded-failed {{
            fill: #ffcdd2;
            stroke: #d32f2f;
            stroke-width: 2;
        }}
        .embedded-success {{
            fill: #a5d6a7;
            stroke: #2e7d32;
        }}
        .embedded-fail {{
            fill: #ef9a9a;
            stroke: #c62828;
        }}
        .embedded-text {{
            font-family: Arial, sans-serif;
            font-size: 11px;
            text-anchor: middle;
            fill: #333;
            pointer-events: none;
        }}
        .embedded-arrow {{
            stroke: #666;
            stroke-width: 2;
            marker-end: url(#embedded-arrowhead);
        }}
    </style>

    <defs>
        <marker id="embedded-arrowhead" markerWidth="8" markerHeight="6"
                refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#666" />
        </marker>
    </defs>"""

        # Position nodes
        {
            name: node
            for name, node in nodes.items()
            if node.get("node_type") not in ["success", "fail"]
        }
        terminal_nodes = {
            name: node
            for name, node in nodes.items()
            if node.get("node_type") in ["success", "fail"]
        }

        y_pos = 40
        node_positions = {}

        # Order nodes by execution
        ordered_nodes = []
        current = start_at
        visited = set()

        while current and current not in visited and current in nodes:
            if nodes[current].get("node_type") not in ["success", "fail"]:
                ordered_nodes.append(current)
            visited.add(current)
            current = nodes[current].get("next_node")

        # Position main execution nodes
        for node_name in ordered_nodes:
            node_positions[node_name] = {"x": svg_width // 2, "y": y_pos}
            y_pos += 70

        # Position terminal nodes
        terminal_y = y_pos + 20
        terminal_x_positions = [svg_width // 2 - 80, svg_width // 2 + 80]
        for i, (node_name, node_data) in enumerate(terminal_nodes.items()):
            if i < len(terminal_x_positions):
                node_positions[node_name] = {
                    "x": terminal_x_positions[i],
                    "y": terminal_y,
                }

        # Draw nodes
        for node_name, pos in node_positions.items():
            node_data = nodes[node_name]
            node_type = node_data.get("node_type", "task")

            # Determine node style based on execution status
            css_class = "embedded-node"
            if node_type == "success":
                css_class = "embedded-success"
            elif node_type == "fail":
                css_class = "embedded-fail"
            elif node_name in self.run_log_data.get("steps", {}):
                step_status = self.run_log_data["steps"][node_name].get("status")
                if step_status == "SUCCESS":
                    css_class = "embedded-executed"
                elif step_status == "FAIL":
                    css_class = "embedded-failed"

            # Get timing info
            timing_text = ""
            if node_name in self.run_log_data.get("steps", {}):
                step_data = self.run_log_data["steps"][node_name]
                attempts = step_data.get("attempts", [])
                if attempts:
                    attempt = attempts[0]
                    if "start_time" in attempt and "end_time" in attempt:
                        start = datetime.fromisoformat(attempt["start_time"])
                        end = datetime.fromisoformat(attempt["end_time"])
                        duration_ms = (end - start).total_seconds() * 1000
                        timing_text = f" ({duration_ms:.0f}ms)"

            # Draw node
            svg_content += f"""
    <g onclick="highlightStep('{node_name}')">
        <rect x="{pos['x'] - 70}" y="{pos['y'] - 20}" width="140" height="40"
              rx="6" class="{css_class}" />
        <text x="{pos['x']}" y="{pos['y'] - 2}" class="embedded-text">{node_name}</text>
        <text x="{pos['x']}" y="{pos['y'] + 12}" class="embedded-text" style="font-size: 9px; fill: #666;">{timing_text}</text>
    </g>"""

        # Draw connections
        for node_name, node_data in nodes.items():
            if node_name not in node_positions:
                continue

            next_node = node_data.get("next_node")

            # Check if this step failed and should go to fail node
            if node_name in self.run_log_data.get("steps", {}):
                step_data = self.run_log_data["steps"][node_name]
                status = step_data.get("status", "UNKNOWN")

                if status == "FAIL":
                    # Check if there's a custom on_failure handler
                    on_failure = node_data.get("on_failure", "")
                    if on_failure:
                        next_node = on_failure
                    else:
                        # Default to fail node if no custom handler
                        next_node = "fail"

            if next_node and next_node in node_positions:
                start_pos = node_positions[node_name]
                end_pos = node_positions[next_node]

                svg_content += f"""
    <line x1="{start_pos['x']}" y1="{start_pos['y'] + 20}"
          x2="{end_pos['x']}" y2="{end_pos['y'] - 20}" class="embedded-arrow" />"""

        svg_content += "\n</svg>"
        return svg_content

    def _generate_js_metadata(self) -> str:
        """Generate JavaScript object containing all node metadata."""
        if not self.run_log_data:
            return "{}"

        dag_info = self.run_log_data.get("run_config", {}).get("dag", {})
        nodes = dag_info.get("nodes", {})
        js_metadata = {}

        for node_name, node_data in nodes.items():
            metadata = self._get_node_metadata(node_name, node_data)
            js_metadata[node_name] = metadata

        # Convert to JavaScript object literal
        import json

        return json.dumps(js_metadata, indent=8)

    def _get_node_metadata(
        self, node_name: str, node_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract comprehensive metadata for a node."""
        metadata = {
            "name": node_name,
            "type": node_data.get("node_type", "task"),
            "command": node_data.get("command", ""),
            "next_node": node_data.get("next_node", ""),
        }

        # Add execution data if available
        if self.run_log_data and node_name in self.run_log_data.get("steps", {}):
            step_data = self.run_log_data["steps"][node_name]
            metadata.update(
                {
                    "status": step_data.get("status", "UNKNOWN"),
                    "attempts": len(step_data.get("attempts", [])),
                    "step_type": step_data.get("step_type", ""),
                    "message": step_data.get("message", ""),
                }
            )

            # Add timing info
            attempts = step_data.get("attempts", [])
            if attempts:
                attempt = attempts[0]
                if "start_time" in attempt and "end_time" in attempt:
                    start = datetime.fromisoformat(attempt["start_time"])
                    end = datetime.fromisoformat(attempt["end_time"])
                    duration_ms = (end - start).total_seconds() * 1000
                    metadata.update(
                        {
                            "start_time": start.strftime("%H:%M:%S.%f")[:-3],
                            "end_time": end.strftime("%H:%M:%S.%f")[:-3],
                            "duration_ms": f"{duration_ms:.1f}ms",
                        }
                    )

                # Add parameters if available
                input_params = attempt.get("input_parameters", {})
                output_params = attempt.get("output_parameters", {})
                if input_params:
                    metadata["input_parameters"] = list(input_params.keys())
                if output_params:
                    metadata["output_parameters"] = list(output_params.keys())

            # Add catalog information
            catalog_data = step_data.get("data_catalog", [])
            if catalog_data:
                put_ops = [item for item in catalog_data if item.get("stage") == "put"]
                get_ops = [item for item in catalog_data if item.get("stage") == "get"]

                metadata["catalog_operations"] = {
                    "put_count": len(put_ops),
                    "get_count": len(get_ops),
                    "put_files": [item.get("name", "") for item in put_ops],
                    "get_files": [item.get("name", "") for item in get_ops],
                }

        return metadata


def visualize_pipeline(
    pipeline_or_path: Union[Graph, str, Path],
    run_log_path: Optional[Union[str, Path]] = None,
    output_format: str = "console",
) -> Optional[str]:
    """
    Convenience function to visualize a pipeline.

    Args:
        pipeline_or_path: Graph object or path to pipeline definition
        run_log_path: Optional path to run log JSON file
        output_format: Output format ('console', 'svg', 'both')

    Returns:
        SVG content if output_format includes 'svg', otherwise None
    """
    graph = None
    if isinstance(pipeline_or_path, Graph):
        graph = pipeline_or_path
    # Note: For now, we only support Graph objects directly
    # Future enhancement could parse pipeline definitions

    viz = PipelineVisualizer(graph=graph, run_log_path=run_log_path)

    svg_content = None

    if output_format in ["console", "both"]:
        viz.print_ascii_dag()
        viz.print_execution_summary()

    if output_format in ["svg", "both"]:
        svg_content = viz.generate_simple_svg()

    return svg_content


def analyze_run_logs(
    run_log_dir: Union[str, Path] = ".run_log_store", limit: int = 10
) -> None:
    """
    Analyze recent run logs and provide summary statistics.

    Args:
        run_log_dir: Directory containing run log JSON files
        limit: Maximum number of recent runs to analyze
    """
    log_dir = Path(run_log_dir)

    if not log_dir.exists():
        print(f"Run log directory not found: {log_dir}")
        return

    # Get recent JSON log files
    json_files = sorted(
        log_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
    )[:limit]

    if not json_files:
        print("No run log files found")
        return

    print(f"\n📈 Recent Pipeline Runs Analysis ({len(json_files)} runs)")
    print("=" * 80)
    print(f"{'Run ID':<30} {'Status':<10} {'Steps':<6} {'Duration':<12}")
    print("-" * 80)

    success_count = 0

    for log_file in json_files:
        try:
            with open(log_file, "r") as f:
                data = json.load(f)

            run_id = data.get("run_id", "unknown")[:29]  # Truncate for display
            status = data.get("status", "UNKNOWN")
            steps_count = len(data.get("steps", {}))

            if status == "SUCCESS":
                success_count += 1

            # Calculate total duration (simplified)
            duration = "N/A"
            steps = data.get("steps", {})
            if steps:
                all_times = []
                for step_data in steps.values():
                    for attempt in step_data.get("attempts", []):
                        if "start_time" in attempt:
                            all_times.append(
                                datetime.fromisoformat(attempt["start_time"])
                            )
                        if "end_time" in attempt:
                            all_times.append(
                                datetime.fromisoformat(attempt["end_time"])
                            )

                if len(all_times) >= 2:
                    total_duration_ms = (
                        max(all_times) - min(all_times)
                    ).total_seconds() * 1000
                    duration = f"{total_duration_ms:.0f}ms"

            status_emoji = "✅" if status == "SUCCESS" else "❌"
            print(
                f"{run_id:<30} {status_emoji}{status:<9} {steps_count:<6} {duration:<12}"
            )

        except Exception as e:
            print(f"Error reading {log_file.name}: {e}")

    print("-" * 80)
    success_rate = (success_count / len(json_files)) * 100 if json_files else 0
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/{len(json_files)})")


def visualize_run_by_id_enhanced(
    run_id: str,
    output_file: Optional[str] = None,
    visualization_type: str = "console",
    open_browser: bool = True,
) -> None:
    """
    Visualize a specific run by Run ID with multiple output options.

    Args:
        run_id: The run ID to visualize
        output_file: Optional output file path
        visualization_type: Type of visualization ('console', 'svg', 'interactive', 'html')
        open_browser: Whether to automatically open the file in the default browser
    """
    from pathlib import Path

    # Find run log file
    run_log_dir = Path(".run_log_store")
    log_file = run_log_dir / f"{run_id}.json"

    if not log_file.exists():
        # Try partial match
        matching_files = [f for f in run_log_dir.glob("*.json") if run_id in f.stem]
        if matching_files:
            log_file = matching_files[0]
        else:
            print(f"❌ Run log not found for: {run_id}")
            print("\n💡 Available recent runs:")
            analyze_run_logs(limit=5)
            return

    print(f"📊 Visualizing run: {log_file.stem}")
    print("=" * 60)

    # Create visualizer and show results
    viz = PipelineVisualizer(run_log_path=log_file)

    # Show console output only if not generating HTML (since HTML contains all this info in better format)
    if visualization_type != "html":
        print("\n🔄 Pipeline Structure:")
        viz.print_dag_from_runlog()

        print("\n📈 Execution Summary:")
        viz.print_execution_summary()

    if output_file and visualization_type != "console":
        file_path = Path(output_file).absolute()

        if visualization_type == "html":
            print(f"\n🌐 Generating HTML Dashboard: {output_file}")
            viz.generate_html_dashboard(output_path=output_file)
            print(f"✅ HTML dashboard saved to: {file_path}")
            print("💡 Features:")
            print("   • Rich dashboard with metrics and timeline")
            print("   • Interactive pipeline visualization")
            print("   • Comprehensive execution analysis")

            if open_browser:
                _open_in_browser(file_path)
            else:
                print(f"🔗 Open manually: file://{file_path}")

        elif visualization_type == "interactive":
            print(f"\n🎨 Generating Interactive SVG: {output_file}")
            viz.generate_interactive_svg(output_path=output_file)
            print(f"✅ Interactive SVG saved to: {file_path}")
            print("💡 Features:")
            print("   • Click nodes to see detailed metadata")
            print("   • Hover for visual feedback")
            print("   • Rich execution details")

            if open_browser:
                _open_in_browser(file_path)
            else:
                print(f"🔗 Open manually: file://{file_path}")

        elif visualization_type == "svg":
            print(f"\n🎨 Generating Static SVG: {output_file}")
            viz.generate_svg_from_runlog(output_path=output_file)
            print(f"✅ SVG saved to: {file_path}")

            if open_browser:
                _open_in_browser(file_path)
            else:
                print(f"🔗 Open manually: file://{file_path}")


def visualize_run_by_id(run_id: str, output_svg: Optional[str] = None) -> None:
    """
    Visualize a specific run by Run ID.

    Args:
        run_id: The run ID to visualize
        output_svg: Optional path for SVG output
    """
    from pathlib import Path

    # Find run log file
    run_log_dir = Path(".run_log_store")
    log_file = run_log_dir / f"{run_id}.json"

    if not log_file.exists():
        # Try partial match
        matching_files = [f for f in run_log_dir.glob("*.json") if run_id in f.stem]
        if matching_files:
            log_file = matching_files[0]
        else:
            print(f"❌ Run log not found for: {run_id}")
            print("\n💡 Available recent runs:")
            analyze_run_logs(limit=5)
            return

    print(f"📊 Visualizing run: {log_file.stem}")
    print("=" * 60)

    # Create visualizer and show results
    viz = PipelineVisualizer(run_log_path=log_file)

    print("\n🔄 Pipeline Structure:")
    viz.print_dag_from_runlog()

    print("\n📈 Execution Summary:")
    viz.print_execution_summary()

    if output_svg:
        print(f"\n🎨 Generating SVG: {output_svg}")
        viz.generate_svg_from_runlog(output_path=output_svg)
        print(f"✅ SVG saved to: {Path(output_svg).absolute()}")


if __name__ == "__main__":
    """CLI interface for visualization tools."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m runnable.viz analyze [run_log_dir] [limit]")
        print("  python -m runnable.viz run <run_id> [--svg output.svg]")
        print("  python -m runnable.viz --help")
        sys.exit(1)

    command = sys.argv[1]

    if command == "analyze":
        run_log_dir = sys.argv[2] if len(sys.argv) > 2 else ".run_log_store"
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        analyze_run_logs(run_log_dir, limit)

    elif command == "run":
        if len(sys.argv) < 3:
            print("❌ Please provide a run ID")
            print("Usage: python -m runnable.viz run <run_id> [--svg output.svg]")
            sys.exit(1)

        run_id = sys.argv[2]
        output_svg = None

        if "--svg" in sys.argv:
            svg_idx = sys.argv.index("--svg")
            if svg_idx + 1 < len(sys.argv):
                output_svg = sys.argv[svg_idx + 1]
            else:
                output_svg = f"{run_id}_diagram.svg"

        visualize_run_by_id(run_id, output_svg)

    else:
        print(f"Unknown command: {command}")
        print("Available commands: analyze, run")
        sys.exit(1)

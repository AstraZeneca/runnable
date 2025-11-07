"""
Lightweight visualization utilities for Runnable pipelines.

This module provides simple, CLI-friendly visualization tools that avoid
complex web frameworks and focus on developer experience.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

from runnable.graph import Graph


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
                "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⏸️ "
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

        # Find next node
        next_node = node.get("next_node")
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
        print(f"{'Step Name':<25} {'Status':<10} {'Duration':<12} {'Attempts':<8}")
        print("-" * 80)

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
                "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⏸️ "
            )

            print(
                f"{step_name:<25} {status_emoji}{status:<9} {duration:<12} {attempts:<8}"
            )

        print("-" * 80)
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
                elif step_status == "FAILED":
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

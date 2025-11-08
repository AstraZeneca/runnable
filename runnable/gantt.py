"""
Gantt chart visualization for Runnable pipeline execution timelines.

This module provides timeline-based visualization that naturally handles
composite nodes like parallel, map, and conditional executions by showing
their temporal relationships and hierarchical structure.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


class GanttVisualizer:
    """
    Timeline-based visualizer for pipeline execution using Gantt chart principles.

    This approach is particularly effective for composite nodes because:
    - Parallel branches are shown as overlapping time bars
    - Hierarchical structure is clear through indentation
    - Performance bottlenecks are immediately visible
    - Execution flow follows natural time progression
    """

    def __init__(self, run_log_path: Union[str, Path]):
        """
        Initialize with run log data.

        Args:
            run_log_path: Path to JSON run log file
        """
        self.run_log_path = Path(run_log_path)
        self.run_log_data = None
        self.timeline_data = []
        self.global_start = None
        self.global_end = None
        self.total_duration_ms = 0

        self.load_run_log()
        self.analyze_timeline()

    def load_run_log(self) -> None:
        """Load and parse run log data."""
        if not self.run_log_path.exists():
            raise FileNotFoundError(f"Run log not found: {self.run_log_path}")

        with open(self.run_log_path, "r") as f:
            self.run_log_data = json.load(f)

    def parse_time(self, time_str: str) -> Optional[datetime]:
        """Parse ISO timestamp string to datetime object."""
        try:
            return datetime.fromisoformat(time_str) if time_str else None
        except (ValueError, TypeError):
            return None

    def get_step_timing(
        self, step_data: Dict[str, Any]
    ) -> Tuple[Optional[datetime], Optional[datetime], float]:
        """
        Extract timing information from step data.

        Returns:
            Tuple of (start_time, end_time, duration_ms)
        """
        attempts = step_data.get("attempts", [])
        if not attempts:
            return None, None, 0

        attempt = attempts[0]  # Use first attempt
        start = self.parse_time(attempt.get("start_time"))
        end = self.parse_time(attempt.get("end_time"))

        if start and end:
            duration_ms = (end - start).total_seconds() * 1000
            return start, end, duration_ms

        return None, None, 0

    def _extract_step_metadata(
        self, step_name: str, step_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract comprehensive metadata for a step.

        Returns metadata including command, parameters, catalog operations, etc.
        """
        metadata = {
            "command": "",
            "command_type": "",
            "attempts": 0,
            "input_parameters": [],
            "output_parameters": [],
            "catalog_operations": {"put": [], "get": []},
            "start_time": "",
            "end_time": "",
        }

        # Get DAG node information for command details
        dag_nodes = (
            self.run_log_data.get("run_config", {}).get("dag", {}).get("nodes", {})
        )
        if step_name in dag_nodes:
            dag_node = dag_nodes[step_name]
            metadata["command"] = dag_node.get("command", "")
            metadata["command_type"] = dag_node.get("command_type", "")

        # Get execution details
        attempts = step_data.get("attempts", [])
        metadata["attempts"] = len(attempts)

        if attempts:
            attempt = attempts[0]  # Use first attempt
            metadata["start_time"] = attempt.get("start_time", "")
            metadata["end_time"] = attempt.get("end_time", "")

            # Extract parameters
            input_params = attempt.get("input_parameters", {})
            output_params = attempt.get("output_parameters", {})
            metadata["input_parameters"] = (
                list(input_params.keys()) if input_params else []
            )
            metadata["output_parameters"] = (
                list(output_params.keys()) if output_params else []
            )

        # Extract catalog operations
        catalog_data = step_data.get("data_catalog", [])
        for item in catalog_data:
            stage = item.get("stage", "")
            if stage == "put":
                metadata["catalog_operations"]["put"].append(item.get("name", ""))
            elif stage == "get":
                metadata["catalog_operations"]["get"].append(item.get("name", ""))

        return metadata

    def analyze_timeline(self) -> None:
        """
        Analyze run log to extract timeline data for all steps including composite nodes.

        This builds a hierarchical timeline structure that can handle:
        - Simple sequential steps
        - Composite nodes (parallel, map, conditional) with sub-steps
        - Branch execution (sequential locally, showing logical structure)
        """
        if not self.run_log_data:
            return

        all_times = []
        timeline_items = []

        # Process top-level steps
        for step_name, step_data in self.run_log_data.get("steps", {}).items():
            start, end, duration = self.get_step_timing(step_data)
            is_composite = bool(step_data.get("branches"))

            # Get additional metadata
            metadata = self._extract_step_metadata(step_name, step_data)

            # For composite steps, we still show them as top-level items
            if start and end:
                all_times.extend([start, end])

            timeline_items.append(
                {
                    "name": step_name,
                    "start": start,
                    "end": end,
                    "duration_ms": duration,
                    "status": step_data.get("status", "UNKNOWN"),
                    "step_type": step_data.get("step_type", "task"),
                    "level": 0,
                    "is_composite": is_composite,
                    "parent": None,
                    "metadata": metadata,
                }
            )

            # Process branches if they exist (composite nodes)
            branches = step_data.get("branches", {})
            if branches:
                for branch_name, branch_data in branches.items():
                    # Add a branch header
                    timeline_items.append(
                        {
                            "name": f"Branch: {branch_name.split('.')[-1]}",
                            "start": None,
                            "end": None,
                            "duration_ms": 0,
                            "status": branch_data.get("status", "UNKNOWN"),
                            "step_type": "branch",
                            "level": 1,
                            "is_composite": False,
                            "parent": step_name,
                            "branch": branch_name,
                            "metadata": {},
                        }
                    )

                    # Process sub-steps within each branch
                    for sub_step_name, sub_step_data in branch_data.get(
                        "steps", {}
                    ).items():
                        sub_start, sub_end, sub_duration = self.get_step_timing(
                            sub_step_data
                        )

                        if sub_start and sub_end:
                            all_times.extend([sub_start, sub_end])

                        # Get metadata for sub-step
                        sub_metadata = self._extract_step_metadata(
                            sub_step_name, sub_step_data
                        )

                        timeline_items.append(
                            {
                                "name": sub_step_data.get(
                                    "name", sub_step_name
                                ),  # Use clean name
                                "start": sub_start,
                                "end": sub_end,
                                "duration_ms": sub_duration,
                                "status": sub_step_data.get("status", "UNKNOWN"),
                                "step_type": sub_step_data.get("step_type", "task"),
                                "level": 2,  # Branch sub-steps are level 2
                                "is_composite": False,
                                "parent": step_name,
                                "branch": branch_name,
                                "metadata": sub_metadata,
                            }
                        )

        # Sort timeline items to group by composite -> branch1 (header + steps) -> branch2 (header + steps) -> remaining
        def sort_key(item):
            branch_name = item.get("branch", "")

            # Extract branch number for ordering
            if "branch1" in branch_name:
                branch_order = "1"
            elif "branch2" in branch_name:
                branch_order = "2"
            else:
                branch_order = "9"

            if item["is_composite"]:
                return (0, item["start"] or datetime.min)  # Composite nodes first
            elif item["step_type"] == "branch":
                return (
                    1,
                    branch_order,
                    "0",
                )  # Branch headers first within their branch
            elif item["parent"] and item.get("branch"):
                # Branch steps grouped by branch, then by execution time
                return (1, branch_order, "1", item["start"] or datetime.min)
            else:
                return (9, item["start"] or datetime.min)  # Regular items last

        timeline_items.sort(key=sort_key)
        self.timeline_data = timeline_items

        # Calculate global timeline bounds
        if all_times:
            self.global_start = min(all_times)
            self.global_end = max(all_times)
            self.total_duration_ms = (
                self.global_end - self.global_start
            ).total_seconds() * 1000

    def print_gantt_console(self) -> None:
        """
        Print a Rich-enhanced console timeline representation.

        This provides a clean text-based timeline with:
        - Dark terminal friendly colors
        - Simple duration bars (not cascading)
        - Metadata in the left column for better readability
        - Sequential timeline understanding
        """
        if not self.timeline_data:
            console = Console()
            console.print("❌ No timeline data available", style="bright_red")
            return

        console = Console(width=120)
        run_id = self.run_log_data.get("run_id", "unknown")
        status = self.run_log_data.get("status", "UNKNOWN")

        # Create header with dark terminal friendly colors
        status_color = (
            "bright_green"
            if status == "SUCCESS"
            else "bright_red"
            if status == "FAIL"
            else "bright_yellow"
        )
        header_text = f"⏱️  Pipeline Timeline - [bold white]{run_id}[/bold white]\n"
        header_text += f"Total Duration: [bold white]{self.total_duration_ms:.1f}ms[/bold white] | "
        header_text += f"Status: [bold {status_color}]{status}[/bold {status_color}]"

        console.print(Panel(header_text, box=box.ROUNDED, style="bright_blue"))
        console.print()

        # Create main timeline table with better spacing
        table = Table(show_header=True, header_style="bold bright_blue", box=box.SIMPLE)
        table.add_column("Step & Details", style="bright_white", width=60)
        table.add_column("Duration", justify="right", style="bright_magenta", width=12)
        table.add_column("Progress Bar", width=40)

        current_branch = None

        for item in self.timeline_data:
            # Handle branch headers (no timing info)
            if item["step_type"] == "branch":
                branch_name = item["name"]
                indent = "  " * item["level"]

                # Add horizontal separator between branches (but not before first branch)
                if current_branch is not None:
                    separator = (
                        f"{indent}[dim bright_black]{'─' * 50}[/dim bright_black]"
                    )
                    table.add_row(separator, "", "")

                branch_row = (
                    f"{indent}[bold bright_cyan]├─ {branch_name}[/bold bright_cyan]"
                )
                table.add_row(branch_row, "", "")
                current_branch = item["branch"]
                continue

            # For regular steps, check if we need to switch branches
            if item.get("branch") and item["branch"] != current_branch:
                # This shouldn't happen with proper sorting, but let's handle it
                current_branch = item["branch"]

            # Calculate simple duration bar (not cascading)
            if self.total_duration_ms > 0 and item["duration_ms"] > 0:
                duration_percentage = item["duration_ms"] / self.total_duration_ms
                bar_width = 30  # Fixed width for progress bar
                duration_width = max(1, int(duration_percentage * bar_width))
            else:
                duration_width = 1 if item["duration_ms"] > 0 else 0

            # Create indentation and icons
            indent = "  " * item["level"]

            # Status styling - bright colors for dark terminals
            if item["status"] == "SUCCESS":
                status_style = "bright_green"
                status_icon = "✅"
            elif item["status"] == "FAIL":
                status_style = "bright_red"
                status_icon = "❌"
            else:
                status_style = "bright_yellow"
                status_icon = "⏸️"

            # Type icons
            type_icons = {
                "task": "⚙️",
                "parallel": "🔀",
                "map": "🔁",
                "conditional": "🔀",
                "stub": "📝",
                "success": "✅",
                "fail": "❌",
            }
            type_icon = type_icons.get(item["step_type"], "⚙️")

            # Format step name with metadata below
            step_name = f"{indent}{type_icon} [{status_style}]{status_icon}[/{status_style}] [bold bright_white]{item['name']}[/bold bright_white]"

            # For composite nodes, add indication
            if item.get("is_composite"):
                step_name += (
                    f" [dim bright_magenta]({item['step_type']})[/dim bright_magenta]"
                )

            # Add metadata right below the step name
            metadata = item.get("metadata", {})
            if metadata:
                metadata_line = self._format_rich_metadata_inline(metadata)
                if metadata_line:
                    step_name += f"\n{indent}   [dim bright_cyan]{metadata_line}[/dim bright_cyan]"

            # Duration with color coding based on performance
            duration_ms = item["duration_ms"]
            if duration_ms == 0:
                duration_text = "[dim]0.0ms[/dim]"
            elif duration_ms < 10:
                duration_color = "bright_green"
                duration_text = (
                    f"[{duration_color}]{duration_ms:.1f}ms[/{duration_color}]"
                )
            elif duration_ms < 100:
                duration_color = "bright_yellow"
                duration_text = (
                    f"[{duration_color}]{duration_ms:.1f}ms[/{duration_color}]"
                )
            else:
                duration_color = "bright_red"
                duration_text = (
                    f"[{duration_color}]{duration_ms:.1f}ms[/{duration_color}]"
                )

            # Create simple progress bar
            if duration_width > 0:
                bar_color = (
                    "bright_green"
                    if item["status"] == "SUCCESS"
                    else "bright_red"
                    if item["status"] == "FAIL"
                    else "bright_yellow"
                )
                progress_bar = f"[{bar_color}]{'█' * duration_width}[/{bar_color}]"
            else:
                progress_bar = ""

            table.add_row(step_name, duration_text, progress_bar)

        console.print(table)
        console.print()

    def _format_rich_metadata_inline(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for inline display in the step column."""
        details = []

        # Command information - keep it concise
        if metadata.get("command") and metadata.get("command_type"):
            command = metadata["command"]
            # Truncate long commands more aggressively for inline display
            if len(command) > 40:
                command = command[:37] + "..."
            details.append(f"{metadata['command_type'].upper()}: {command}")

        # Catalog operations
        catalog_ops = metadata.get("catalog_operations", {})
        put_ops = catalog_ops.get("put", [])
        get_ops = catalog_ops.get("get", [])

        catalog_info = []
        if put_ops:
            put_str = ", ".join(put_ops[:2]) + ("..." if len(put_ops) > 2 else "")
            catalog_info.append(f"📤 {put_str}")

        if get_ops:
            get_str = ", ".join(get_ops[:2]) + ("..." if len(get_ops) > 2 else "")
            catalog_info.append(f"📥 {get_str}")

        if catalog_info:
            details.append(" | ".join(catalog_info))

        # Start time - just show time, not full timestamp
        start_time = metadata.get("start_time", "")
        if start_time:
            try:
                parsed_time = datetime.fromisoformat(start_time)
                time_str = parsed_time.strftime("%H:%M:%S.%f")[:-3]
                details.append(f"🕐 {time_str}")
            except (ValueError, TypeError):
                pass

        return " • ".join(details)

    def _format_rich_metadata(self, metadata: Dict[str, Any], level: int) -> str:
        """Format metadata for Rich display in a compact format."""
        details = []

        # Command information
        if metadata.get("command") and metadata.get("command_type"):
            command = (
                metadata["command"][:50] + "..."
                if len(metadata["command"]) > 50
                else metadata["command"]
            )
            details.append(f"📝 {metadata['command_type'].upper()}: {command}")

        # Start time
        start_time = metadata.get("start_time", "")
        if start_time:
            try:
                parsed_time = datetime.fromisoformat(start_time)
                time_str = parsed_time.strftime("%H:%M:%S.%f")[:-3]
                details.append(f"🕐 {time_str}")
            except (ValueError, TypeError):
                pass

        # Catalog operations
        catalog_ops = metadata.get("catalog_operations", {})
        put_ops = catalog_ops.get("put", [])
        get_ops = catalog_ops.get("get", [])

        if put_ops:
            put_str = ", ".join(put_ops[:2]) + ("..." if len(put_ops) > 2 else "")
            details.append(f"📤 PUT: {put_str}")

        if get_ops:
            get_str = ", ".join(get_ops[:2]) + ("..." if len(get_ops) > 2 else "")
            details.append(f"📥 GET: {get_str}")

        # Parameters
        input_params = metadata.get("input_parameters", [])
        output_params = metadata.get("output_parameters", [])

        if input_params:
            param_str = ", ".join(input_params[:2]) + (
                "..." if len(input_params) > 2 else ""
            )
            details.append(f"🔗 In: {param_str}")

        if output_params:
            param_str = ", ".join(output_params[:2]) + (
                "..." if len(output_params) > 2 else ""
            )
            details.append(f"🔗 Out: {param_str}")

        return " | ".join(details)

    def _print_step_metadata(self, metadata: Dict[str, Any], level: int) -> None:
        """Print formatted metadata for a step."""
        indent = "  " * (level + 1)  # Extra indentation for metadata

        # Command information
        if metadata.get("command"):
            command_type = metadata.get("command_type", "").upper()
            command = metadata.get("command", "")
            print(f"{indent}📝 {command_type}: {command}")

        # Execution details
        details = []
        if metadata.get("attempts", 0) > 1:
            details.append(f"Attempts: {metadata['attempts']}")

        start_time = metadata.get("start_time", "")
        if start_time:
            try:
                parsed_time = datetime.fromisoformat(start_time)
                time_str = parsed_time.strftime("%H:%M:%S.%f")[:-3]
                details.append(f"Started: {time_str}")
            except (ValueError, TypeError):
                pass

        if details:
            print(f"{indent}📊 {' | '.join(details)}")

        # Parameters
        input_params = metadata.get("input_parameters", [])
        output_params = metadata.get("output_parameters", [])
        param_info = []

        if input_params:
            param_info.append(
                f"In: {', '.join(input_params[:3])}{'...' if len(input_params) > 3 else ''}"
            )
        if output_params:
            param_info.append(
                f"Out: {', '.join(output_params[:3])}{'...' if len(output_params) > 3 else ''}"
            )

        if param_info:
            print(f"{indent}🔗 Parameters: {' | '.join(param_info)}")

        # Catalog operations
        catalog_ops = metadata.get("catalog_operations", {})
        put_ops = catalog_ops.get("put", [])
        get_ops = catalog_ops.get("get", [])

        if put_ops or get_ops:
            catalog_info = []
            if put_ops:
                catalog_info.append(
                    f"📤 PUT: {', '.join(put_ops[:2])}{'...' if len(put_ops) > 2 else ''}"
                )
            if get_ops:
                catalog_info.append(
                    f"📥 GET: {', '.join(get_ops[:2])}{'...' if len(get_ops) > 2 else ''}"
                )
            print(f"{indent}📁 Catalog: {' | '.join(catalog_info)}")

        # Add a small separator for readability
        if (
            metadata.get("command")
            or input_params
            or output_params
            or put_ops
            or get_ops
        ):
            print()

    def generate_html_gantt(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate an interactive HTML Gantt chart.

        This creates a rich web-based timeline visualization with:
        - Zoomable timeline
        - Hover tooltips with detailed info
        - Expandable/collapsible composite nodes
        - Color coding for status and performance
        """
        if not self.timeline_data:
            return "<html><body>No timeline data available</body></html>"

        run_id = self.run_log_data.get("run_id", "unknown")
        status = self.run_log_data.get("status", "UNKNOWN")

        # Generate HTML with embedded CSS and JavaScript
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Gantt Chart - {run_id}</title>
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
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 1rem;
        }}

        .timeline-header {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .gantt-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            overflow: hidden;
        }}

        .timeline-scale {{
            background: #f1f5f9;
            border-bottom: 2px solid #e2e8f0;
            padding: 0.5rem 1rem;
            font-family: monospace;
            font-size: 0.875rem;
            display: grid;
            grid-template-columns: 300px 1fr;
        }}

        .scale-markers {{
            position: relative;
            height: 30px;
            background: linear-gradient(to right, #e2e8f0 0%, #e2e8f0 100%);
            border-radius: 4px;
        }}

        .timeline-row {{
            display: grid;
            grid-template-columns: 300px 1fr;
            border-bottom: 1px solid #f1f5f9;
            min-height: 40px;
            align-items: center;
            transition: background-color 0.2s ease;
        }}

        .timeline-row:hover {{
            background: #f8fafc;
        }}

        .step-info {{
            padding: 0.5rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            position: relative;
        }}

        .step-info:hover::before {{
            content: "Click to expand details";
            position: absolute;
            top: -1.5rem;
            left: 50%;
            transform: translateX(-50%);
            background: #1f2937;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            white-space: nowrap;
            z-index: 100;
            opacity: 0;
            animation: fadeIn 0.2s ease forwards;
        }}

        @keyframes fadeIn {{
            to {{ opacity: 1; }}
        }}

        .step-level-0 {{ padding-left: 1rem; }}
        .step-level-1 {{ padding-left: 2rem; }}
        .step-level-2 {{ padding-left: 3rem; }}

        .step-icon {{
            font-size: 1.1em;
        }}

        .status-success {{ color: #16a34a; }}
        .status-fail {{ color: #dc2626; }}

        .timeline-bar-container {{
            position: relative;
            height: 24px;
            margin: 0 1rem;
            background: #f1f5f9;
            border-radius: 4px;
        }}

        .timeline-bar {{
            position: absolute;
            top: 4px;
            height: 16px;
            border-radius: 3px;
            transition: all 0.2s ease;
            cursor: pointer;
        }}

        .timeline-bar:hover {{
            transform: scaleY(1.2);
            z-index: 10;
        }}

        .bar-success {{ background: linear-gradient(90deg, #22c55e, #16a34a); }}
        .bar-fail {{ background: linear-gradient(90deg, #ef4444, #dc2626); }}
        .bar-unknown {{ background: linear-gradient(90deg, #f59e0b, #d97706); }}

        .tooltip {{
            position: absolute;
            background: #1f2937;
            color: white;
            padding: 0.75rem;
            border-radius: 6px;
            font-size: 0.875rem;
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
            max-width: 400px;
            line-height: 1.4;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }}

        .tooltip.show {{ opacity: 1; }}

        .status-badge {{
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .badge-success {{ background: #dcfce7; color: #16a34a; }}
        .badge-fail {{ background: #fecaca; color: #dc2626; }}
        .badge-unknown {{ background: #fef3c7; color: #d97706; }}

        .branch-header {{
            background: #f8fafc !important;
            border-top: 2px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
        }}

        .branch-name {{
            font-weight: 600;
            color: #0ea5e9;
            font-size: 0.95rem;
        }}

        .branch-separator {{
            width: 100%;
            height: 2px;
            background: linear-gradient(to right, #0ea5e9, transparent);
            border-radius: 1px;
            margin: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⏱️ Pipeline Execution Timeline</h1>
        <p>Gantt Chart Visualization</p>
    </div>

    <div class="container">
        <div class="timeline-header">
            <div>
                <h2>Run ID: {run_id}</h2>
                <p>Total Duration: {self.total_duration_ms:.1f}ms</p>
            </div>
            <div>
                <span class="status-badge badge-{status.lower()}">
                    {status}
                </span>
            </div>
        </div>

        <div class="gantt-container">
            <div class="timeline-scale">
                <div><strong>Pipeline Steps</strong></div>
                <div class="scale-markers" id="timeline-scale">
                    <!-- Scale markers will be added by JavaScript -->
                </div>
            </div>

            <div id="timeline-rows">
                {self._generate_timeline_rows_html()}
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
        const totalDuration = {self.total_duration_ms};

        // Generate timeline scale markers
        function generateTimelineScale() {{
            const scaleContainer = document.getElementById('timeline-scale');
            const markerCount = 10;

            for (let i = 0; i <= markerCount; i++) {{
                const percentage = (i / markerCount) * 100;
                const timeMs = (percentage / 100) * totalDuration;

                const marker = document.createElement('div');
                marker.style.position = 'absolute';
                marker.style.left = percentage + '%';
                marker.style.top = '0';
                marker.style.width = '1px';
                marker.style.height = '100%';
                marker.style.background = '#94a3b8';

                const label = document.createElement('div');
                label.style.position = 'absolute';
                label.style.left = percentage + '%';
                label.style.top = '100%';
                label.style.transform = 'translateX(-50%)';
                label.style.fontSize = '0.75rem';
                label.style.color = '#64748b';
                label.textContent = Math.round(timeMs) + 'ms';

                scaleContainer.appendChild(marker);
                scaleContainer.appendChild(label);
            }}
        }}

        // Enhanced tooltip functionality with rich metadata
        const tooltip = document.getElementById('tooltip');

        document.querySelectorAll('.timeline-bar').forEach(bar => {{
            bar.addEventListener('mouseenter', (e) => {{
                const stepName = e.target.dataset.step;
                const duration = e.target.dataset.duration;
                const status = e.target.dataset.status;
                const startTime = e.target.dataset.start;
                const command = e.target.dataset.command || '';
                const commandType = e.target.dataset.commandType || '';
                const inputParams = e.target.dataset.inputParams || '';
                const outputParams = e.target.dataset.outputParams || '';
                const catalogPut = e.target.dataset.catalogPut || '';
                const catalogGet = e.target.dataset.catalogGet || '';
                const attempts = e.target.dataset.attempts || '1';

                let tooltipContent = `<strong>${{stepName}}</strong><br>
                    Duration: ${{duration}}ms | Status: ${{status}}<br>
                    Started: ${{startTime}}`;

                // Add command information
                if (command && commandType) {{
                    tooltipContent += `<br><br><strong>📝 ${{commandType.toUpperCase()}}:</strong> ${{command}}`;
                }}

                // Add execution details
                const details = [];
                if (attempts > 1) {{
                    details.push(`Attempts: ${{attempts}}`);
                }}
                if (details.length > 0) {{
                    tooltipContent += `<br><strong>📊 Execution:</strong> ${{details.join(' | ')}}`;
                }}

                // Add parameters
                const paramInfo = [];
                if (inputParams) {{
                    const params = inputParams.split(',').slice(0, 3);
                    paramInfo.push(`In: ${{params.join(', ')}}${{inputParams.split(',').length > 3 ? '...' : ''}}`);
                }}
                if (outputParams) {{
                    const params = outputParams.split(',').slice(0, 3);
                    paramInfo.push(`Out: ${{params.join(', ')}}${{outputParams.split(',').length > 3 ? '...' : ''}}`);
                }}
                if (paramInfo.length > 0) {{
                    tooltipContent += `<br><strong>🔗 Parameters:</strong> ${{paramInfo.join(' | ')}}`;
                }}

                // Add catalog operations
                const catalogInfo = [];
                if (catalogPut) {{
                    const putOps = catalogPut.split(',').slice(0, 2);
                    catalogInfo.push(`📤 PUT: ${{putOps.join(', ')}}${{catalogPut.split(',').length > 2 ? '...' : ''}}`);
                }}
                if (catalogGet) {{
                    const getOps = catalogGet.split(',').slice(0, 2);
                    catalogInfo.push(`📥 GET: ${{getOps.join(', ')}}${{catalogGet.split(',').length > 2 ? '...' : ''}}`);
                }}
                if (catalogInfo.length > 0) {{
                    tooltipContent += `<br><strong>📁 Catalog:</strong> ${{catalogInfo.join(' | ')}}`;
                }}

                tooltip.innerHTML = tooltipContent;
                tooltip.classList.add('show');
            }});

            bar.addEventListener('mousemove', (e) => {{
                tooltip.style.left = e.pageX + 10 + 'px';
                tooltip.style.top = e.pageY - 10 + 'px';
            }});

            bar.addEventListener('mouseleave', () => {{
                tooltip.classList.remove('show');
            }});
        }});

        // Click to expand functionality
        document.querySelectorAll('.timeline-row').forEach(row => {{
            const stepInfo = row.querySelector('.step-info');
            stepInfo.style.cursor = 'pointer';

            stepInfo.addEventListener('click', () => {{
                const existingDetails = row.querySelector('.step-details');
                if (existingDetails) {{
                    existingDetails.remove();
                    return;
                }}

                const bar = row.querySelector('.timeline-bar');
                const stepName = bar.dataset.step;
                const command = bar.dataset.command || '';
                const commandType = bar.dataset.commandType || '';
                const inputParams = bar.dataset.inputParams || '';
                const outputParams = bar.dataset.outputParams || '';
                const catalogPut = bar.dataset.catalogPut || '';
                const catalogGet = bar.dataset.catalogGet || '';
                const attempts = bar.dataset.attempts || '1';
                const startTime = bar.dataset.start;

                const detailsDiv = document.createElement('div');
                detailsDiv.className = 'step-details';
                detailsDiv.style.cssText = `
                    grid-column: 1 / -1;
                    background: #f8fafc;
                    padding: 1rem;
                    font-size: 0.875rem;
                    border-top: 1px solid #e2e8f0;
                    line-height: 1.6;
                `;

                let detailsContent = '';

                // Command information
                if (command && commandType) {{
                    detailsContent += `<div style="margin-bottom: 0.5rem;"><strong>📝 ${{commandType.toUpperCase()}}:</strong> ${{command}}</div>`;
                }}

                // Execution details
                const details = [];
                if (attempts > 1) {{
                    details.push(`Attempts: ${{attempts}}`);
                }}
                details.push(`Started: ${{startTime}}`);
                if (details.length > 0) {{
                    detailsContent += `<div style="margin-bottom: 0.5rem;"><strong>📊 Execution:</strong> ${{details.join(' | ')}}</div>`;
                }}

                // Parameters
                const paramInfo = [];
                if (inputParams) {{
                    paramInfo.push(`<strong>Input:</strong> ${{inputParams.split(',').join(', ')}}`);
                }}
                if (outputParams) {{
                    paramInfo.push(`<strong>Output:</strong> ${{outputParams.split(',').join(', ')}}`);
                }}
                if (paramInfo.length > 0) {{
                    detailsContent += `<div style="margin-bottom: 0.5rem;"><strong>🔗 Parameters:</strong><br>`;
                    paramInfo.forEach(info => detailsContent += `&nbsp;&nbsp;• ${{info}}<br>`);
                    detailsContent += `</div>`;
                }}

                // Catalog operations
                const catalogInfo = [];
                if (catalogPut) {{
                    catalogInfo.push(`<strong>PUT Operations:</strong> ${{catalogPut.split(',').join(', ')}}`);
                }}
                if (catalogGet) {{
                    catalogInfo.push(`<strong>GET Operations:</strong> ${{catalogGet.split(',').join(', ')}}`);
                }}
                if (catalogInfo.length > 0) {{
                    detailsContent += `<div><strong>📁 Catalog Operations:</strong><br>`;
                    catalogInfo.forEach(info => detailsContent += `&nbsp;&nbsp;• ${{info}}<br>`);
                    detailsContent += `</div>`;
                }}

                if (!detailsContent) {{
                    detailsContent = '<div style="color: #64748b;">No additional metadata available</div>';
                }}

                detailsContent += '<div style="margin-top: 0.5rem; font-size: 0.75rem; color: #64748b;">Click to collapse</div>';

                detailsContent = `<div style="max-height: 0; overflow: hidden; transition: max-height 0.3s ease;">${{detailsContent}}</div>`;

                detailsDiv.innerHTML = detailsContent;
                row.appendChild(detailsDiv);

                // Animate expansion
                setTimeout(() => {{
                    const content = detailsDiv.querySelector('div');
                    content.style.maxHeight = content.scrollHeight + 'px';
                }}, 10);
            }});
        }});

        // Initialize
        generateTimelineScale();
    </script>
</body>
</html>"""

        # Save to file if path provided
        if output_path:
            Path(output_path).write_text(html_content)
            print(f"Gantt chart saved to: {output_path}")

        return html_content

    def _generate_timeline_rows_html(self) -> str:
        """Generate HTML for timeline rows."""
        html_parts = []

        for item in self.timeline_data:
            # Handle branch headers in HTML
            if item["step_type"] == "branch":
                branch_name = item["name"]
                level_class = f"step-level-{item['level']}"

                html_parts.append(f"""
                    <div class="timeline-row branch-header">
                        <div class="step-info {level_class}">
                            <span class="step-icon">🌿</span>
                            <span class="branch-name">{branch_name}</span>
                        </div>
                        <div class="timeline-bar-container">
                            <div class="branch-separator"></div>
                        </div>
                    </div>
                """)
                continue

            # Calculate bar position and width
            if self.total_duration_ms > 0 and item["start"] and item["duration_ms"] > 0:
                start_offset_ms = (
                    item["start"] - self.global_start
                ).total_seconds() * 1000
                start_percentage = (start_offset_ms / self.total_duration_ms) * 100
                width_percentage = (item["duration_ms"] / self.total_duration_ms) * 100
            else:
                start_percentage = 0
                width_percentage = 0.5  # Minimum width for visibility

            # Status and styling
            status_class = item["status"].lower()
            bar_class = (
                f"bar-{status_class}"
                if status_class in ["success", "fail"]
                else "bar-unknown"
            )

            # Type icon
            type_icons = {
                "task": "⚙️",
                "parallel": "🔀",
                "map": "🔁",
                "stub": "📝",
                "success": "✅",
                "fail": "❌",
            }
            type_icon = type_icons.get(item["step_type"], "⚙️")

            # Status emoji
            status_emoji = (
                "✅"
                if item["status"] == "SUCCESS"
                else "❌"
                if item["status"] == "FAIL"
                else "⏸️"
            )

            # Extract metadata for data attributes
            metadata = item.get("metadata", {})
            command = metadata.get("command", "").replace('"', "&quot;")
            command_type = metadata.get("command_type", "")
            input_params = ",".join(metadata.get("input_parameters", []))
            output_params = ",".join(metadata.get("output_parameters", []))
            catalog_put = ",".join(
                metadata.get("catalog_operations", {}).get("put", [])
            )
            catalog_get = ",".join(
                metadata.get("catalog_operations", {}).get("get", [])
            )
            attempts = str(metadata.get("attempts", 1))

            html_parts.append(f"""
                <div class="timeline-row">
                    <div class="step-info step-level-{item["level"]}">
                        <span class="step-icon">{type_icon}</span>
                        <span class="status-success" title="{item["status"]}">{status_emoji}</span>
                        <span title="{item["name"]}">{item["name"][:25]}{'...' if len(item["name"]) > 25 else ''}</span>
                        <small>({item["duration_ms"]:.1f}ms)</small>
                    </div>
                    <div class="timeline-bar-container">
                        <div class="timeline-bar {bar_class}"
                             style="left: {start_percentage:.2f}%; width: {max(0.5, width_percentage):.2f}%;"
                             data-step="{item["name"]}"
                             data-duration="{item["duration_ms"]:.1f}"
                             data-status="{item["status"]}"
                             data-start="{item['start'].strftime('%H:%M:%S.%f')[:-3] if item['start'] else ''}"
                             data-command="{command}"
                             data-command-type="{command_type}"
                             data-input-params="{input_params}"
                             data-output-params="{output_params}"
                             data-catalog-put="{catalog_put}"
                             data-catalog-get="{catalog_get}"
                             data-attempts="{attempts}">
                        </div>
                    </div>
                </div>
            """)

        return "".join(html_parts)


def visualize_gantt(
    run_id: str, output_file: Optional[str] = None, open_browser: bool = True
) -> None:
    """
    Convenience function to visualize pipeline execution as a Gantt chart.

    Args:
        run_id: The run ID to visualize
        output_file: Optional output HTML file path
        open_browser: Whether to open the result in browser
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
            return

    print(f"⏱️  Generating Gantt chart for: {log_file.stem}")

    # Create visualizer
    gantt = GanttVisualizer(log_file)

    # Show console timeline
    gantt.print_gantt_console()

    # Generate HTML if requested
    if output_file:
        gantt.generate_html_gantt(output_file)

        if open_browser:
            import webbrowser

            file_path = Path(output_file).absolute()
            print(f"🌐 Opening Gantt chart in browser: {file_path.name}")
            webbrowser.open(file_path.as_uri())

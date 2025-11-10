"""
Simplified visualization for Runnable pipeline execution.

This module provides lightweight, reusable components that understand
the composite pipeline structure documented in the run logs.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class StepInfo:
    """Clean representation of a pipeline step."""

    name: str
    internal_name: str
    status: str
    step_type: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_ms: float
    level: int  # 0=top-level, 1=branch, 2=nested
    parent: Optional[str]
    branch: Optional[str]
    command: str
    command_type: str
    input_params: List[str]
    output_params: List[str]
    catalog_ops: Dict[str, List[str]]


class StepHierarchyParser:
    """Parse internal names to understand pipeline hierarchy."""

    @staticmethod
    def parse_internal_name(internal_name: str) -> Dict[str, str]:
        """
        Parse internal name into components.

        Examples:
        - "hello" -> {"step": "hello"}
        - "parallel_step.branch1.hello_stub" -> {
            "composite": "parallel_step",
            "branch": "branch1",
            "step": "hello_stub"
          }
        """
        parts = internal_name.split(".")

        if len(parts) == 1:
            return {"step": parts[0]}
        elif len(parts) == 2:
            return {"composite": parts[0], "branch": parts[1]}
        elif len(parts) == 3:
            return {"composite": parts[0], "branch": parts[1], "step": parts[2]}
        else:
            # Handle deeper nesting if needed
            return {
                "composite": parts[0],
                "branch": ".".join(parts[1:-1]),
                "step": parts[-1],
            }

    @staticmethod
    def get_step_level(internal_name: str) -> int:
        """Determine hierarchy level from internal name."""
        parts = internal_name.split(".")
        if len(parts) == 1:
            return 0  # Top-level step
        elif len(parts) == 2:
            return 1  # Branch level (for composite step parent)
        else:
            return 2  # Branch step


class TimelineExtractor:
    """Extract chronological timeline from run log."""

    def __init__(self, run_log_data: Dict[str, Any]):
        self.run_log_data = run_log_data
        self.dag_nodes = (
            run_log_data.get("run_config", {}).get("dag", {}).get("nodes", {})
        )

    def parse_time(self, time_str: str) -> Optional[datetime]:
        """Parse ISO timestamp string."""
        try:
            return datetime.fromisoformat(time_str) if time_str else None
        except (ValueError, TypeError):
            return None

    def get_step_timing(
        self, step_data: Dict[str, Any]
    ) -> Tuple[Optional[datetime], Optional[datetime], float]:
        """Extract timing from step attempts."""
        attempts = step_data.get("attempts", [])
        if not attempts:
            return None, None, 0

        attempt = attempts[0]
        start = self.parse_time(attempt.get("start_time"))
        end = self.parse_time(attempt.get("end_time"))

        if start and end:
            duration_ms = (end - start).total_seconds() * 1000
            return start, end, duration_ms

        return None, None, 0

    def find_dag_node(self, internal_name: str, clean_name: str) -> Dict[str, Any]:
        """Find DAG node info for command details."""
        # Try direct lookup first
        if clean_name in self.dag_nodes:
            return self.dag_nodes[clean_name]

        # For composite steps, look in branch structures
        hierarchy = StepHierarchyParser.parse_internal_name(internal_name)
        if "composite" in hierarchy:
            composite_node = self.dag_nodes.get(hierarchy["composite"], {})
            if composite_node.get("is_composite"):
                branches = composite_node.get("branches", {})
                branch_key = hierarchy.get("branch", "")

                if branch_key in branches:
                    branch_nodes = branches[branch_key].get("nodes", {})
                    if clean_name in branch_nodes:
                        return branch_nodes[clean_name]
                # For map nodes, the branch structure might be different
                elif "branch" in composite_node:  # Map node structure
                    branch_nodes = composite_node["branch"].get("nodes", {})
                    if clean_name in branch_nodes:
                        return branch_nodes[clean_name]

        return {}

    def _format_parameter_value(self, value: Any, kind: str) -> str:
        """Format parameter value for display."""
        if kind == "metric":
            if isinstance(value, (int, float)):
                return f"{value:.3g}"
            return str(value)

        if isinstance(value, str):
            # Truncate long strings
            if len(value) > 50:
                return f'"{value[:47]}..."'
            return f'"{value}"'
        elif isinstance(value, (list, tuple)):
            if len(value) > 3:
                preview = ", ".join(str(v) for v in value[:3])
                return f"[{preview}, ...+{len(value)-3}]"
            return str(value)
        elif isinstance(value, dict):
            if len(value) > 2:
                keys = list(value.keys())[:2]
                preview = ", ".join(f'"{k}": {value[k]}' for k in keys)
                return f"{{{preview}, ...+{len(value)-2}}}"
            return str(value)
        else:
            return str(value)

    def extract_timeline(self) -> List[StepInfo]:
        """Extract all steps in chronological order."""
        steps = []

        # Process top-level steps
        for step_name, step_data in self.run_log_data.get("steps", {}).items():
            step_info = self._create_step_info(step_name, step_data)
            steps.append(step_info)

            # Process branches if they exist
            branches = step_data.get("branches", {})
            for branch_name, branch_data in branches.items():
                # Add branch steps
                for sub_step_name, sub_step_data in branch_data.get(
                    "steps", {}
                ).items():
                    sub_step_info = self._create_step_info(
                        sub_step_name,
                        sub_step_data,
                        parent=step_name,
                        branch=branch_name,
                    )
                    steps.append(sub_step_info)

        # Sort by start time for chronological order
        return sorted(steps, key=lambda x: x.start_time or datetime.min)

    def _create_step_info(
        self,
        step_name: str,
        step_data: Dict[str, Any],
        parent: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> StepInfo:
        """Create StepInfo from raw step data."""
        internal_name = step_data.get("internal_name", step_name)
        clean_name = step_data.get("name", step_name)

        # Get timing
        start, end, duration = self.get_step_timing(step_data)

        # Get command info from DAG
        dag_node = self.find_dag_node(internal_name, clean_name)
        command = dag_node.get("command", "")
        command_type = dag_node.get("command_type", "")

        # Extract parameters with detailed metadata (exclude pickled/object types)
        input_params = []
        output_params = []
        catalog_ops: Dict[str, List[str]] = {"put": [], "get": []}

        attempts = step_data.get("attempts", [])
        if attempts:
            attempt = attempts[0]
            input_param_data = attempt.get("input_parameters", {})
            output_param_data = attempt.get("output_parameters", {})

            # Process input parameters (exclude object/pickled types)
            for name, param in input_param_data.items():
                if isinstance(param, dict):
                    kind = param.get("kind", "")
                    if kind in ("json", "metric"):
                        value = param.get("value", "")
                        # Format value for display
                        formatted_value = self._format_parameter_value(value, kind)
                        input_params.append(f"{name}={formatted_value}")
                    # Skip object/pickled parameters entirely

            # Process output parameters (exclude object/pickled types)
            for name, param in output_param_data.items():
                if isinstance(param, dict):
                    kind = param.get("kind", "")
                    if kind in ("json", "metric"):
                        value = param.get("value", "")
                        # Format value for display
                        formatted_value = self._format_parameter_value(value, kind)
                        output_params.append(f"{name}={formatted_value}")
                    # Skip object/pickled parameters entirely

        # Extract catalog operations
        catalog_data = step_data.get("data_catalog", [])
        for item in catalog_data:
            stage = item.get("stage", "")
            name = item.get("name", "")
            if stage == "put":
                catalog_ops["put"].append(name)
            elif stage == "get":
                catalog_ops["get"].append(name)

        return StepInfo(
            name=clean_name,
            internal_name=internal_name,
            status=step_data.get("status", "UNKNOWN"),
            step_type=step_data.get("step_type", "task"),
            start_time=start,
            end_time=end,
            duration_ms=duration,
            level=StepHierarchyParser.get_step_level(internal_name),
            parent=parent,
            branch=branch,
            command=command,
            command_type=command_type,
            input_params=input_params,
            output_params=output_params,
            catalog_ops=catalog_ops,
        )


class SimpleVisualizer:
    """Simple, lightweight pipeline visualizer."""

    def __init__(self, run_log_path: Union[str, Path]):
        self.run_log_path = Path(run_log_path)
        self.run_log_data = self._load_run_log()
        self.extractor = TimelineExtractor(self.run_log_data)
        self.timeline = self.extractor.extract_timeline()

    def _load_run_log(self) -> Dict[str, Any]:
        """Load run log JSON."""
        if not self.run_log_path.exists():
            raise FileNotFoundError(f"Run log not found: {self.run_log_path}")

        with open(self.run_log_path, "r") as f:
            return json.load(f)

    def print_simple_timeline(self) -> None:
        """Print a clean console timeline."""
        run_id = self.run_log_data.get("run_id", "unknown")
        status = self.run_log_data.get("status", "UNKNOWN")

        print(f"\n🔄 Pipeline Timeline - {run_id}")
        print(f"Status: {status}")
        print("=" * 80)

        # Group by composite steps for better display
        current_composite = None
        current_branch = None

        for step in self.timeline:
            # Skip composite steps themselves (they have no timing)
            if (
                step.step_type in ["parallel", "map", "conditional"]
                and not step.start_time
            ):
                continue

            # Detect composite/branch changes
            hierarchy = StepHierarchyParser.parse_internal_name(step.internal_name)
            composite = hierarchy.get("composite")
            branch = hierarchy.get("branch")

            # Show composite header
            if composite and composite != current_composite:
                print(f"\n🔀 {composite} ({self._get_composite_type(composite)})")
                current_composite = composite
                current_branch = None

            # Show branch header
            if branch and branch != current_branch:
                branch_display = self._format_branch_name(composite or "", branch)
                print(f"  ├─ Branch: {branch_display}")
                current_branch = branch

            # Show step
            indent = (
                "  " if step.level == 0 else "    " if step.level == 1 else "      "
            )
            status_emoji = (
                "✅"
                if step.status == "SUCCESS"
                else "❌"
                if step.status == "FAIL"
                else "⏸️"
            )

            # Type icon
            type_icons = {
                "task": "⚙️",
                "stub": "📝",
                "success": "✅",
                "fail": "❌",
                "parallel": "🔀",
                "map": "🔁",
                "conditional": "🔀",
            }
            type_icon = type_icons.get(step.step_type, "⚙️")

            timing = f"({step.duration_ms:.1f}ms)" if step.duration_ms > 0 else ""

            print(f"{indent}{type_icon} {status_emoji} {step.name} {timing}")

            # Show metadata for tasks
            if step.step_type == "task" and (
                step.command
                or step.input_params
                or step.output_params
                or step.catalog_ops["put"]
                or step.catalog_ops["get"]
            ):
                if step.command:
                    cmd_short = (
                        step.command[:50] + "..."
                        if len(step.command) > 50
                        else step.command
                    )
                    print(f"{indent}   📝 {step.command_type.upper()}: {cmd_short}")

                # Show input parameters - compact horizontal display
                if step.input_params:
                    params_display = " • ".join(step.input_params)
                    print(f"{indent}   📥 {params_display}")

                # Show output parameters - compact horizontal display
                if step.output_params:
                    params_display = " • ".join(step.output_params)
                    print(f"{indent}   📤 {params_display}")

                # Show catalog operations - compact horizontal display
                if step.catalog_ops.get("put") or step.catalog_ops.get("get"):
                    catalog_items = []
                    if step.catalog_ops.get("put"):
                        catalog_items.extend(
                            [f"PUT:{item}" for item in step.catalog_ops["put"]]
                        )
                    if step.catalog_ops.get("get"):
                        catalog_items.extend(
                            [f"GET:{item}" for item in step.catalog_ops["get"]]
                        )
                    if catalog_items:
                        catalog_display = " • ".join(catalog_items)
                        print(f"{indent}   💾 {catalog_display}")

        print("=" * 80)

    def _get_composite_type(self, composite_name: str) -> str:
        """Get composite node type from DAG."""
        dag_nodes = (
            self.run_log_data.get("run_config", {}).get("dag", {}).get("nodes", {})
        )
        node = dag_nodes.get(composite_name, {})
        return node.get("node_type", "composite")

    def _format_branch_name(self, composite: str, branch: str) -> str:
        """Format branch name based on composite type."""
        # Remove composite prefix if present
        if branch.startswith(f"{composite}."):
            branch_clean = branch[len(f"{composite}.") :]
        else:
            branch_clean = branch

        # Check if it's a map iteration (numeric)
        if branch_clean.isdigit():
            return f"Iteration {branch_clean}"

        return branch_clean

    def print_execution_summary(self) -> None:
        """Print execution summary table."""
        run_id = self.run_log_data.get("run_id", "unknown")

        print(f"\n📊 Execution Summary - {run_id}")
        print("=" * 80)

        # Filter to actual executed steps (with timing)
        executed_steps = [step for step in self.timeline if step.start_time]

        if not executed_steps:
            print("No executed steps found")
            return

        # Table header
        print(f"{'Step':<30} {'Status':<10} {'Duration':<12} {'Type':<10}")
        print("-" * 80)

        total_duration = 0
        success_count = 0

        for step in executed_steps:
            status_emoji = (
                "✅"
                if step.status == "SUCCESS"
                else "❌"
                if step.status == "FAIL"
                else "⏸️"
            )
            duration_text = (
                f"{step.duration_ms:.1f}ms" if step.duration_ms > 0 else "0.0ms"
            )

            # Truncate long names
            display_name = step.name[:28] + ".." if len(step.name) > 30 else step.name

            print(
                f"{display_name:<30} {status_emoji}{step.status:<9} {duration_text:<12} {step.step_type:<10}"
            )

            total_duration += int(step.duration_ms)
            if step.status == "SUCCESS":
                success_count += 1

        print("-" * 80)
        success_rate = (
            (success_count / len(executed_steps)) * 100 if executed_steps else 0
        )
        overall_status = self.run_log_data.get("status", "UNKNOWN")
        overall_emoji = "✅" if overall_status == "SUCCESS" else "❌"

        print(
            f"Total Duration: {total_duration:.1f}ms | Success Rate: {success_rate:.1f}% | Status: {overall_emoji} {overall_status}"
        )

    def generate_html_timeline(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate an interactive HTML timeline visualization.

        This creates a lightweight HTML version with:
        - Clean timeline layout
        - Hover tooltips with metadata
        - Expandable composite sections
        - Timing bars proportional to execution duration
        """
        run_id = self.run_log_data.get("run_id", "unknown")
        status = self.run_log_data.get("status", "UNKNOWN")

        # Calculate total timeline for proportional bars
        executed_steps = [step for step in self.timeline if step.start_time]
        if executed_steps:
            earliest = min(
                step.start_time for step in executed_steps if step.start_time
            )
            latest = max(step.end_time for step in executed_steps if step.end_time)
            total_duration_ms = (
                (latest - earliest).total_seconds() * 1000 if latest and earliest else 1
            )
        else:
            total_duration_ms = 1

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Timeline - {run_id}</title>
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

        .timeline-card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            overflow: hidden;
            margin-bottom: 2rem;
        }}

        .timeline-header {{
            background: #f8fafc;
            padding: 1.5rem;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .timeline-content {{
            padding: 1rem;
        }}

        .step-row {{
            display: grid;
            grid-template-columns: 300px 1fr 80px;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid #f1f5f9;
            transition: background 0.2s ease;
            gap: 1rem;
            min-height: 40px;
            overflow: visible;
        }}

        .step-row:hover {{
            background: #f8fafc;
        }}

        .step-info {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
            font-weight: 500;
            min-height: 24px;
            justify-content: flex-start;
            overflow: visible;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }}

        .step-level-0 {{ padding-left: 0; }}
        .step-level-1 {{ padding-left: 1rem; }}
        .step-level-2 {{ padding-left: 2rem; }}

        .composite-header {{
            background: #e0f2fe !important;
            border-left: 4px solid #0277bd;
            font-weight: 600;
            color: #01579b;
        }}

        .branch-header {{
            background: #f3e5f5 !important;
            border-left: 4px solid #7b1fa2;
            font-weight: 600;
            color: #4a148c;
        }}

        .gantt-container {{
            position: relative;
            height: 30px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            min-width: 100%;
            overflow: hidden;
        }}

        .gantt-bar {{
            position: absolute;
            top: 3px;
            height: 24px;
            border-radius: 3px;
            transition: all 0.2s ease;
            cursor: pointer;
            border: 1px solid rgba(255,255,255,0.3);
        }}

        .gantt-bar:hover {{
            transform: scaleY(1.1);
            z-index: 10;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}

        .time-grid {{
            position: absolute;
            top: 0;
            bottom: 0;
            border-left: 1px solid #e2e8f0;
            opacity: 0.3;
        }}

        .time-scale {{
            position: relative;
            height: 20px;
            background: #f1f5f9;
            border-bottom: 1px solid #d1d5db;
            font-size: 0.75rem;
            color: #6b7280;
        }}

        .time-marker {{
            position: absolute;
            top: 0;
            height: 100%;
            display: flex;
            align-items: center;
            padding-left: 4px;
            font-weight: 500;
        }}

        .timeline-bar:hover {{
            transform: scaleY(1.1);
            z-index: 10;
        }}

        .bar-success {{ background: linear-gradient(90deg, #22c55e, #16a34a); }}
        .bar-fail {{ background: linear-gradient(90deg, #ef4444, #dc2626); }}
        .bar-unknown {{ background: linear-gradient(90deg, #f59e0b, #d97706); }}

        .duration-text {{
            font-family: monospace;
            font-size: 0.875rem;
            font-weight: 600;
        }}

        .duration-fast {{ color: #16a34a; }}
        .duration-medium {{ color: #f59e0b; }}
        .duration-slow {{ color: #dc2626; }}

        .status-success {{ color: #16a34a; }}
        .status-fail {{ color: #dc2626; }}
        .status-unknown {{ color: #f59e0b; }}

        .expandable {{
            cursor: pointer;
            user-select: none;
        }}

        .expandable:hover {{
            background: #e2e8f0 !important;
        }}

        .step-header.expandable {{
            padding: 0.25rem;
            border-radius: 4px;
            margin: -0.25rem;
        }}

        .step-header.expandable:hover {{
            background: #f1f5f9 !important;
        }}

        .collapsible-content {{
            max-height: none;
            overflow: visible;
            transition: max-height 0.3s ease;
        }}

        .collapsible-content.collapsed {{
            max-height: 0;
            overflow: hidden;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }}

        .summary-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            text-align: center;
        }}

        .summary-number {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}

        .summary-label {{
            color: #64748b;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 Pipeline Timeline Visualization</h1>
        <p>Interactive execution analysis for {run_id}</p>
    </div>

    <div class="container">
        <div class="timeline-card">
            <div class="timeline-header">
                <div>
                    <h2>Run ID: {run_id}</h2>
                    <p>Status: <span class="status-{status.lower()}">{status}</span></p>
                </div>
                <div>
                    <span class="duration-text">Total: {total_duration_ms:.1f}ms</span>
                </div>
            </div>

            <div class="timeline-content">
                {self._generate_html_timeline_rows(total_duration_ms)}
            </div>
        </div>

        {self._generate_html_summary()}
    </div>

    <script>
        // Collapsible sections
        document.querySelectorAll('.expandable').forEach(element => {{
            element.addEventListener('click', () => {{
                let content;

                // Handle step metadata (using data-target)
                if (element.dataset.target) {{
                    content = document.getElementById(element.dataset.target);
                }} else {{
                    // Handle composite sections (using nextElementSibling)
                    content = element.nextElementSibling;
                }}

                if (content && content.classList.contains('collapsible-content')) {{
                    content.classList.toggle('collapsed');

                    // Update expand/collapse indicator
                    const indicator = element.querySelector('.expand-indicator');
                    if (indicator) {{
                        indicator.textContent = content.classList.contains('collapsed') ? '▶' : '▼';
                    }}
                }}
            }});
        }});
    </script>
</body>
</html>"""

        # Save to file if path provided
        if output_path:
            Path(output_path).write_text(html_content)
            print(f"HTML timeline saved to: {output_path}")

        return html_content

    def _generate_html_timeline_rows(self, total_duration_ms: float) -> str:
        """Generate HTML rows for the Gantt chart timeline display."""
        executed_steps = [
            step for step in self.timeline if step.start_time and step.end_time
        ]

        if not executed_steps:
            return "<div>No executed steps found</div>"

        # Calculate the absolute timeline
        earliest_start = min(
            step.start_time for step in executed_steps if step.start_time
        )
        latest_end = max(step.end_time for step in executed_steps if step.end_time)
        total_timeline_ms = (
            (latest_end - earliest_start).total_seconds() * 1000
            if latest_end and earliest_start
            else 1
        )

        # Generate time scale and Gantt rows
        time_scale_html = self._generate_time_scale(total_timeline_ms)
        gantt_rows_html = self._generate_gantt_rows(
            executed_steps, earliest_start, total_timeline_ms
        )

        return time_scale_html + "\n" + gantt_rows_html

    def _generate_time_scale(self, total_timeline_ms: float) -> str:
        """Generate the time scale header for the Gantt chart."""
        # Create time markers at regular intervals
        num_markers = 10
        interval_ms = total_timeline_ms / num_markers

        markers_html = []
        for i in range(num_markers + 1):
            time_ms = i * interval_ms
            position_percent = (time_ms / total_timeline_ms) * 100
            time_display = (
                f"{time_ms:.0f}ms" if time_ms < 1000 else f"{time_ms/1000:.1f}s"
            )

            markers_html.append(f"""
                <div class="time-marker" style="left: {position_percent:.1f}%;">
                    {time_display}
                </div>
            """)

            # Add grid line (except for the first one)
            if i > 0:
                markers_html.append(
                    f'<div class="time-grid" style="left: {position_percent:.1f}%;"></div>'
                )

        return f"""
            <div class="step-row" style="border-bottom: 2px solid #d1d5db;">
                <div class="step-info">
                    <strong>Timeline</strong>
                </div>
                <div class="time-scale">
                    {"".join(markers_html)}
                </div>
                <div></div>
            </div>
        """

    def _generate_gantt_rows(
        self, executed_steps: List, earliest_start, total_timeline_ms: float
    ) -> str:
        """Generate HTML rows for the Gantt chart display."""
        html_parts = []

        # Group by composite steps for better display
        current_composite = None
        current_branch = None

        for step in executed_steps:
            # Calculate timing positions for Gantt chart
            start_offset_ms = (step.start_time - earliest_start).total_seconds() * 1000
            start_percent = (start_offset_ms / total_timeline_ms) * 100
            width_percent = (step.duration_ms / total_timeline_ms) * 100

            # Detect composite/branch changes
            hierarchy = StepHierarchyParser.parse_internal_name(step.internal_name)
            composite = hierarchy.get("composite")
            branch = hierarchy.get("branch")

            # Show composite header
            if composite and composite != current_composite:
                composite_type = self._get_composite_type(composite)
                composite_id = f"composite-{composite.replace(' ', '-')}"

                html_parts.append(f"""
                    <div class="step-row composite-header expandable" data-composite="{composite}">
                        <div class="step-info step-level-0">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span class="expand-indicator">▼</span>
                                🔀 <strong>{composite}</strong> ({composite_type})
                            </div>
                        </div>
                        <div class="gantt-container"></div>
                        <div></div>
                    </div>
                """)

                # Start collapsible content
                html_parts.append(
                    f'<div class="collapsible-content" id="{composite_id}">'
                )
                current_composite = composite
                current_branch = None

            # Show branch header for parallel/map steps
            if branch and branch != current_branch:
                branch_display = self._format_branch_name(composite or "", branch)

                html_parts.append(f"""
                    <div class="step-row branch-header">
                        <div class="step-info step-level-1">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                🌿 <strong>Branch: {branch_display}</strong>
                            </div>
                        </div>
                        <div class="gantt-container"></div>
                        <div></div>
                    </div>
                """)
                current_branch = branch

            # Status styling
            status_class = step.status.lower()
            bar_class = (
                f"bar-{status_class}"
                if status_class in ["success", "fail"]
                else "bar-unknown"
            )

            # Type icon and status emoji
            type_icons = {
                "task": "⚙️",
                "stub": "📝",
                "success": "✅",
                "fail": "❌",
                "parallel": "🔀",
                "map": "🔁",
                "conditional": "🔀",
            }
            type_icon = type_icons.get(step.step_type, "⚙️")
            status_emoji = (
                "✅"
                if step.status == "SUCCESS"
                else "❌"
                if step.status == "FAIL"
                else "⏸️"
            )

            # Build parameter display - compact horizontal format
            param_info = []

            if step.input_params:
                params_text = " • ".join(step.input_params)
                param_info.append(
                    f'<div style="color: #059669; font-size: 0.7rem; margin-top: 0.2rem; font-family: monospace; word-break: break-all; line-height: 1.3;">📥 {params_text}</div>'
                )

            if step.output_params:
                params_text = " • ".join(step.output_params)
                param_info.append(
                    f'<div style="color: #dc2626; font-size: 0.7rem; margin-top: 0.2rem; font-family: monospace; word-break: break-all; line-height: 1.3;">📤 {params_text}</div>'
                )

            if step.catalog_ops.get("put") or step.catalog_ops.get("get"):
                catalog_items = []
                if step.catalog_ops.get("put"):
                    catalog_items.extend(
                        [f"PUT:{item}" for item in step.catalog_ops["put"]]
                    )
                if step.catalog_ops.get("get"):
                    catalog_items.extend(
                        [f"GET:{item}" for item in step.catalog_ops["get"]]
                    )
                if catalog_items:
                    catalog_text = " • ".join(catalog_items)
                    param_info.append(
                        f'<div style="color: #7c3aed; font-size: 0.7rem; margin-top: 0.2rem; font-family: monospace; word-break: break-all; line-height: 1.3;">💾 {catalog_text}</div>'
                    )

            # Create unique ID for this step's metadata
            step_id = f"step-{step.internal_name.replace('.', '-')}-{step.start_time.isoformat()}"

            html_parts.append(f"""
                <div class="step-row">
                    <div class="step-info step-level-{step.level}">
                        <div style="display: flex; align-items: center; gap: 0.5rem;" class="step-header expandable" data-target="{step_id}">
                            <span class="expand-indicator" style="font-size: 0.8rem; color: #6b7280;">{'▼' if param_info else ''}</span>
                            {type_icon} {status_emoji} <strong>{step.name}</strong>
                        </div>
                        {f'<div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;"><strong>{step.command_type.upper()}:</strong> {step.command[:40]}{"..." if len(step.command) > 40 else ""}</div>' if step.command else ''}
                        <div class="step-metadata collapsible-content collapsed" id="{step_id}">
                            {''.join(param_info)}
                        </div>
                    </div>
                    <div class="gantt-container">
                        <div class="gantt-bar {bar_class}"
                             style="left: {start_percent:.2f}%; width: {max(width_percent, 0.5):.2f}%;"
                             title="{step.name}: {step.duration_ms:.1f}ms">
                        </div>
                    </div>
                    <div style="font-family: monospace; font-size: 0.75rem; color: #6b7280;">
                        {step.duration_ms:.1f}ms
                    </div>
                </div>
            """)

        # Close any open composite sections
        if current_composite:
            html_parts.append("</div>")  # Close collapsible-content

        return "\n".join(html_parts)

    def _generate_html_summary(self) -> str:
        """Generate HTML summary cards."""
        executed_steps = [step for step in self.timeline if step.start_time]
        total_duration = sum(step.duration_ms for step in executed_steps)
        success_count = sum(1 for step in executed_steps if step.status == "SUCCESS")
        success_rate = (
            (success_count / len(executed_steps)) * 100 if executed_steps else 0
        )

        # Find slowest step
        slowest_step = (
            max(executed_steps, key=lambda x: x.duration_ms) if executed_steps else None
        )

        return f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-number status-success">{len(executed_steps)}</div>
                <div class="summary-label">Total Steps</div>
            </div>
            <div class="summary-card">
                <div class="summary-number duration-medium">{total_duration:.1f}ms</div>
                <div class="summary-label">Total Duration</div>
            </div>
            <div class="summary-card">
                <div class="summary-number {'status-success' if success_rate == 100 else 'status-fail'}">{success_rate:.1f}%</div>
                <div class="summary-label">Success Rate</div>
            </div>
            <div class="summary-card">
                <div class="summary-number duration-slow">{'%.1fms' % slowest_step.duration_ms if slowest_step else 'N/A'}</div>
                <div class="summary-label">Slowest Step<br><small>{slowest_step.name if slowest_step else 'N/A'}</small></div>
            </div>
        </div>
        """


def visualize_simple(
    run_id: str, show_summary: bool = False, output_html: Optional[str] = None
) -> None:
    """
    Simple visualization of a pipeline run.

    Args:
        run_id: Run ID to visualize
        show_summary: Whether to show execution summary (deprecated, timeline has enough info)
        output_html: Optional path to save HTML timeline
    """
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

    print(f"📊 Visualizing: {log_file.stem}")

    viz = SimpleVisualizer(log_file)
    viz.print_simple_timeline()

    if show_summary:
        viz.print_execution_summary()

    # Generate HTML if requested
    if output_html:
        viz.generate_html_timeline(output_html)


def generate_html_timeline(
    run_id: str, output_file: str, open_browser: bool = True
) -> None:
    """
    Generate HTML timeline for a specific run ID.

    Args:
        run_id: The run ID to visualize
        output_file: Output HTML file path
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

    print(f"🌐 Generating HTML timeline for: {log_file.stem}")

    # Create visualizer and generate HTML
    viz = SimpleVisualizer(log_file)
    viz.generate_html_timeline(output_file)

    if open_browser:
        import webbrowser

        file_path = Path(output_file).absolute()
        print(f"🌐 Opening timeline in browser: {file_path.name}")
        webbrowser.open(file_path.as_uri())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if len(sys.argv) > 2 and sys.argv[2].endswith(".html"):
            # Generate HTML: python viz_simple.py <run_id> <output.html>
            generate_html_timeline(sys.argv[1], sys.argv[2])
        else:
            # Console visualization: python viz_simple.py <run_id>
            visualize_simple(sys.argv[1])
    else:
        print("Usage: python viz_simple.py <run_id> [output.html]")

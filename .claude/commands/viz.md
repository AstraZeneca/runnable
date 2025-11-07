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

Remember: Keep visualization features lightweight, simple, and focused on core developer needs.

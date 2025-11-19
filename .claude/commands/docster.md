# Documentation Development

You are helping with documentation development for the Runnable framework. Focus on maintaining consistency with the existing documentation structure and style.

## Context

The Runnable framework documentation is built with MkDocs and follows established patterns. Your role is to extend and improve the documentation while maintaining the current structure and style.

## Key Principles

- **Maintain existing structure**: Follow the current documentation organization and layout
- **Consistent style**: Match the tone, formatting, and presentation style of existing docs
- **Code from examples**: Always use working code examples from the `examples/` directory
- **Python focus**: Prioritize Python API examples over YAML (YAML is being deprecated)
- **Progressive complexity**: Layer examples from simple to complex following the examples structure
- **Executable patterns**: All code examples must follow the executable patterns used in actual examples

## Documentation Guidelines

### Content Creation
- Use the same formatting patterns as existing documentation
- Reference working examples from `examples/` directory
- Maintain the current section organization and navigation structure
- Follow established naming conventions and terminology
- Keep the same level of detail and explanation depth

### Code Examples
When adding documentation:
1. Use existing examples from `examples/` folder
2. Show `uv run` commands as demonstrated in current docs
3. Follow the same code block formatting and syntax highlighting
4. Include the same types of explanations and context as existing sections
5. Match the balance between overview and detailed examples

#### ⚠️ CRITICAL: Correct Code Patterns
**ALL documentation code examples must follow the executable pattern used in actual examples:**

```python
from runnable import PythonJob, Pipeline, PythonTask

def main():
    # Job or pipeline setup and execution
    job = PythonJob(function=my_function)
    job.execute()
    return job  # Always return the job/pipeline object

if __name__ == "__main__":
    main()
```

**❌ NEVER use these misleading patterns in documentation:**
```python
# WRONG - Don't do this in docs
from runnable import PythonJob
job = PythonJob(function=my_function)
job.execute()
```

**✅ ALWAYS use the correct executable pattern:**
```python
# CORRECT - Always do this
from runnable import PythonJob

def main():
    job = PythonJob(function=my_function)
    job.execute()
    return job

if __name__ == "__main__":
    main()
```

### Structure Consistency
- Follow the same heading hierarchy and organization
- Use identical markdown formatting patterns
- Maintain consistent cross-referencing style
- Keep the same approach to code snippets and callouts
- Preserve the current flow from concept to implementation

## Development Approach

### When extending documentation:
1. Review existing docs to understand the current style and structure
2. Find relevant examples in `examples/` directory that match the existing pattern
3. Run examples with `uv run` to understand behavior
4. Write documentation that seamlessly fits with existing content
5. Use the same mkdocs configuration and formatting

### Areas to Focus On:
- Fill gaps in existing documentation sections
- Add missing examples that follow established patterns
- Improve clarity while maintaining the current voice and style
- Extend existing sections with additional use cases
- Add troubleshooting sections using the same format as existing ones

## Documentation Maintenance

### Quality Assurance
When working on documentation, always verify:

1. **Pattern Consistency**: All code examples follow the `main()` function pattern
2. **Link Integrity**: All internal links point to existing documentation sections
3. **Example Verification**: Referenced examples actually exist in the `examples/` directory
4. **Executable Code**: All code snippets can actually be run by users
5. **Navigation Alignment**: New content fits into the existing mkdocs navigation structure

### Common Issues to Avoid
- ❌ Code examples that don't follow the established executable pattern
- ❌ Links to non-existent files (check with mkdocs serve for warnings)
- ❌ References to deprecated YAML patterns instead of Python API
- ❌ Inconsistent heading styles or markdown formatting
- ❌ Examples that haven't been tested with `uv run`

### Testing Documentation Changes
1. Always run `uv run mkdocs serve` to check for warnings
2. Test referenced examples with `uv run examples/path/to/example.py`
3. Verify all internal links work correctly
4. Ensure code examples are copy-pasteable and executable

### Lessons Learned from Recent Fixes
Based on recent documentation improvements, pay special attention to:

1. **Code Pattern Consistency**: The biggest issue was inline code examples that didn't match the actual executable patterns in the `examples/` directory
2. **Link Maintenance**: Several broken links existed due to:
   - References to non-existent `concepts/task.md` (should point to `concepts/building-blocks/task-types.md`)
   - Missing concept files that were referenced but never created
   - Self-referencing links that made no sense
3. **Unused Files**: Remove files that aren't in the mkdocs navigation structure
4. **Anchor Accuracy**: Ensure links to page sections use the correct anchor format
5. **README Consistency**: Keep README.md examples consistent with documentation patterns

## Key Reminders
- Preserve the current documentation structure and organization
- Match the existing writing style and tone
- Use working code from examples directory
- Follow established formatting and presentation patterns
- Maintain consistency with existing cross-references and navigation
- **CRITICAL**: All code examples must use the `main()` function pattern

## Recent Documentation Updates (2024)

### Jobs Documentation Structure
- **NEW**: Jobs now have dedicated documentation with progressive complexity structure:
  - `jobs/first-job.md` - Basic execution and concepts
  - `jobs/working-with-data.md` - Return values and data storage
  - `jobs/parameters.md` - Configuration without code changes
  - `jobs/file-storage.md` - Catalog system for file management
  - `jobs/job-types.md` - Shell, Notebook, and Python Jobs
- **Navigation**: "Core Concepts" renamed to "Pipelines" for clarity
- **Progressive Learning**: Each Jobs page builds on previous concepts

### Catalog System Understanding
- **File Storage**: Both Jobs and Pipelines support `Catalog(put=[...])` for file storage
- **No-Copy Mode**: `store_copy=False` captures MD5 hash without copying files
  - Use for large datasets (GB+ files) where copying is expensive
  - Files remain in original location, tracked by hash for integrity
  - Saves disk space and improves performance
- **Glob Patterns**: Currently supported (`"*.csv"`, `"plots/*.png"`, `"data/**/*.json"`)
- **Return Types**: `pickled()`, `metric()`, and JSON serialization for different data types

### Parameter System
- **Three-Layer Precedence**: Jobs inherit same parameter system as Pipelines
  1. `RUNNABLE_PRM_key="value"` (highest priority)
  2. `RUNNABLE_PARAMETERS_FILE="config.yaml"`
  3. `job.execute(parameters_file="config.yaml")` (lowest priority)
- **Environment Variables**: Always override YAML values

### Documentation Formatting
- **Markdown Lists**: Always add blank line between headings and lists
- **Emoji Usage**: Conservative use only for key section headers (📦, ⚙️, 📁, 🔍)
- **Code Patterns**: Only show patterns that exist in actual examples directory

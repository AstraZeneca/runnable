# FastAPI LLM Streaming Example

Demonstrates native async streaming from `AsyncPipeline` functions to SSE clients.

## Key Concepts

- **AsyncPipeline**: Pipeline that executes async functions natively
- **AsyncPythonTask**: Task wrapper for async functions
- **execute_streaming()**: Returns AsyncGenerator for event streaming
- **AsyncGenerator functions**: Yield events that flow to SSE clients

## Installation

This example requires FastAPI and uvicorn:

```bash
pip install fastapi uvicorn
```

## Running

```bash
uv run uvicorn examples.fastapi_llm.main:app --reload
```

## Testing

```bash
curl -N -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello!"}'
```

## Event Types

- `{"type": "status", "status": "thinking"}` - Processing status
- `{"type": "chunk", "text": "..."}` - Token/word chunk
- `{"type": "done", "full_text": "..."}` - Completion

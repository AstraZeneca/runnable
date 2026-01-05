# FastAPI LLM Streaming Example

Demonstrates native async streaming from `AsyncPipeline` functions to SSE clients, including parallel execution of translation tasks.

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

### Single LLM Response

```bash
curl -N -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello!"}'
```

### LLM Response + Summary

```bash
curl -N -X POST http://localhost:8000/chat-and-summarize \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Tell me about machine learning"}'
```

### LLM Response + Parallel Translation

```bash
curl -N -X POST http://localhost:8000/chat-and-translate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, how are you today?"}'
```

### LLM Response + Map Translation (Multiple Languages)

```bash
curl -N -X POST http://localhost:8000/chat-map-translate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, how are you today?", "languages": ["spanish", "french"]}'
```

### LLM Response + Conditional Translation (User Choice)

```bash
curl -N -X POST http://localhost:8000/chat-conditional-translate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, how are you today?", "language": "spanish"}'
```

## Event Types

### Common Events
- `{"type": "status", "status": "thinking"}` - Processing status
- `{"type": "chunk", "text": "..."}` - Token/word chunk
- `{"type": "done", "full_text": "..."}` - Completion

### Summary Events (chat-and-summarize)
- `{"type": "status", "status": "summarizing"}` - Summary processing
- `{"type": "done", "summary": "..."}` - Summary completion

### Translation Events (chat-and-translate)
- `{"type": "status", "status": "translating"}` - Translation processing
- `{"type": "chunk", "text": "traducido"}` - Translated word chunk
- `{"type": "done", "spanish_translation": "..."}` - Spanish completion
- `{"type": "done", "french_translation": "..."}` - French completion

Events include a `step_name` field to identify which translation branch is streaming.

### Map Events (chat-map-translate)
- `{"type": "status", "status": "translating"}` - Translation processing for current language
- `{"type": "chunk", "text": "traducido"}` - Translated word chunk for current language
- `{"type": "done", "translation_result": "..."}` - Translation completion for current language

Map iterates over requested languages and translates to each. Results are collected into arrays.

### Conditional Events (chat-conditional-translate)
- `{"type": "status", "status": "translating"}` - Translation processing for chosen language
- `{"type": "chunk", "text": "traducido"}` - Translated word chunk for chosen language
- `{"type": "done", "spanish_translation": "..."}` or `{"type": "done", "french_translation": "..."}` - Translation completion for chosen language

Conditional chooses one translation branch based on user's language preference.

import asyncio
import pytest
from runnable.context import PipelineContext, get_run_context, set_run_context


def test_context_isolation_sync():
    """Test that different contexts don't interfere in sync execution."""
    # Create first context
    context1 = PipelineContext(
        pipeline_definition_file="test1.py",
        run_id="test-run-1",
        catalog={"type": "file-system"},
        secrets={"type": "env-secrets"},
        pickler={"type": "pickle"},
        run_log_store={"type": "file-system"},
        pipeline_executor={"type": "local"}
    )

    # Create second context
    context2 = PipelineContext(
        pipeline_definition_file="test2.py",
        run_id="test-run-2",
        catalog={"type": "file-system"},
        secrets={"type": "env-secrets"},
        pickler={"type": "pickle"},
        run_log_store={"type": "file-system"},
        pipeline_executor={"type": "local"}
    )

    # Set context1, verify it's returned
    set_run_context(context1)
    assert get_run_context().run_id == "test-run-1"

    # Set context2, verify it overwrites (current behavior)
    set_run_context(context2)
    assert get_run_context().run_id == "test-run-2"


@pytest.mark.asyncio
async def test_context_isolation_async():
    """Test that async tasks maintain separate contexts."""
    results = []

    async def async_task(context_id: str):
        context = PipelineContext(
            pipeline_definition_file=f"test{context_id}.py",
            run_id=f"test-run-{context_id}",
            catalog={"type": "file-system"},
            secrets={"type": "env-secrets"},
            pickler={"type": "pickle"},
            run_log_store={"type": "file-system"},
            pipeline_executor={"type": "local"}
        )
        set_run_context(context)

        # Each task should maintain its own context
        current_context = get_run_context()
        assert current_context is not None
        assert current_context.run_id == f"test-run-{context_id}"

        # Simulate some async work
        await asyncio.sleep(0.1)

        # Context should still be correct after await
        current_context = get_run_context()
        assert current_context.run_id == f"test-run-{context_id}"
        results.append(context_id)

    # Run multiple async tasks concurrently
    await asyncio.gather(
        async_task("1"),
        async_task("2"),
        async_task("3")
    )

    assert len(results) == 3
    assert set(results) == {"1", "2", "3"}

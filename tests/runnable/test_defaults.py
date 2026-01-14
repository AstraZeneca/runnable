def test_loop_placeholder_constant():
    """Test LOOP_PLACEHOLDER constant exists."""
    from runnable.defaults import LOOP_PLACEHOLDER

    assert LOOP_PLACEHOLDER == "loop_variable_placeholder"
    assert isinstance(LOOP_PLACEHOLDER, str)

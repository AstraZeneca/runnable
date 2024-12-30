from extensions.pipeline_executor.local import LocalExecutor


def test_local_executor_execute_node_just_calls___execute_node(mocker, monkeypatch):
    mock__execute_node = mocker.MagicMock()

    monkeypatch.setattr(LocalExecutor, "_execute_node", mock__execute_node)
    executor = LocalExecutor()

    mock_node = mocker.MagicMock()

    executor.execute_node(mock_node)

    assert mock__execute_node.call_count == 1

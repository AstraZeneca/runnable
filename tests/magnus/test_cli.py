import pytest

from magnus import cli


def test_init_prints_help_if_command_is_not_recognised(mocker, monkeypatch):
    mock_argparse = mocker.MagicMock()

    monkeypatch.setattr(cli, 'argparse', mock_argparse)
    monkeypatch.setattr(cli.sys, 'argv', ['magnus', 'command_to_run'])

    mock_parser = mocker.MagicMock()
    mock_argparse.ArgumentParser.return_value = mock_parser

    mock_command = mocker.MagicMock()
    mock_parser.parse_args.return_value = mock_command
    mock_command.command = 'command_to_run'

    mock_help = mocker.MagicMock()
    mock_parser.print_help = mock_help

    with pytest.raises(SystemExit):
        cli.MagnusCLI()

    assert mock_help.call_count == 1


def test_init_calls_the_method_if_command_recognised(mocker, monkeypatch):
    mock_argparse = mocker.MagicMock()

    monkeypatch.setattr(cli, 'argparse', mock_argparse)
    monkeypatch.setattr(cli.sys, 'argv', ['magnus', 'command_to_run'])

    mock_parser = mocker.MagicMock()
    mock_argparse.ArgumentParser.return_value = mock_parser

    mock_command = mocker.MagicMock()
    mock_parser.parse_args.return_value = mock_command
    mock_command.command = 'dummy_function'

    mock_function = mocker.MagicMock()
    cli.MagnusCLI.dummy_function = mock_function

    cli.MagnusCLI()

    assert mock_function.call_count == 1


def test_execute_raises_exception_if_use_cached_is_used_without_run_id(mocker, monkeypatch):
    mock_argparse = mocker.MagicMock()
    monkeypatch.setattr(cli, 'argparse', mock_argparse)

    mock_resolve_args = mocker.MagicMock()
    monkeypatch.setattr(cli.MagnusCLI, '_resolve_args', mock_resolve_args)

    monkeypatch.setattr(cli.MagnusCLI, '__init__', mocker.MagicMock(return_value=None))

    mock_args = mocker.MagicMock()
    mock_args.use_cached = True
    mock_args.run_id = None
    mock_resolve_args.return_value = mock_args, {}

    magnus_cli = cli.MagnusCLI()
    with pytest.raises(Exception):
        magnus_cli.execute()


def test_execute_raises_exception_if_run_id_is_not_correct(monkeypatch, mocker):
    mock_argparse = mocker.MagicMock()
    monkeypatch.setattr(cli, 'argparse', mock_argparse)

    mock_resolve_args = mocker.MagicMock()
    monkeypatch.setattr(cli.MagnusCLI, '_resolve_args', mock_resolve_args)

    monkeypatch.setattr(cli.MagnusCLI, '__init__', mocker.MagicMock(return_value=None))

    mock_args = mocker.MagicMock()
    mock_args.use_cached = True
    mock_args.run_id = 'somethingwrong'
    mock_resolve_args.return_value = mock_args, {}

    magnus_cli = cli.MagnusCLI()
    with pytest.raises(Exception):
        magnus_cli.execute()


def test_execute_calls_pipeline_execute_with_right_variables(monkeypatch, mocker):
    mock_argparse = mocker.MagicMock()
    monkeypatch.setattr(cli, 'argparse', mock_argparse)

    mock_resolve_args = mocker.MagicMock()
    monkeypatch.setattr(cli.MagnusCLI, '_resolve_args', mock_resolve_args)

    monkeypatch.setattr(cli.MagnusCLI, '__init__', mocker.MagicMock(return_value=None))

    mock_pipeline_execute = mocker.MagicMock()
    monkeypatch.setattr(cli.pipeline, 'execute', mock_pipeline_execute)

    mock_args = mocker.MagicMock()
    mock_args.use_cached = False
    mock_args.run_id = 'some_run_id'
    mock_args.var_file = 'variables_file'
    mock_args.file = 'pipeline_file'
    mock_args.tag = 'tag'
    mock_args.use_cached_force = False
    mock_args.log_level = 0

    mock_resolve_args.return_value = mock_args, {'a': 1}

    magnus_cli = cli.MagnusCLI()
    magnus_cli.execute()
    mock_pipeline_execute.assert_called_once_with(
        variables_file='variables_file', pipeline_file='pipeline_file',
        tag='tag', run_id='some_run_id',
        use_cached=False, use_cached_force=False, a=1)


def test_execute_calls_pipeline_execute_single_node_with_right_variables(monkeypatch, mocker):
    mock_argparse = mocker.MagicMock()
    monkeypatch.setattr(cli, 'argparse', mock_argparse)

    mock_resolve_args = mocker.MagicMock()
    monkeypatch.setattr(cli.MagnusCLI, '_resolve_args', mock_resolve_args)

    monkeypatch.setattr(cli.MagnusCLI, '__init__', mocker.MagicMock(return_value=None))

    mock_pipeline_execute = mocker.MagicMock()
    monkeypatch.setattr(cli.pipeline, 'execute_single_node', mock_pipeline_execute)

    mock_args = mocker.MagicMock()
    mock_args.step_name = 'step_name'
    mock_args.run_id = 'some_run_id'
    mock_args.var_file = 'variables_file'
    mock_args.file = 'pipeline_file'
    mock_args.tag = 'tag'
    mock_args.map_variable = 'map_variable'

    mock_args.log_level = 0

    mock_resolve_args.return_value = mock_args, {'a': 1}

    magnus_cli = cli.MagnusCLI()
    magnus_cli.execute_single_node()
    mock_pipeline_execute.assert_called_once_with(
        variables_file='variables_file', pipeline_file='pipeline_file',
        step_name='step_name', map_variable='map_variable',
        tag='tag', run_id='some_run_id',
        a=1)


def test_execute_calls_pipeline_execute_single_branch_with_right_variables(monkeypatch, mocker):
    mock_argparse = mocker.MagicMock()
    monkeypatch.setattr(cli, 'argparse', mock_argparse)

    mock_resolve_args = mocker.MagicMock()
    monkeypatch.setattr(cli.MagnusCLI, '_resolve_args', mock_resolve_args)

    monkeypatch.setattr(cli.MagnusCLI, '__init__', mocker.MagicMock(return_value=None))

    mock_pipeline_execute = mocker.MagicMock()
    monkeypatch.setattr(cli.pipeline, 'execute_single_brach', mock_pipeline_execute)

    mock_args = mocker.MagicMock()
    mock_args.branch_name = 'branch_name'
    mock_args.run_id = 'some_run_id'
    mock_args.var_file = 'variables_file'
    mock_args.file = 'pipeline_file'
    mock_args.tag = 'tag'
    mock_args.map_variable = 'map_variable'

    mock_args.log_level = 0

    mock_resolve_args.return_value = mock_args, {'a': 1}

    magnus_cli = cli.MagnusCLI()
    magnus_cli.execute_single_branch()
    mock_pipeline_execute.assert_called_once_with(
        variables_file='variables_file', pipeline_file='pipeline_file',
        branch_name='branch_name', map_variable='map_variable',
        tag='tag', run_id='some_run_id',
        a=1)

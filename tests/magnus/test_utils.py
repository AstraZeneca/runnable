import json
import os
import sys

import pytest

from magnus import defaults  # pylint: disable=import-error
from magnus import exceptions  # pylint: disable=import-error
from magnus import utils  # pylint: disable=import-error


def test_does_file_exist_returns_true_if_path_true(mocker, monkeypatch):
    monkeypatch.setattr(utils.Path, 'is_file', lambda x: True)
    assert utils.does_file_exist('test_path')


def test_does_file_exist_returns_false_if_path_false(mocker, monkeypatch):
    monkeypatch.setattr(utils.Path, 'is_file', lambda x: False)
    assert not utils.does_file_exist('test_path')


def test_does_dir_exist_returns_true_if_path_true(mocker, monkeypatch):
    monkeypatch.setattr(utils.Path, 'is_dir', lambda x: True)
    assert utils.does_dir_exist('test_path')


def test_does_dir_exist_returns_false_if_path_false(mocker, monkeypatch):
    monkeypatch.setattr(utils.Path, 'is_dir', lambda x: False)
    assert not utils.does_dir_exist('test_path')


def test_safe_make_dir_calls_with_correct_arguments(mocker, monkeypatch):
    mock_mkdir = mocker.MagicMock()
    monkeypatch.setattr(utils.Path, 'mkdir', mock_mkdir)

    utils.safe_make_dir('test')
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_generate_run_id_makes_one_if_not_provided():
    run_id = utils.generate_run_id()
    assert run_id


def test_generate_run_id_returns_the_same_run_id_if_provided():
    run_id = utils.generate_run_id('test')

    assert run_id == 'test'


def test_apply_variables_raises_exception_if_variables_is_not_dict():
    with pytest.raises(Exception):
        utils.apply_variables('test', '')


def test_apply_variables_applies_variables():
    apply_to = '${var}_${var1}'

    transformed = utils.apply_variables(
        apply_to, variables={'var': 'hello', 'var1': 'me'})
    assert transformed == 'hello_me'


def test_apply_variables_applies_known_variables():
    apply_to = '${var}_${var1}'

    transformed = utils.apply_variables(apply_to, variables={'var': 'hello'})
    assert transformed == 'hello_${var1}'


def test_get_module_and_func_names_raises_exception_for_incorrect_command():
    command = 'hello'

    with pytest.raises(Exception):
        utils.get_module_and_func_names(command)


def test_get_module_and_func_names_returns_module_and_func_names():
    command = 'module.func'

    m, f = utils.get_module_and_func_names(command)

    assert m == 'module'
    assert f == 'func'


def test_get_module_and_func_names_returns_module_and_func_names_inner():
    command = 'module1.module2.func'

    m, f = utils.get_module_and_func_names(command)

    assert m == 'module1.module2'
    assert f == 'func'


def test_get_dag_hash_sorts_keys(mocker, monkeypatch):
    dag = {'b': 1, 'a': 2}

    mock_sha1 = mocker.MagicMock()
    mock_hashlib = mocker.MagicMock()
    mock_hashlib.sha1 = mock_sha1
    monkeypatch.setattr(utils, 'hashlib', mock_hashlib)

    utils.get_dag_hash(dag)

    mock_sha1.assert_called_once_with('{"a": 2, "b": 1}'.encode('utf-8'))


def test_load_yaml_raises_exception_if_yaml_load_does(mocker, monkeypatch):
    mocker.patch('builtins.open', mocker.mock_open(read_data='data'))

    mock_yaml_load = mocker.MagicMock(side_effect=Exception())
    mock_yaml = mocker.MagicMock()

    mock_yaml.load = mock_yaml_load

    monkeypatch.setattr(
        utils, 'YAML', mocker.MagicMock(return_value=mock_yaml))
    with pytest.raises(Exception):
        utils.load_yaml('test')


def test_load_yaml_returns_from_yaml_load(mocker, monkeypatch):
    mocker.patch('builtins.open', mocker.mock_open(read_data='data'))

    mock_yaml_load = mocker.MagicMock(return_value='test')
    mock_yaml = mocker.MagicMock()

    mock_yaml.load = mock_yaml_load

    monkeypatch.setattr(
        utils, 'YAML', mocker.MagicMock(return_value=mock_yaml))
    assert 'test' == utils.load_yaml('does not matter')


def test_is_a_git_repo_suppresses_exceptions(mocker, monkeypatch):
    mock_subprocess = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)

    assert utils.is_a_git_repo() == False


def test_is_a_git_repo_returns_true_if_command_worked(mocker, monkeypatch):
    mock_subprocess = mocker.MagicMock()

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)

    assert utils.is_a_git_repo()


def test_get_current_code_commit_does_suppresses_exceptions(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)
    monkeypatch.setattr(utils, 'is_a_git_repo', mock_is_a_git_repo)

    with pytest.raises(Exception):
        utils.get_current_code_commit()


def test_get_current_code_returns_none_if_not_a_git_repo(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=False)

    monkeypatch.setattr(utils, 'is_a_git_repo', mock_is_a_git_repo)

    assert utils.get_current_code_commit() is None


def test_get_current_code_commit_returns_calls_value(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(return_value=b'test')
    # mock_subprocess.return_value.return_value = 'test'

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)
    monkeypatch.setattr(utils, 'is_a_git_repo', mock_is_a_git_repo)

    assert utils.get_current_code_commit() == 'test'


def test_is_git_clean_suppresses_exceptions(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)
    monkeypatch.setattr(utils, 'is_a_git_repo', mock_is_a_git_repo)

    assert utils.is_git_clean() == (False, None)


def test_is_git_clean_returns_none_if_not_a_git_repo(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=False)

    monkeypatch.setattr(utils, 'is_a_git_repo', mock_is_a_git_repo)

    assert utils.is_git_clean() == (False, None)


def test_is_git_clean_returns_false_when_call_is_not_empty(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(return_value=b'test')
    # mock_subprocess.return_value.return_value = 'test'

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)
    monkeypatch.setattr(utils, 'is_a_git_repo', mock_is_a_git_repo)

    assert utils.is_git_clean() == (False, 'test')


def test_is_git_clean_returns_true_when_call_is_empty(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(return_value=b'')
    # mock_subprocess.return_value.return_value = 'test'

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)
    monkeypatch.setattr(utils, 'is_a_git_repo', mock_is_a_git_repo)

    assert utils.is_git_clean() == (True, None)


def test_get_git_remote_does_not_suppresses_exceptions(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)
    monkeypatch.setattr(utils, 'is_a_git_repo', mock_is_a_git_repo)
    with pytest.raises(Exception):
        utils.get_git_remote()


def test_get_git_remote_returns_none_if_not_a_git_repo(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=False)

    monkeypatch.setattr(utils, 'is_a_git_repo', mock_is_a_git_repo)

    assert utils.get_git_remote() is None


def test_get_git_remote_returns_calls_value(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(return_value=b'test')

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)
    monkeypatch.setattr(utils, 'is_a_git_repo', mock_is_a_git_repo)

    assert utils.get_git_remote() == 'test'


def test_get_git_code_identity_returns_default_in_case_of_exception(mocker, monkeypatch):
    mock_get_current_code_commit = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils, 'get_current_code_commit',
                        mock_get_current_code_commit)

    class MockCodeIdentity:
        pass

    run_log_store = mocker.MagicMock()
    run_log_store.create_code_identity.return_value = MockCodeIdentity()

    assert isinstance(utils.get_git_code_identity(
        run_log_store), MockCodeIdentity)


def test_get_git_code_identity_returns_entities_from_other_functions(monkeypatch, mocker):
    mock_get_current_code_commit = mocker.MagicMock(return_value='code commit')
    mock_is_git_clean = mocker.MagicMock(
        return_value=(False, 'first file, second file'))
    mock_get_git_remote = mocker.MagicMock(return_value='git remote')

    monkeypatch.setattr(utils, 'get_current_code_commit',
                        mock_get_current_code_commit)
    monkeypatch.setattr(utils, 'is_git_clean', mock_is_git_clean)
    monkeypatch.setattr(utils, 'get_git_remote', mock_get_git_remote)

    mock_code_id = mocker.MagicMock()

    run_log_store = mocker.MagicMock()
    run_log_store.create_code_identity.return_value = mock_code_id

    utils.get_git_code_identity(run_log_store)

    assert mock_code_id.code_identifier == 'code commit'
    assert mock_code_id.code_identifier_dependable is False
    assert mock_code_id.code_identifier_url == 'git remote'


def test_remove_prefix_returns_text_as_found_if_prefix_not_found():
    text = 'hi'

    assert utils.remove_prefix(text, 'b') == text


def test_remove_prefix_returns_text_removes_prefix_if_found():
    text = 'hi'

    assert utils.remove_prefix(text, 'h') == 'i'


def test_remove_prefix_returns_text_removes_prefix_if_found_full():
    text = 'hi'

    assert utils.remove_prefix(text, 'hi') == ''


def test_get_user_set_parameters_does_nothing_if_prefix_does_not_match(monkeypatch):
    monkeypatch.setenv('random', 'value')

    assert utils.get_user_set_parameters() == {}


def test_get_user_set_parameters_returns_the_parameter_if_prefix_match_int(monkeypatch):
    monkeypatch.setenv(defaults.PARAMETER_PREFIX + 'key', '1')

    assert utils.get_user_set_parameters() == {'key': 1}


def test_get_user_set_parameters_returns_the_parameter_if_prefix_match_string(monkeypatch):
    monkeypatch.setenv(defaults.PARAMETER_PREFIX + 'key', '"value"')

    assert utils.get_user_set_parameters() == {'key': 'value'}


def test_get_user_set_parameters_removes_the_parameter_if_prefix_match_remove(monkeypatch):
    monkeypatch.setenv(defaults.PARAMETER_PREFIX + 'key', '1')

    assert defaults.PARAMETER_PREFIX+'key' in os.environ

    utils.get_user_set_parameters(remove=True)

    assert defaults.PARAMETER_PREFIX+'key' not in os.environ


def test_get_tracked_data_does_nothing_if_prefix_does_not_match(monkeypatch):
    monkeypatch.setenv('random', 'value')

    assert utils.get_tracked_data() == {}


def test_get_tracked_data_returns_the_data_if_prefix_match_int(monkeypatch):
    monkeypatch.setenv(defaults.TRACK_PREFIX + 'key', '1')

    assert utils.get_tracked_data() == {'key': 1}


def test_get_tracked_data_returns_the_data_if_prefix_match_string(monkeypatch):
    monkeypatch.setenv(defaults.TRACK_PREFIX + 'key', '"value"')

    assert utils.get_tracked_data() == {'key': 'value'}


def test_get_tracked_data_removes_the_data_if_prefix_match_remove(monkeypatch):
    monkeypatch.setenv(defaults.TRACK_PREFIX + 'key', '1')

    assert defaults.TRACK_PREFIX + 'key' in os.environ

    utils.get_tracked_data()

    assert defaults.TRACK_PREFIX + 'key' not in os.environ


def test_get_local_docker_image_id_gets_image_from_docker_client(mocker, monkeypatch):
    mock_client = mocker.MagicMock()
    mock_docker = mocker.MagicMock()

    mock_docker.from_env.return_value = mock_client

    monkeypatch.setattr(utils, 'docker', mock_docker)

    class MockImage:
        attrs = {'Id': 'I am a docker image ID'}

    mock_client.images.get = mocker.MagicMock(return_value=MockImage())
    assert utils.get_local_docker_image_id('test') == 'I am a docker image ID'


def test_get_local_docker_image_id_returns_none_in_exception(mocker, monkeypatch):
    mock_client = mocker.MagicMock()
    mock_docker = mocker.MagicMock()

    mock_docker.from_env.return_value = mock_client

    monkeypatch.setattr(utils, 'docker', mock_docker)

    mock_client.images.get = mocker.MagicMock(
        side_effect=Exception('No Image exists'))

    assert utils.get_local_docker_image_id('test') is ''


def test_filter_arguments_for_func_works_only_named_arguments_in_func_spec():
    def my_func(a, b):
        pass

    parameters = {'a': 1, 'b': 1}

    assert parameters == utils.filter_arguments_for_func(
        my_func, parameters, map_variable=None)


def test_filter_arguments_for_func_returns_empty_if_no_parameters():
    def my_func(a=2, b=1):
        pass

    parameters = {}

    assert parameters == utils.filter_arguments_for_func(
        my_func, parameters, map_variable=None)


def test_filter_arguments_for_func_identifies_args_from_map_variables():
    def my_func(y_i, a=2, b=1):
        pass

    parameters = {'a': 1, 'b': 1}

    assert {'a': 1, 'b': 1, 'y_i': 'y'} == utils.filter_arguments_for_func(
        my_func, parameters, map_variable={'y_i': 'y'})


def test_get_node_execution_command_returns_magnus_execute():
    class MockExecutor:
        run_id = 'test_run_id'
        pipeline_file = 'test_pipeline_file'
        variables_file = None
        configuration_file = None

    class MockNode:
        internal_name = 'test_node_id'

        def command_friendly_name(self):
            return 'test_node_id'

    assert utils.get_node_execution_command(MockExecutor(), MockNode()) == \
        'magnus execute_single_node test_run_id test_node_id --file test_pipeline_file'


def test_get_node_execution_command_overwrites_run_id_if_asked():
    class MockExecutor:
        run_id = 'test_run_id'
        pipeline_file = 'test_pipeline_file'
        variables_file = None
        configuration_file = None

    class MockNode:
        internal_name = 'test_node_id'

        def command_friendly_name(self):
            return 'test_node_id'

    assert utils.get_node_execution_command(MockExecutor(), MockNode(), over_write_run_id='override') == \
        'magnus execute_single_node override test_node_id --file test_pipeline_file'


def test_get_node_execution_command_returns_magnus_execute_appends_variables_file():
    class MockExecutor:
        run_id = 'test_run_id'
        pipeline_file = 'test_pipeline_file'
        variables_file = 'test_variables_file'
        configuration_file = None

    class MockNode:
        internal_name = 'test_node_id'

        def command_friendly_name(self):
            return 'test_node_id'

    assert utils.get_node_execution_command(MockExecutor(), MockNode()) == \
        'magnus execute_single_node test_run_id test_node_id --file test_pipeline_file --var-file test_variables_file'


def test_get_node_execution_command_returns_magnus_execute_appends_map_variable():
    class MockExecutor:
        run_id = 'test_run_id'
        pipeline_file = 'test_pipeline_file'
        variables_file = None
        configuration_file = None

    class MockNode:
        internal_name = 'test_node_id'

        def command_friendly_name(self):
            return 'test_node_id'

    map_variable = {'test_map': 'map_value'}
    json_dump = json.dumps(map_variable)
    assert utils.get_node_execution_command(MockExecutor(), MockNode(), map_variable=map_variable) == \
        f"magnus execute_single_node test_run_id test_node_id --file test_pipeline_file --map-variable '{json_dump}'"


def test_get_service_base_class_throws_exception_for_unknown_service():
    with pytest.raises(Exception):
        utils.get_service_base_class('Does not exist')


def test_get_subclasses_works_with_one_level():
    class Parent:
        pass

    class Child(Parent):
        pass

    assert len(list(utils.get_subclasses(Parent))) == 1


def test_get_subclasses_works_with_one_level_multiple():
    class Parent:
        pass

    class Child(Parent):
        pass

    class Child1(Parent):
        pass

    assert len(list(utils.get_subclasses(Parent))) == 2


def test_get_subclasses_works_with_two_level_multiple():
    class Parent:
        pass

    class Child(Parent):
        pass

    class ChildOfChild(Child):
        pass

    assert len(list(utils.get_subclasses(Parent))) == 2

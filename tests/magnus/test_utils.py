import pytest
import os

from magnus import utils  # pylint: disable=import-error
from magnus import exceptions  # pylint: disable=import-error
from magnus import defaults  # pylint: disable=import-error


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


def test_validate_run_id_errors_if_incorrect_format():
    run_id = '1_2_3'
    with pytest.raises(Exception):
        utils.validate_run_id(run_id)


def test_validate_run_id_correct_format():
    utils.validate_run_id('1_2')
    utils.validate_run_id('1')
    utils.validate_run_id('')


def test_generate_run_id_makes_one_if_not_provided(mocker, monkeypatch):
    mock_validate = mocker.MagicMock()

    monkeypatch.setattr(utils, 'validate_run_id', mock_validate)

    run_id = utils.generate_run_id()
    assert len(run_id.split('_')) == 2


def test_generate_run_id_raises_exception_if_validate_does(mocker, monkeypatch):
    mock_validate = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils, 'validate_run_id', mock_validate)

    with pytest.raises(Exception):
        _ = utils.generate_run_id()


def test_generate_run_id_only_retains_the_first_part(mocker, monkeypatch):
    mock_validate = mocker.MagicMock()

    monkeypatch.setattr(utils, 'validate_run_id', mock_validate)

    run_id = utils.generate_run_id('retain_ignore')
    retain, _ = run_id.split('_')
    assert retain == 'retain'


def test_generate_run_id_only_randoms_the_second_part_len(mocker, monkeypatch):
    mock_validate = mocker.MagicMock()

    monkeypatch.setattr(utils, 'validate_run_id', mock_validate)

    run_id = utils.generate_run_id('retain_ignore')
    _, random_gen = run_id.split('_')
    assert len(random_gen) == defaults.RANDOM_RUN_ID_LEN


def test_apply_variables_raises_exception_if_variables_is_not_dict():
    with pytest.raises(Exception):
        utils.apply_variables('test', '')


def test_apply_variables_applies_variables():
    apply_to = '${var}_${var1}'

    transformed = utils.apply_variables(apply_to, variables={'var': 'hello', 'var1': 'me'})
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

    monkeypatch.setattr(utils, 'YAML', mocker.MagicMock(return_value=mock_yaml))
    with pytest.raises(Exception):
        utils.load_yaml('test')


def test_is_a_git_repo_suppresses_exceptions(mocker, monkeypatch):
    mock_subprocess = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)

    assert utils.is_a_git_repo() == False


def test_is_a_git_repo_returns_true_if_command_worked(mocker, monkeypatch):
    mock_subprocess = mocker.MagicMock()

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)

    assert utils.is_a_git_repo()


def test_get_current_code_commit_suppresses_exceptions(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)
    monkeypatch.setattr(utils, 'is_a_git_repo', mock_is_a_git_repo)

    assert utils.get_current_code_commit() is None


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


def test_get_git_remote_suppresses_exceptions(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils.subprocess, 'check_output', mock_subprocess)
    monkeypatch.setattr(utils, 'is_a_git_repo', mock_is_a_git_repo)

    assert utils.get_git_remote() is None


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

    mock_client.images.get = mocker.MagicMock(side_effect=Exception('No Image exists'))

    assert utils.get_local_docker_image_id('test') is None


def test_filter_arguments_for_func_works_only_named_arguments_in_func_spec():
    def my_func(a, b):
        pass

    parameters = {'a': 1, 'b': 1}

    assert parameters == utils.filter_arguments_for_func(my_func, parameters)


def test_filter_arguments_for_func_works_with_keywords_arguments_in_func_spec():
    def my_func(a=2, b=1):
        pass

    parameters = {'a': 1, 'b': 1}

    assert parameters == utils.filter_arguments_for_func(my_func, parameters)


def test_filter_arguments_for_func_returns_empty_if_no_parameters():
    def my_func(a=2, b=1):
        pass

    parameters = {}

    assert parameters == utils.filter_arguments_for_func(my_func, parameters)


def test_get_node_execution_command_returns_magnus_execute():
    class MockExecutor:
        run_id = 'test_run_id'
        pipeline_file = 'test_pipeline_file'
        variables_file = None

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

    class MockNode:
        internal_name = 'test_node_id'

        def command_friendly_name(self):
            return 'test_node_id'

    assert utils.get_node_execution_command(MockExecutor(), MockNode(), map_variable='test_map') == \
        'magnus execute_single_node test_run_id test_node_id --file test_pipeline_file --map-variable test_map'

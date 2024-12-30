import json
import os
import sys

import pytest

from runnable import (
    defaults,  # pylint: disable=import-error
    utils,  # pylint: disable=import-error
)


def test_does_file_exist_returns_true_if_path_true(mocker, monkeypatch):
    monkeypatch.setattr(utils.Path, "is_file", lambda x: True)
    assert utils.does_file_exist("test_path")


def test_does_file_exist_returns_false_if_path_false(mocker, monkeypatch):
    monkeypatch.setattr(utils.Path, "is_file", lambda x: False)
    assert not utils.does_file_exist("test_path")


def test_does_dir_exist_returns_true_if_path_true(mocker, monkeypatch):
    monkeypatch.setattr(utils.Path, "is_dir", lambda x: True)
    assert utils.does_dir_exist("test_path")


def test_does_dir_exist_returns_false_if_path_false(mocker, monkeypatch):
    monkeypatch.setattr(utils.Path, "is_dir", lambda x: False)
    assert not utils.does_dir_exist("test_path")


def test_safe_make_dir_calls_with_correct_arguments(mocker, monkeypatch):
    mock_mkdir = mocker.MagicMock()
    monkeypatch.setattr(utils.Path, "mkdir", mock_mkdir)

    utils.safe_make_dir("test")
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_generate_run_id_makes_one_if_not_provided():
    run_id = utils.generate_run_id()
    assert run_id


def test_generate_run_id_returns_the_same_run_id_if_provided():
    run_id = utils.generate_run_id("test")

    assert run_id == "test"


def test_apply_variables_raises_exception_if_variables_is_not_dict():
    with pytest.raises(Exception):
        utils.apply_variables("test", "")


def test_apply_variables_applies_variables():
    apply_to = "${var}_${var1}"

    transformed = utils.apply_variables(
        apply_to, variables={"var": "hello", "var1": "me"}
    )
    assert transformed == "hello_me"


def test_apply_variables_applies_known_variables():
    apply_to = "${var}_${var1}"

    with pytest.raises(KeyError):
        transformed = utils.apply_variables(apply_to, variables={"var": "hello"})


def test_get_module_and_func_names_raises_exception_for_incorrect_command():
    command = "hello"

    with pytest.raises(Exception):
        utils.get_module_and_attr_names(command)


def test_get_module_and_func_names_returns_module_and_func_names():
    command = "module.func"

    m, f = utils.get_module_and_attr_names(command)

    assert m == "module"
    assert f == "func"


def test_get_module_and_func_names_returns_module_and_func_names_inner():
    command = "module1.module2.func"

    m, f = utils.get_module_and_attr_names(command)

    assert m == "module1.module2"
    assert f == "func"


def test_get_dag_hash_sorts_keys(mocker, monkeypatch):
    dag = {"b": 1, "a": 2}

    mock_sha1 = mocker.MagicMock()
    mock_hashlib = mocker.MagicMock()
    mock_hashlib.sha1 = mock_sha1
    monkeypatch.setattr(utils, "hashlib", mock_hashlib)

    utils.get_dag_hash(dag)

    mock_sha1.assert_called_once_with('{"a": 2, "b": 1}'.encode("utf-8"))


def test_load_yaml_raises_exception_if_yaml_load_does(mocker, monkeypatch):
    mocker.patch("builtins.open", mocker.mock_open(read_data="data"))

    mock_yaml_load = mocker.MagicMock(side_effect=Exception())
    mock_yaml = mocker.MagicMock()

    mock_yaml.load = mock_yaml_load

    monkeypatch.setattr(utils, "YAML", mocker.MagicMock(return_value=mock_yaml))
    with pytest.raises(Exception):
        utils.load_yaml("test")


def test_load_yaml_returns_from_yaml_load(mocker, monkeypatch):
    mocker.patch("builtins.open", mocker.mock_open(read_data="data"))

    mock_yaml_load = mocker.MagicMock(return_value="test")
    mock_yaml = mocker.MagicMock()

    mock_yaml.load = mock_yaml_load

    monkeypatch.setattr(utils, "YAML", mocker.MagicMock(return_value=mock_yaml))
    assert "test" == utils.load_yaml("does not matter")


def test_is_a_git_repo_suppresses_exceptions(mocker, monkeypatch):
    mock_subprocess = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils.subprocess, "check_output", mock_subprocess)

    assert utils.is_a_git_repo() is False


def test_is_a_git_repo_returns_true_if_command_worked(mocker, monkeypatch):
    mock_subprocess = mocker.MagicMock()

    monkeypatch.setattr(utils.subprocess, "check_output", mock_subprocess)

    assert utils.is_a_git_repo()


def test_get_current_code_commit_does_suppresses_exceptions(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils.subprocess, "check_output", mock_subprocess)
    monkeypatch.setattr(utils, "is_a_git_repo", mock_is_a_git_repo)

    with pytest.raises(Exception):
        utils.get_current_code_commit()


def test_get_current_code_returns_none_if_not_a_git_repo(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=False)

    monkeypatch.setattr(utils, "is_a_git_repo", mock_is_a_git_repo)

    assert utils.get_current_code_commit() is None


def test_get_current_code_commit_returns_calls_value(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(return_value=b"test")
    # mock_subprocess.return_value.return_value = 'test'

    monkeypatch.setattr(utils.subprocess, "check_output", mock_subprocess)
    monkeypatch.setattr(utils, "is_a_git_repo", mock_is_a_git_repo)

    assert utils.get_current_code_commit() == "test"


def test_is_git_clean_suppresses_exceptions(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils.subprocess, "check_output", mock_subprocess)
    monkeypatch.setattr(utils, "is_a_git_repo", mock_is_a_git_repo)

    assert utils.is_git_clean() == (False, None)


def test_is_git_clean_returns_none_if_not_a_git_repo(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=False)

    monkeypatch.setattr(utils, "is_a_git_repo", mock_is_a_git_repo)

    assert utils.is_git_clean() == (False, None)


def test_is_git_clean_returns_false_when_call_is_not_empty(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(return_value=b"test")
    # mock_subprocess.return_value.return_value = 'test'

    monkeypatch.setattr(utils.subprocess, "check_output", mock_subprocess)
    monkeypatch.setattr(utils, "is_a_git_repo", mock_is_a_git_repo)

    assert utils.is_git_clean() == (False, "test")


def test_is_git_clean_returns_true_when_call_is_empty(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(return_value=b"")
    # mock_subprocess.return_value.return_value = 'test'

    monkeypatch.setattr(utils.subprocess, "check_output", mock_subprocess)
    monkeypatch.setattr(utils, "is_a_git_repo", mock_is_a_git_repo)

    assert utils.is_git_clean() == (True, None)


def test_get_git_remote_does_not_suppresses_exceptions(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils.subprocess, "check_output", mock_subprocess)
    monkeypatch.setattr(utils, "is_a_git_repo", mock_is_a_git_repo)
    with pytest.raises(Exception):
        utils.get_git_remote()


def test_get_git_remote_returns_none_if_not_a_git_repo(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=False)

    monkeypatch.setattr(utils, "is_a_git_repo", mock_is_a_git_repo)

    assert utils.get_git_remote() is None


def test_get_git_remote_returns_calls_value(mocker, monkeypatch):
    mock_is_a_git_repo = mocker.MagicMock(return_value=True)

    mock_subprocess = mocker.MagicMock(return_value=b"test")

    monkeypatch.setattr(utils.subprocess, "check_output", mock_subprocess)
    monkeypatch.setattr(utils, "is_a_git_repo", mock_is_a_git_repo)

    assert utils.get_git_remote() == "test"


def test_get_git_code_identity_returns_default_in_case_of_exception(
    mocker, monkeypatch
):
    mock_get_current_code_commit = mocker.MagicMock(side_effect=Exception())

    monkeypatch.setattr(utils, "get_current_code_commit", mock_get_current_code_commit)

    mock_code_identity = mocker.MagicMock()
    mock_run_context = mocker.MagicMock()
    mock_run_context.run_log_store.create_code_identity.return_value = (
        mock_code_identity
    )

    monkeypatch.setattr(utils.context, "run_context", mock_run_context)

    assert utils.get_git_code_identity() == mock_code_identity


def test_get_git_code_identity_returns_entities_from_other_functions(
    monkeypatch, mocker
):
    mock_get_current_code_commit = mocker.MagicMock(return_value="code commit")
    mock_is_git_clean = mocker.MagicMock(
        return_value=(False, "first file, second file")
    )
    mock_get_git_remote = mocker.MagicMock(return_value="git remote")

    monkeypatch.setattr(utils, "get_current_code_commit", mock_get_current_code_commit)
    monkeypatch.setattr(utils, "is_git_clean", mock_is_git_clean)
    monkeypatch.setattr(utils, "get_git_remote", mock_get_git_remote)

    mock_code_identity = mocker.MagicMock()
    mock_run_context = mocker.MagicMock()
    mock_run_context.run_log_store.create_code_identity.return_value = (
        mock_code_identity
    )

    monkeypatch.setattr(utils.context, "run_context", mock_run_context)

    utils.get_git_code_identity()

    assert mock_code_identity.code_identifier == "code commit"
    assert mock_code_identity.code_identifier_dependable is False
    assert mock_code_identity.code_identifier_url == "git remote"


def test_remove_prefix_returns_text_as_found_if_prefix_not_found():
    text = "hi"

    assert utils.remove_prefix(text, "b") == text


def test_remove_prefix_returns_text_removes_prefix_if_found():
    text = "hi"

    assert utils.remove_prefix(text, "h") == "i"


def test_remove_prefix_returns_text_removes_prefix_if_found_full():
    text = "hi"

    assert utils.remove_prefix(text, "hi") == ""


def test_get_tracked_data_does_nothing_if_prefix_does_not_match(monkeypatch):
    monkeypatch.setenv("random", "value")

    assert utils.get_tracked_data() == {}


def test_get_tracked_data_returns_the_data_if_prefix_match_int(monkeypatch):
    monkeypatch.setenv(defaults.TRACK_PREFIX + "key", "1")

    assert utils.get_tracked_data() == {"key": 1}


def test_get_tracked_data_returns_the_data_if_prefix_match_string(monkeypatch):
    monkeypatch.setenv(defaults.TRACK_PREFIX + "key", '"value"')

    assert utils.get_tracked_data() == {"key": "value"}


def test_get_tracked_data_removes_the_data_if_prefix_match_remove(monkeypatch):
    monkeypatch.setenv(defaults.TRACK_PREFIX + "key", "1")

    assert defaults.TRACK_PREFIX + "key" in os.environ

    utils.get_tracked_data()

    assert defaults.TRACK_PREFIX + "key" not in os.environ


def test_get_local_docker_image_id_gets_image_from_docker_client(mocker, monkeypatch):
    mock_client = mocker.MagicMock()
    mock_docker = mocker.MagicMock()

    mock_docker.from_env.return_value = mock_client

    with pytest.MonkeyPatch().context() as ctx:
        sys.modules["docker"] = mock_docker

        class MockImage:
            attrs = {"Id": "I am a docker image ID"}

        mock_client.images.get = mocker.MagicMock(return_value=MockImage())
        assert utils.get_local_docker_image_id("test") == "I am a docker image ID"


def test_get_local_docker_image_id_returns_none_in_exception(mocker, monkeypatch):
    mock_client = mocker.MagicMock()
    mock_docker = mocker.MagicMock()

    mock_docker.from_env.return_value = mock_client

    with pytest.MonkeyPatch().context() as ctx:
        sys.modules["docker"] = mock_docker

        mock_client.images.get = mocker.MagicMock(
            side_effect=Exception("No Image exists")
        )

        assert utils.get_local_docker_image_id("test") == ""


def test_get_node_execution_command_returns_runnable_execute(mocker, monkeypatch):
    import logging

    mock_context = mocker.MagicMock()
    mock_context.run_context.run_id = "test_run_id"
    mock_context.run_context.pipeline_file = "test_pipeline_file"
    mock_context.run_context.configuration_file = "test_configuration_file"
    mock_context.run_context.parameters_file = "test_parameters_file"
    mock_context.run_context.tag = "test_tag"

    monkeypatch.setattr(utils, "context", mock_context)

    logger = logging.getLogger(name="runnable")
    old_level = logger.level
    logger.setLevel(defaults.LOG_LEVEL)

    class MockNode:
        internal_name = "test_node_id"

        def _command_friendly_name(self):
            return "test_node_id"

    test_map_variable = {"a": "b"}
    try:
        assert utils.get_node_execution_command(
            MockNode(), map_variable=test_map_variable
        ) == (
            "runnable execute-single-node test_run_id test_pipeline_file test_node_id "
            f"--log-level WARNING --mode python --map-variable '{json.dumps(test_map_variable)}' --config test_configuration_file "
            "--parameters-file test_parameters_file --tag test_tag"
        )
    finally:
        logger.setLevel(old_level)


def test_get_node_execution_command_overwrites_run_id_if_asked(mocker, monkeypatch):
    import logging

    mock_context = mocker.MagicMock()
    mock_context.run_context.run_id = "test_run_id"
    mock_context.run_context.pipeline_file = "test_pipeline_file"
    mock_context.run_context.configuration_file = "test_configuration_file"
    mock_context.run_context.parameters_file = "test_parameters_file"
    mock_context.run_context.tag = "test_tag"

    monkeypatch.setattr(utils, "context", mock_context)

    class MockNode:
        internal_name = "test_node_id"

        def _command_friendly_name(self):
            return "test_node_id"

    logger = logging.getLogger(name="runnable")
    old_level = logger.level
    logger.setLevel(defaults.LOG_LEVEL)

    test_map_variable = {"a": "b"}
    try:
        assert utils.get_node_execution_command(
            MockNode(), map_variable=test_map_variable, over_write_run_id="this"
        ) == (
            "runnable execute-single-node this test_pipeline_file test_node_id "
            f"--log-level WARNING --mode python --map-variable '{json.dumps(test_map_variable)}' --config test_configuration_file "
            "--parameters-file test_parameters_file --tag test_tag"
        )
    finally:
        logger.setLevel(old_level)


def test_get_service_base_class_throws_exception_for_unknown_service():
    with pytest.raises(Exception):
        utils.get_service_base_class("Does not exist")

# ruff: noqa

import tempfile
from pathlib import Path
from rich import print

import pytest
import ruamel.yaml

from magnus import defaults, entrypoints, utils

yaml = ruamel.yaml.YAML()

PIPELINES_DEFINITION = Path("examples/")


def get_config():
    config = {
        "executor": {
            "type": "local",
        },
        "run_log_store": {"type": "file-system", "config": {"log_folder": ""}},
    }
    return config


def get_container_config():
    config = {
        "executor": {"type": "local-container", "config": {"docker_image": "does-not-matter"}},
        "run_log_store": {"type": "file-system", "config": {"log_folder": ""}},
    }
    return config


def get_chunked_config():
    config = {
        "executor": {
            "type": "local",
        },
        "run_log_store": {"type": "chunked-fs", "config": {"log_folder": ""}},
    }
    return config


def get_configs():
    return [get_config(), get_chunked_config()]


def write_config(work_dir: Path, config: dict):
    config["run_log_store"]["config"]["log_folder"] = str(work_dir)
    with open(work_dir / "config.yaml", "wb") as f:
        yaml.dump(config, f)


def get_run_log(work_dir, run_id):
    config_file = work_dir / "config.yaml"

    if utils.does_file_exist(config_file):
        mode_executor = entrypoints.prepare_configurations(configuration_file=str(config_file), run_id=run_id)
        return mode_executor.run_log_store.get_run_log_by_id(run_id=run_id, full=True).model_dump()
    raise Exception


@pytest.mark.no_cover
def test_success():
    configs = get_configs()

    for config in configs:
        with tempfile.TemporaryDirectory() as context_dir:
            context_dir_path = Path(context_dir)

            write_config(context_dir_path, config)

            run_id = "testing_success"

            entrypoints.execute(
                configuration_file=str(context_dir_path / "config.yaml"),
                pipeline_file=str(PIPELINES_DEFINITION / "mocking.yaml"),
                run_id=run_id,
            )

            try:
                run_log = get_run_log(context_dir_path, run_id)
                assert run_log["status"] == defaults.SUCCESS
                assert list(run_log["steps"].keys()) == ["step 1", "step 2", "step 3", "success"]
            except:
                assert False


@pytest.mark.no_cover
def test_failure():
    configs = get_configs()

    for config in configs:
        with tempfile.TemporaryDirectory() as context_dir:
            context_dir_path = Path(context_dir)

            write_config(context_dir_path, config)

            run_id = "testing_failure"

            try:
                entrypoints.execute(
                    configuration_file=str(context_dir_path / "config.yaml"),
                    pipeline_file=str(PIPELINES_DEFINITION / "default-fail.yaml"),
                    run_id=run_id,
                )
            except Exception as ex:
                print(ex)

            try:
                run_log = get_run_log(context_dir_path, run_id)
                assert run_log["status"] == defaults.FAIL
                assert list(run_log["steps"].keys()) == ["step 1", "step 2", "fail"]
            except:
                assert False


@pytest.mark.no_cover
def test_on_failure():
    configs = get_configs()
    for config in configs:
        with tempfile.TemporaryDirectory() as context_dir:
            context_dir_path = Path(context_dir)

            write_config(context_dir_path, config)

            run_id = "testing_on_failure"

            try:
                entrypoints.execute(
                    configuration_file=str(context_dir_path / "config.yaml"),
                    pipeline_file=str(PIPELINES_DEFINITION / "on-failure.yaml"),
                    run_id=run_id,
                )
            except:
                pass

            try:
                run_log = get_run_log(context_dir_path, run_id)
                assert run_log["status"] == defaults.SUCCESS
                assert list(run_log["steps"].keys()) == ["step 1", "step 3", "success"]
            except:
                assert False


@pytest.mark.no_cover
def test_parallel():
    configs = get_configs()
    for config in configs:
        with tempfile.TemporaryDirectory() as context_dir:
            context_dir_path = Path(context_dir)

            write_config(context_dir_path, config)
            run_id = "testing_parallel"

            entrypoints.execute(
                configuration_file=str(context_dir_path / "config.yaml"),
                pipeline_file=str(PIPELINES_DEFINITION / "parallel.yaml"),
                run_id=run_id,
            )

            try:
                run_log = get_run_log(context_dir_path, run_id)
                assert run_log["status"] == defaults.SUCCESS
                assert list(run_log["steps"].keys()) == ["step 1", "step 2", "step 3", "success"]
                assert list(run_log["steps"]["step 2"]["branches"]["step 2.branch_a"]["steps"].keys()) == [
                    "step 2.branch_a.step 1",
                    "step 2.branch_a.step 2",
                    "step 2.branch_a.success",
                ]
                assert list(run_log["steps"]["step 2"]["branches"]["step 2.branch_b"]["steps"].keys()) == [
                    "step 2.branch_b.step 1",
                    "step 2.branch_b.step 2",
                    "step 2.branch_b.success",
                ]
            except:
                assert False


# @pytest.mark.no_cover
# def test_parallel_fail(parallel_fail_graph):
#     configs = get_configs()
#     for config in configs:
#         with tempfile.TemporaryDirectory() as context_dir:
#             context_dir_path = Path(context_dir)
#             dag = {"dag": parallel_fail_graph().dict()}

#             write_dag_and_config(context_dir_path, dag, config)
#             run_id = "testing_parallel"

#             try:
#                 entrypoints.execute(
#                     configuration_file=str(context_dir_path / "config.yaml"),
#                     pipeline_file=str(context_dir_path / "dag.yaml"),
#                     run_id=run_id,
#                 )
#             except:
#                 pass

#             try:
#                 run_log = get_run_log(context_dir_path, run_id)
#                 assert run_log["status"] == defaults.FAIL
#                 assert list(run_log["steps"].keys()) == ["first", "second", "fail"]
#                 assert list(run_log["steps"]["second"]["branches"]["second.a"]["steps"].keys()) == [
#                     "second.a.first",
#                     "second.a.fail",
#                 ]
#                 assert list(run_log["steps"]["second"]["branches"]["second.b"]["steps"].keys()) == [
#                     "second.b.first",
#                     "second.b.fail",
#                 ]
#             except:
#                 assert False

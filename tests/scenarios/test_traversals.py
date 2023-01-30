import json
import tempfile
from pathlib import Path

import pytest
import ruamel.yaml

from magnus import defaults, pipeline, utils

yaml = ruamel.yaml.YAML()


def get_config():
    config = {
        'mode': {
            'type': 'local',
        },
        'run_log_store': {
            'type': 'file-system',
            'config': {
                'log_folder': ''
            }
        }
    }
    return config


def write_dag_and_config(work_dir: str, dag: dict, config: dict):
    with open(work_dir / 'dag.yaml', 'wb') as f:
        yaml.dump(dag, f)

    config['run_log_store']['config']['log_folder'] = str(work_dir)
    with open(work_dir / 'config.yaml', 'wb') as f:
        yaml.dump(config, f)


def get_run_log(work_dir, run_id):
    run_log_path = work_dir / f"{run_id}.json"

    if utils.does_file_exist(run_log_path):
        return json.load(open(run_log_path))
    raise Exception


@pytest.mark.no_cover
def test_success(success_graph):
    config = get_config()

    with tempfile.TemporaryDirectory() as context_dir:
        context_dir_path = Path(context_dir)
        dag = {'dag': success_graph()._to_dict()}

        write_dag_and_config(context_dir_path, dag, config)

        run_id = 'testing_success'

        pipeline.execute(configuration_file=str(context_dir_path / 'config.yaml'),
                         pipeline_file=str(context_dir_path / 'dag.yaml'), run_id=run_id)

        try:
            run_log = get_run_log(context_dir_path, run_id)
            assert run_log['status'] == defaults.SUCCESS
            assert list(run_log['steps'].keys()) == ['first', 'second', 'success']
        except:
            assert False


@pytest.mark.no_cover
def test_failure(fail_graph):
    config = get_config()

    with tempfile.TemporaryDirectory() as context_dir:
        context_dir_path = Path(context_dir)
        dag = {'dag': fail_graph()._to_dict()}

        write_dag_and_config(context_dir_path, dag, config)

        run_id = 'testing_failure'

        try:
            pipeline.execute(configuration_file=str(context_dir_path / 'config.yaml'),
                             pipeline_file=str(context_dir_path / 'dag.yaml'), run_id=run_id)
        except:
            pass

        try:
            run_log = get_run_log(context_dir_path, run_id)
            assert run_log['status'] == defaults.FAIL
            assert list(run_log['steps'].keys()) == ['first', 'fail']
        except:
            assert False


@pytest.mark.no_cover
def test_on_failure(on_fail_graph):
    config = get_config()
    with tempfile.TemporaryDirectory() as context_dir:
        context_dir_path = Path(context_dir)
        dag = {'dag': on_fail_graph()._to_dict()}

        write_dag_and_config(context_dir_path, dag, config)

        run_id = 'testing_failure'

        try:
            pipeline.execute(configuration_file=str(context_dir_path / 'config.yaml'),
                             pipeline_file=str(context_dir_path / 'dag.yaml'), run_id=run_id)
        except:
            pass

        try:
            run_log = get_run_log(context_dir_path, run_id)
            assert run_log['status'] == defaults.SUCCESS
            assert list(run_log['steps'].keys()) == ['first', 'third', 'success']
        except:
            assert False


@pytest.mark.no_cover
def test_parallel(parallel_success_graph):
    config = get_config()
    with tempfile.TemporaryDirectory() as context_dir:
        context_dir_path = Path(context_dir)
        dag = {'dag': parallel_success_graph()._to_dict()}

        write_dag_and_config(context_dir_path, dag, config)
        run_id = 'testing_parallel'

        pipeline.execute(configuration_file=str(context_dir_path / 'config.yaml'),
                         pipeline_file=str(context_dir_path / 'dag.yaml'), run_id=run_id)

        try:
            run_log = get_run_log(context_dir_path, run_id)
            assert run_log['status'] == defaults.SUCCESS
            assert list(run_log['steps'].keys()) == ['first', 'second', 'success']
            assert list(run_log['steps']['second']['branches']['second.a']
                        ['steps'].keys()) == ['second.a.first', 'second.a.second', 'second.a.success']
            assert list(run_log['steps']['second']['branches']['second.b']
                        ['steps'].keys()) == ['second.b.first', 'second.b.second', 'second.b.success']
        except:
            assert False


@pytest.mark.no_cover
def test_parallel_fail(parallel_fail_graph):
    config = get_config()
    with tempfile.TemporaryDirectory() as context_dir:
        context_dir_path = Path(context_dir)
        dag = {'dag': parallel_fail_graph()._to_dict()}

        write_dag_and_config(context_dir_path, dag, config)
        run_id = 'testing_parallel'

        try:
            pipeline.execute(configuration_file=str(context_dir_path / 'config.yaml'),
                             pipeline_file=str(context_dir_path / 'dag.yaml'), run_id=run_id)
        except:
            pass

        try:
            run_log = get_run_log(context_dir_path, run_id)
            assert run_log['status'] == defaults.FAIL
            assert list(run_log['steps'].keys()) == ['first', 'second', 'fail']
            assert list(run_log['steps']['second']['branches']['second.a']
                        ['steps'].keys()) == ['second.a.first', 'second.a.fail']
            assert list(run_log['steps']['second']['branches']['second.b']
                        ['steps'].keys()) == ['second.b.first', 'second.b.fail']
        except:
            assert False

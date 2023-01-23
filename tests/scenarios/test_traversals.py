import json
import tempfile
from pathlib import Path

import ruamel.yaml

from magnus import defaults, pipeline, utils

yaml = ruamel.yaml.YAML()


def get_dags_from_stubs():
    path_to_stubs = 'tests/scenarios/stubs.yaml'
    return utils.load_yaml(path_to_stubs)['dags']


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
    print(dag)
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


def test_success():
    dags = get_dags_from_stubs()
    config = get_config()

    with tempfile.TemporaryDirectory() as context_dir:
        context_dir_path = Path(context_dir)
        dag = dags['success_dag']
        write_dag_and_config(context_dir_path, dag, config)

        run_id = 'testing_success'

        pipeline.execute(configuration_file=str(context_dir_path / 'config.yaml'),
                         pipeline_file=str(context_dir_path / 'dag.yaml'), run_id=run_id)

        try:
            run_log = get_run_log(context_dir_path, run_id)
            assert run_log['status'] == defaults.SUCCESS
        except:
            assert False


def test_failure():
    dags = get_dags_from_stubs()
    config = get_config()

    with tempfile.TemporaryDirectory() as context_dir:
        context_dir_path = Path(context_dir)
        dag = dags['fail_dag']
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
        except:
            assert False


def test_on_failure():
    dags = get_dags_from_stubs()
    config = get_config()

    with tempfile.TemporaryDirectory() as context_dir:
        context_dir_path = Path(context_dir)
        dag = dags['on_fail_dag']
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
        except:
            assert False


def test_parallel():
    dags = get_dags_from_stubs()
    config = get_config()

    with tempfile.TemporaryDirectory() as context_dir:
        context_dir_path = Path(context_dir)
        dag = dags['success_dag']

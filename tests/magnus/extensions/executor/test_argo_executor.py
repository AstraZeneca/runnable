import pytest

from magnus.extensions.executor.argo import implementation


def test_secret_env_var_has_value_from_field():
    secret_env = implementation.SecretEnvVar(
        environment_variable="test_env", secret_name="secret_name", secret_key="secret_key"
    )

    assert secret_env.environment_variable == "test_env"
    assert secret_env.valueFrom == {"secretKeyRef": {"name": "secret_name", "key": "secret_key"}}


def test_secret_env_renders_properly():
    secret_env = implementation.SecretEnvVar(
        environment_variable="test_env", secret_name="secret_name", secret_key="secret_key"
    )
    assert secret_env.model_dump(by_alias=True) == {
        "name": "test_env",
        "valueFrom": {"secretKeyRef": {"name": "secret_name", "key": "secret_key"}},
    }


def test_retry_serialize_makes_limit_str():
    retry = implementation.Retry(limit=10)
    assert retry.model_dump(by_alias=True) == {"limit": "10", "retryPolicy": "Always"}


def test_limit_renders_gpu_when_available():
    limit = implementation.Limit(gpu=1)

    request = implementation.Request()

    assert limit.model_dump(by_alias=True, exclude_none=True) == {**request.model_dump(), "nvidia.com/gpu": "1"}


def test_limit_ignores_gpu_when_none():
    limit = implementation.Limit()

    request = implementation.Request()

    assert limit.model_dump(by_alias=True, exclude_none=True) == {**request.model_dump()}


def test_user_controls_with_defaults():
    user_controls = implementation.UserControls()

    assert user_controls.model_dump(by_alias=True, exclude_none=True) == {
        "activeDeadlineSeconds": 7200,
        "imagePullPolicy": "",
        "limits": {"cpu": "250m", "memory": "1Gi"},
        "nodeSelector": {},
        "requests": {"cpu": "250m", "memory": "1Gi"},
        "retryStrategy": {"limit": "0", "retryPolicy": "Always"},
    }


def test_out_put_parameter_renders_properly():
    output_parameter = implementation.OutputParameter(name="test_name", value="test_value")

    assert output_parameter.model_dump(by_alias=True) == {
        "name": "test_name",
        "value": "test_value",
        "valueFrom": {"path": "/tmp/output.txt"},
    }


def test_dag_template_renders_properly():
    task = implementation.TaskTemplate(name="test_name", template="test_template")

    dag = implementation.DagTemplate(tasks=[task])

    assert dag.model_dump(by_alias=True, exclude_none=True) == {
        "name": "magnus-dag",
        "dag": {"tasks": [task.model_dump(by_alias=True, exclude_none=True)]},
        "failFast": True,
    }


def test_volume_renders_properly():
    volume = implementation.Volume(name="test_name", claim="test_claim", mount_path="mount here")

    assert volume.model_dump(by_alias=True, exclude_none=True) == {
        "name": "test_name",
        "persistentVolumeClaim": {"claimName": "test_claim"},
    }


def test_spec_reshapes_arguments():
    test_env1 = implementation.EnvVar(name="test_env1", value="test_value1")
    test_env2 = implementation.EnvVar(name="test_env2", value="test_value2")

    spec = implementation.Spec(arguments=[test_env1, test_env2])

    assert spec.model_dump(by_alias=True, exclude_none=True)["arguments"] == {
        "parameters": [{"name": "test_env1", "value": "test_value1"}, {"name": "test_env2", "value": "test_value2"}]
    }


def test_spec_populates_container_volumes_and_persistent_volumes():
    volume1 = implementation.UserVolumeMounts(name="test_name1", mount_path="test_mount_path1")
    volume2 = implementation.UserVolumeMounts(name="test_name2", mount_path="test_mount_path2")

    spec = implementation.Spec(persistent_volumes=[volume1, volume2])

    model_dump = spec.model_dump(by_alias=True, exclude_none=True)

    assert model_dump["volumes"] == [
        {"name": "executor-0", "persistentVolumeClaim": {"claimName": "test_name1"}},
        {"name": "executor-1", "persistentVolumeClaim": {"claimName": "test_name2"}},
    ]


def test_user_controls_defaults_limit_and_request():
    test_user_controls = implementation.UserControls()

    default_limit = implementation.Limit()
    default_requests = implementation.Request()

    model_dump = test_user_controls.model_dump(by_alias=True, exclude_none=True)

    assert model_dump["limits"] == default_limit.model_dump(by_alias=True, exclude_none=True)
    assert model_dump["requests"] == default_requests.model_dump(by_alias=True, exclude_none=True)


def test_user_controls_overrides_defaults_if_provided():
    from_config = {
        "image": "test",
        "limits": {"cpu": "1000m", "memory": "1Gi"},
        "requests": {"cpu": "500m", "memory": "1Gi"},
    }
    test_user_controls = implementation.UserControls(**from_config)

    model_dump = test_user_controls.model_dump(by_alias=True, exclude_none=True)

    assert model_dump["limits"] == {"cpu": "1000m", "memory": "1Gi"}
    assert model_dump["requests"] == {"cpu": "500m", "memory": "1Gi"}


def test_default_container_env_is_none_if_secrets_or_env_vars_are_empty():
    test_default_container = implementation.DefaultContainer(image="test", command="test_command")

    assert test_default_container.model_dump(by_alias=True)["env"] is None


def test_default_container_env_is_sum_of_k8s_secrets_and_env_vars():
    test_default_container = implementation.DefaultContainer(image="test", command="test_command")

    env_var = implementation.EnvVar(name="test_env", value="test_value")
    secret_var = implementation.SecretEnvVar(environment_variable="env", secret_name="name", secret_key="key")

    test_default_container._env_vars = [env_var]
    test_default_container._secrets_from_k8s = [secret_var]

    assert test_default_container.model_dump(by_alias=True, exclude_none=True)["env"] == [
        {**env_var.model_dump(by_alias=True, exclude_none=True)},
        {**secret_var.model_dump(by_alias=True, exclude_none=True)},
    ]


def test_user_controls_env_is_none_if_secrets_or_env_vars_are_empty():
    test_user_controls = implementation.UserControls()

    assert test_user_controls.model_dump(by_alias=True)["env"] is None


def test_user_controls_env_is_sum_of_k8s_secrets_and_env_vars():
    test_user_controls = implementation.UserControls()

    env_var = implementation.EnvVar(name="test_env", value="test_value")
    secret_var = implementation.SecretEnvVar(environment_variable="env", secret_name="name", secret_key="key")

    test_user_controls._env_vars = [env_var]
    test_user_controls._secrets_from_k8s = [secret_var]

    assert test_user_controls.model_dump(by_alias=True, exclude_none=True)["env"] == [
        {**env_var.model_dump(by_alias=True, exclude_none=True)},
        {**secret_var.model_dump(by_alias=True, exclude_none=True)},
    ]


def test_output_parameter_valuefrom_includes_path():
    test_out_put_parameter = implementation.OutputParameter(name="test_name", path="test_path")

    assert test_out_put_parameter.model_dump(by_alias=True, exclude_none=True) == {
        "name": "test_name",
        "valueFrom": {"path": "test_path"},
    }


def test_container_command_gets_split():
    test_container = implementation.Container(image="test_image", command="am I splitting?")

    assert test_container.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)["command"] == [
        "am",
        "I",
        "splitting?",
    ]

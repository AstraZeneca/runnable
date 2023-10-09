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
    user_controls = implementation.UserControls(image="test")

    assert user_controls.model_dump(by_alias=True, exclude_none=True) == {
        "image": "test",
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
        "valueFrom": "/tmp/output.txt",
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


def test_user_controls_defaults_limit_and_request():
    test_user_controls = implementation.UserControls(image="test")

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


def test_user_controls_can_be_asked_to_give_only_non_default_fields():
    test_user_controls = implementation.UserControls(image="test")

    assert test_user_controls.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True) == {"image": "test"}


def test_init_of_argo_executor(mocker, monkeypatch):
    template_defaults = implementation.UserControls(image="test")

    monkeypatch.setattr(implementation.ArgoExecutor, "_get_parameters", mocker.MagicMock(return_value={}))
    monkeypatch.setattr(
        implementation.ArgoExecutor, "_get_template_defaults", mocker.MagicMock(return_value=template_defaults)
    )

    test_executor = implementation.ArgoExecutor(image="test_image")

    assert test_executor.image == "test_image"
    assert test_executor._workflow.spec.template_defaults == template_defaults
    assert len(test_executor._workflow.spec.arguments) == 2


def test_init_of_argo_executor_adds_parameters_from_get_parameters(mocker, monkeypatch):
    template_defaults = implementation.UserControls(image="test")

    monkeypatch.setattr(implementation.ArgoExecutor, "_get_parameters", mocker.MagicMock(return_value={"a": 2}))
    monkeypatch.setattr(
        implementation.ArgoExecutor, "_get_template_defaults", mocker.MagicMock(return_value=template_defaults)
    )

    test_executor = implementation.ArgoExecutor(image="test_image")

    assert test_executor.image == "test_image"
    assert test_executor._workflow.spec.template_defaults == template_defaults
    assert len(test_executor._workflow.spec.arguments) == 3


def test_init_of_argo_executor_adds_pvcs_from_user_config(mocker, monkeypatch):
    template_defaults = implementation.UserControls(image="test")

    monkeypatch.setattr(implementation.ArgoExecutor, "_get_parameters", mocker.MagicMock(return_value={}))
    monkeypatch.setattr(
        implementation.ArgoExecutor, "_get_template_defaults", mocker.MagicMock(return_value=template_defaults)
    )

    user_config = {
        "image": "test_image",
        "persistent_volumes": [{"name": "test_volume", "mount_path": "/tmp/test_volume"}],
    }

    test_executor = implementation.ArgoExecutor(**user_config)

    assert test_executor._workflow.spec.volumes[0].model_dump(by_alias=True, exclude_none=True) == {
        "name": "executor-0",
        "persistentVolumeClaim": {"claimName": "test_volume"},
    }

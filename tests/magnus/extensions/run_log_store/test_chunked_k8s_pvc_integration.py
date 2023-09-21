import pytest

from magnus.extensions.run_log_store.chunked_k8s_pvc import integration


def test_k8s_pvc_errors_for_local():
    test_integration = integration.LocalCompute(executor="executor", integration_service="catalog")

    with pytest.raises(Exception, match="We can't use the local compute"):
        test_integration.validate()


def test_k8s_pvc_errors_for_local_container():
    test_integration = integration.LocalContainerCompute(executor="executor", integration_service="catalog")

    with pytest.raises(Exception, match="We can't use the local-container compute"):
        test_integration.validate()

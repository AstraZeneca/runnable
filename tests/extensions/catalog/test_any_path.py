import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from extensions.catalog.any_path import AnyPathCatalog
from runnable.datastore import DataCatalog


class TestAnyPathCatalog(AnyPathCatalog):
    """Concrete implementation of AnyPathCatalog for testing"""

    catalog_location: str = "test_catalog"

    def get_summary(self):
        return {"name": "test-catalog"}

    def upload_to_catalog(self, file: Path):
        pass

    def download_from_catalog(self, file: Path):
        pass

    def get_catalog_location(self) -> Path:
        return (
            Path(self.catalog_location)
            / self._context.run_id
            / self.compute_data_folder
        )


@pytest.fixture
def catalog_setup(mocker):
    """Setup basic catalog structure"""
    # Create a mock context
    mock_context = Mock()
    mock_context.run_id = "test_run"
    mock_context.run_log_store = Mock()
    mock_context.run_log_store.create_data_catalog.return_value = DataCatalog(
        name="test"
    )

    # Patch the context property at the module level
    mocker.patch("runnable.context.get_run_context", return_value=mock_context)

    catalog = TestAnyPathCatalog()

    # Create temporary directory structure
    os.makedirs("test_catalog/test_run/data", exist_ok=True)
    # catalog.catalog_location = "test_catalog"
    catalog.compute_data_folder = "data"

    yield catalog

    # Cleanup
    import shutil

    if os.path.exists("test_catalog"):
        shutil.rmtree("test_catalog")


def test_get_with_matching_files(catalog_setup):
    """Test get method with matching files"""
    # Create test files
    test_file = Path("test_catalog/test_run/data/test.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("test content")

    with patch("extensions.catalog.any_path.utils.get_data_hash") as mock_hash:
        # Update to expect SHA256 hash length (64 chars instead of 32)
        mock_hash.return_value = "a" * 64  # SHA256 hash length
        catalogs = catalog_setup.get("*.txt")

    assert len(catalogs) == 1
    assert catalogs[0].name == "test"
    assert catalogs[0].data_hash == "a" * 64  # Updated expectation
    assert catalogs[0].stage == "get"


def test_get_with_no_matching_files(catalog_setup):
    """Test get method with no matching files"""
    with pytest.raises(Exception) as exc_info:
        catalog_setup.get("nonexistent*.txt")

    assert "Did not find any files matching" in str(exc_info.value)


def test_get_ignores_execution_logs(catalog_setup):
    """Test get method ignores execution log files"""
    # Create test files
    test_file = Path("test_catalog/test_run/data/test.execution.log")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("test content")

    with pytest.raises(Exception) as exc_info:
        catalog_setup.get("*.log")

    assert "Did not find any files matching" in str(exc_info.value)


def test_put_with_matching_files(catalog_setup):
    """Test put method with matching files"""
    # Create test files in compute data folder
    os.makedirs("data", exist_ok=True)
    test_file = Path("data/test.txt")
    test_file.write_text("test content")

    with patch("extensions.catalog.any_path.utils.get_data_hash") as mock_hash:
        # Update to expect SHA256 hash length (64 chars instead of 32)
        mock_hash.return_value = "b" * 64  # SHA256 hash length
        catalogs = catalog_setup.put("*.txt")

    assert len(catalogs) == 1
    assert catalogs[0].name == "test"
    assert catalogs[0].data_hash == "b" * 64  # Updated expectation
    assert catalogs[0].stage == "put"

    # Cleanup
    if os.path.exists("data"):
        import shutil

        shutil.rmtree("data")


def test_put_with_no_matching_files(catalog_setup):
    """Test put method with no matching files"""
    os.makedirs("data", exist_ok=True)

    with pytest.raises(Exception) as exc_info:
        catalog_setup.put("nonexistent*.txt")

    assert "Did not find any files matching" in str(exc_info.value)

    # Cleanup
    if os.path.exists("data"):
        import shutil

        shutil.rmtree("data")


def test_put_with_missing_compute_folder(catalog_setup):
    """Test put method with missing compute data folder"""
    with pytest.raises(Exception) as exc_info:
        catalog_setup.put("*.txt")

    assert "Expected compute data folder to be present" in str(exc_info.value)

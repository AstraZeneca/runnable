import os
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from extensions.catalog.file_system import FileSystemCatalog


@pytest.fixture
def catalog_setup(mocker):
    """Setup test catalog environment"""
    # Create mock context
    mock_context = Mock()
    mock_context.run_id = "test_run"

    # Patch context at module level
    mocker.patch("runnable.context.get_run_context", return_value=mock_context)

    # Create test catalog
    catalog = FileSystemCatalog()
    catalog.compute_data_folder = "data"
    catalog.catalog_location = "test_catalog"

    # Create test directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("test_catalog/test_run/data", exist_ok=True)

    yield catalog

    # Cleanup
    if Path("data").exists():
        shutil.rmtree("data")
    if Path("test_catalog").exists():
        shutil.rmtree("test_catalog")


def test_get_summary(catalog_setup):
    """Test getting catalog summary"""
    summary = catalog_setup.get_summary()

    assert summary["compute_data_folder"] == "data"
    assert summary["catalog_location"] == "test_catalog"


def test_get_catalog_location(catalog_setup):
    """Test getting catalog location"""
    location = catalog_setup.get_catalog_location()

    assert location == Path("test_catalog/test_run/data")
    assert isinstance(location, Path)


def test_upload_to_catalog(catalog_setup):
    """Test uploading file to catalog"""
    # Create test file
    test_file = Path("data/test.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("test content")

    # Upload file
    catalog_setup.upload_to_catalog(test_file)

    # Verify file was uploaded
    uploaded_file = Path("test_catalog/test_run/data/test.txt")
    assert uploaded_file.exists()
    assert uploaded_file.read_text() == "test content"


def test_upload_to_catalog_nested(catalog_setup):
    """Test uploading nested file structure"""
    # Create nested test file
    test_file = Path("data/nested/deep/test.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("nested content")

    # Upload file
    catalog_setup.upload_to_catalog(test_file)

    # Verify file was uploaded with structure
    uploaded_file = Path("test_catalog/test_run/data/nested/deep/test.txt")
    assert uploaded_file.exists()
    assert uploaded_file.read_text() == "nested content"


def test_download_from_catalog(catalog_setup):
    """Test downloading file from catalog"""
    # Create test file in catalog
    catalog_file = Path("test_catalog/test_run/data/test.txt")
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    catalog_file.write_text("catalog content")

    # Download file
    catalog_setup.download_from_catalog(catalog_file)

    # Verify file was downloaded
    downloaded_file = Path("data/test.txt")
    assert downloaded_file.exists()
    assert downloaded_file.read_text() == "catalog content"


def test_download_from_catalog_nested(catalog_setup):
    """Test downloading nested file structure"""
    # Create nested test file in catalog
    catalog_file = Path("test_catalog/test_run/data/nested/deep/test.txt")
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    catalog_file.write_text("nested catalog content")

    # Download file
    catalog_setup.download_from_catalog(catalog_file)

    # Verify file was downloaded with structure
    downloaded_file = Path("data/nested/deep/test.txt")
    assert downloaded_file.exists()
    assert downloaded_file.read_text() == "nested catalog content"


def test_download_non_path(catalog_setup):
    """Test downloading with non-Path object raises error"""
    with pytest.raises(AssertionError):
        catalog_setup.download_from_catalog("not_a_path")


def test_upload_file_not_in_compute_folder(catalog_setup):
    """Test uploading file outside compute folder raises error"""
    test_file = Path("outside_data/test.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("test content")

    with pytest.raises(ValueError):
        catalog_setup.upload_to_catalog(test_file)

    # Cleanup
    if Path("outside_data").exists():
        shutil.rmtree("outside_data")

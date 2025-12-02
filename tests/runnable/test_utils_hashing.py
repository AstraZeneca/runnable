import pytest
from unittest.mock import patch, mock_open
from pathlib import Path
from runnable.utils import get_data_hash

def test_get_data_hash_small_file_full_hash():
    """Test that small files get full SHA256 hash"""
    file_content = b"small file content"

    with patch("pathlib.Path.stat") as mock_stat, \
         patch("builtins.open", mock_open(read_data=file_content)) as mock_file:

        mock_stat.return_value.st_size = len(file_content)

        result = get_data_hash("test_file.txt")

        # Should be SHA256 of full content
        import hashlib
        expected = hashlib.sha256(file_content).hexdigest()
        assert result == expected

def test_get_data_hash_large_file_fingerprint():
    """Test that large files get fingerprint hash (first/last chunks + metadata)"""
    # Create mock file larger than threshold (default 1GB)
    large_file_size = 2 * 1024 * 1024 * 1024  # 2GB
    first_chunk = b"A" * 1024 * 1024  # 1MB first chunk
    last_chunk = b"Z" * 1024 * 1024   # 1MB last chunk

    with patch("pathlib.Path.stat") as mock_stat, \
         patch("builtins.open", mock_open()) as mock_file:

        mock_stat.return_value.st_size = large_file_size

        # Mock file reads: first chunk, then last chunk
        mock_file.return_value.read.side_effect = [first_chunk, last_chunk, b""]

        result = get_data_hash("large_file.bin")

        # Should be SHA256 of (file_size + first_chunk + last_chunk)
        import hashlib
        hasher = hashlib.sha256()
        hasher.update(str(large_file_size).encode())
        hasher.update(first_chunk)
        hasher.update(last_chunk)
        expected = hasher.hexdigest()

        assert result == expected


def test_get_data_hash_empty_file():
    """Test hash computation for empty files"""
    with patch("pathlib.Path.stat") as mock_stat, \
         patch("builtins.open", mock_open(read_data=b"")) as mock_file:

        mock_stat.return_value.st_size = 0

        result = get_data_hash("empty_file.txt")

        # Should be SHA256 of empty content
        import hashlib
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected


def test_get_data_hash_exactly_threshold_size():
    """Test hash computation for file exactly at threshold size"""
    threshold_size = 1024 * 1024 * 1024  # 1GB exactly
    file_content = b"X" * 1024  # Small content for mock

    with patch("pathlib.Path.stat") as mock_stat, \
         patch("builtins.open", mock_open(read_data=file_content)) as mock_file:

        mock_stat.return_value.st_size = threshold_size

        result = get_data_hash("threshold_file.bin")

        # Should use fingerprint method (large file handling)
        assert len(result) == 64  # SHA256 hex length


def test_get_data_hash_file_not_found():
    """Test hash computation handles file not found error gracefully"""
    with pytest.raises(FileNotFoundError):
        get_data_hash("nonexistent_file.txt")


def test_get_data_hash_permission_error():
    """Test hash computation handles permission errors gracefully"""
    with patch("pathlib.Path.stat") as mock_stat, \
         patch("builtins.open", side_effect=PermissionError("Access denied")):

        mock_stat.return_value.st_size = 100  # Mock file size

        with pytest.raises(PermissionError):
            get_data_hash("restricted_file.txt")


def test_get_data_hash_performance_logging():
    """Test that performance metrics are logged for large files"""
    large_file_size = 2 * 1024 * 1024 * 1024  # 2GB

    with patch("pathlib.Path.stat") as mock_stat, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("runnable.utils.logger") as mock_logger:

        mock_stat.return_value.st_size = large_file_size
        mock_file.return_value.read.side_effect = [b"A" * 1024, b"Z" * 1024, b""]

        get_data_hash("large_file.bin")

        # Should log performance info for large files
        mock_logger.info.assert_called()
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        performance_logged = any("fingerprint hash computed" in msg.lower() for msg in log_calls)
        assert performance_logged

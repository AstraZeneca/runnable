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

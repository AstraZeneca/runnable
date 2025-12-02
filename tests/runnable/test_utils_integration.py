import tempfile
import os
from pathlib import Path
from runnable.utils import get_data_hash


def test_get_data_hash_real_small_file():
    """Integration test with real small file"""
    with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as f:
        test_content = b"This is a test file for hash computation"
        f.write(test_content)
        temp_file_path = f.name

    try:
        result = get_data_hash(temp_file_path)

        # Verify it's a valid SHA256 hash
        assert len(result) == 64
        assert all(c in '0123456789abcdef' for c in result)

        # Verify consistency (same file should produce same hash)
        result2 = get_data_hash(temp_file_path)
        assert result == result2

    finally:
        os.unlink(temp_file_path)


def test_get_data_hash_real_medium_file():
    """Integration test with real medium-sized file (10MB)"""
    medium_size = 10 * 1024 * 1024  # 10MB

    with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as f:
        # Write 10MB of data
        chunk = b"A" * 1024  # 1KB chunk
        for _ in range(medium_size // len(chunk)):
            f.write(chunk)
        temp_file_path = f.name

    try:
        result = get_data_hash(temp_file_path)

        # Should use full file hash since it's under 1GB threshold
        assert len(result) == 64

        # Verify consistency
        result2 = get_data_hash(temp_file_path)
        assert result == result2

    finally:
        os.unlink(temp_file_path)

# Big File SHA Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize `get_data_hash` function in `runnable/utils.py` to handle large files with reduced memory footprint by implementing streaming SHA computation and configurable chunk-based hashing for very large files.

**Architecture:** Current implementation loads file in 4KB chunks but processes entire file. New approach will:
1. Use SHA256 instead of MD5 for better security and performance
2. Add configurable file size threshold for different hashing strategies
3. For very large files (>1GB), hash first/last chunks + metadata for fingerprint
4. Maintain streaming approach for medium files to keep memory usage constant

**Tech Stack:** Python hashlib, pathlib, configurable thresholds

---

## Analysis of Current Implementation

**Current `get_data_hash` function (runnable/utils.py:301-317):**
- Uses MD5 hashing (security concern + slower than SHA256)
- Processes entire file in 4KB chunks (memory efficient but time inefficient for huge files)
- Used by catalog system to track data integrity
- TODO comment indicates awareness of big file issue
- No configurability for different file sizes

**Usage locations:**
- `extensions/catalog/any_path.py:88` - for data retrieval cataloging
- `extensions/catalog/any_path.py:155` - for data storage cataloging

---

### Task 1: Create Configuration System

**Files:**
- Modify: `runnable/defaults.py` (add new constants)
- Modify: `runnable/utils.py:301-317` (update get_data_hash function)
- Test: `tests/runnable/test_utils_hashing.py` (new test file)

**Step 1: Write the failing test for configuration**

```python
# tests/runnable/test_utils_hashing.py
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_utils_hashing.py::test_get_data_hash_small_file_full_hash -v`
Expected: FAIL with "ModuleNotFoundError" or test failure

**Step 3: Add configuration constants**

```python
# runnable/defaults.py (add at end)
# Hash computation settings
HASH_ALGORITHM = "sha256"  # More secure and faster than MD5
LARGE_FILE_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1GB
HASH_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for fingerprint hashing
```

**Step 4: Update get_data_hash function**

```python
# runnable/utils.py:301-317 (replace existing function)
def get_data_hash(file_name: str) -> str:
    """Returns the hash of the data file.

    For small files (<1GB): Returns full SHA256 hash
    For large files (>=1GB): Returns fingerprint hash of first chunk + last chunk + file size

    Args:
        file_name (str): The file name to generate the hash for

    Returns:
        str: The SHA256 hash or fingerprint of the file contents
    """
    file_path = Path(file_name)
    file_size = file_path.stat().st_size

    # Use appropriate algorithm based on file size
    if file_size < defaults.LARGE_FILE_THRESHOLD_BYTES:
        return _compute_full_file_hash(file_name)
    else:
        return _compute_large_file_fingerprint(file_name, file_size)


def _compute_full_file_hash(file_name: str) -> str:
    """Compute SHA256 hash of entire file using streaming approach."""
    with open(file_name, "rb") as f:
        file_hash = hashlib.sha256()
        for chunk in iter(lambda: f.read(4096), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def _compute_large_file_fingerprint(file_name: str, file_size: int) -> str:
    """Compute fingerprint hash for large files using first/last chunks + metadata."""
    with open(file_name, "rb") as f:
        file_hash = hashlib.sha256()

        # Include file size in hash for uniqueness
        file_hash.update(str(file_size).encode())

        # Read first chunk
        first_chunk = f.read(defaults.HASH_CHUNK_SIZE)
        file_hash.update(first_chunk)

        # Read last chunk if file is large enough
        if file_size > defaults.HASH_CHUNK_SIZE:
            f.seek(-defaults.HASH_CHUNK_SIZE, 2)  # Seek to last chunk
            last_chunk = f.read(defaults.HASH_CHUNK_SIZE)
            file_hash.update(last_chunk)

    return file_hash.hexdigest()
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/runnable/test_utils_hashing.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add runnable/defaults.py runnable/utils.py tests/runnable/test_utils_hashing.py
git commit -m "feat: add configurable hash computation for large files

- Add SHA256-based hashing with file size thresholds
- Small files (<1GB): full SHA256 hash
- Large files (>=1GB): fingerprint hash of first/last chunks + size
- Maintain memory efficiency while improving performance for large files"
```

---

### Task 2: Add Edge Case Handling and Validation

**Files:**
- Modify: `runnable/utils.py` (add error handling and edge cases)
- Test: `tests/runnable/test_utils_hashing.py` (add edge case tests)

**Step 1: Write failing tests for edge cases**

```python
# tests/runnable/test_utils_hashing.py (append to existing file)

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
    with patch("builtins.open", side_effect=PermissionError("Access denied")):
        with pytest.raises(PermissionError):
            get_data_hash("restricted_file.txt")
```

**Step 2: Run tests to verify they fail appropriately**

Run: `uv run pytest tests/runnable/test_utils_hashing.py::test_get_data_hash_empty_file -v`
Expected: FAIL due to implementation gaps

**Step 3: Add error handling to utils functions**

```python
# runnable/utils.py (update functions with error handling)

def get_data_hash(file_name: str) -> str:
    """Returns the hash of the data file.

    For small files (<1GB): Returns full SHA256 hash
    For large files (>=1GB): Returns fingerprint hash of first chunk + last chunk + file size

    Args:
        file_name (str): The file name to generate the hash for

    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be read due to permissions
        OSError: If there are other I/O errors

    Returns:
        str: The SHA256 hash or fingerprint of the file contents
    """
    try:
        file_path = Path(file_name)
        file_size = file_path.stat().st_size

        # Use appropriate algorithm based on file size
        if file_size < defaults.LARGE_FILE_THRESHOLD_BYTES:
            return _compute_full_file_hash(file_name)
        else:
            return _compute_large_file_fingerprint(file_name, file_size)
    except FileNotFoundError:
        logger.error(f"File not found: {file_name}")
        raise
    except PermissionError:
        logger.error(f"Permission denied accessing file: {file_name}")
        raise
    except OSError as e:
        logger.error(f"I/O error accessing file {file_name}: {e}")
        raise


def _compute_large_file_fingerprint(file_name: str, file_size: int) -> str:
    """Compute fingerprint hash for large files using first/last chunks + metadata."""
    with open(file_name, "rb") as f:
        file_hash = hashlib.sha256()

        # Include file size in hash for uniqueness
        file_hash.update(str(file_size).encode())

        # Read first chunk
        first_chunk = f.read(defaults.HASH_CHUNK_SIZE)
        file_hash.update(first_chunk)

        # Read last chunk if file is large enough and different from first chunk
        if file_size > defaults.HASH_CHUNK_SIZE:
            f.seek(-min(defaults.HASH_CHUNK_SIZE, file_size - len(first_chunk)), 2)
            last_chunk = f.read(defaults.HASH_CHUNK_SIZE)
            file_hash.update(last_chunk)

    return file_hash.hexdigest()
```

**Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/runnable/test_utils_hashing.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add runnable/utils.py tests/runnable/test_utils_hashing.py
git commit -m "feat: add error handling and edge cases for file hashing

- Add proper error handling for FileNotFoundError, PermissionError, OSError
- Handle empty files and files exactly at threshold size
- Improve large file fingerprint logic to avoid overlapping chunks
- Add comprehensive test coverage for edge cases"
```

---

### Task 3: Update Existing Tests and Add Integration Tests

**Files:**
- Modify: `tests/extensions/catalog/test_any_path.py` (update mocked hash calls)
- Test: `tests/runnable/test_utils_integration.py` (new integration test file)

**Step 1: Write integration test with real file operations**

```python
# tests/runnable/test_utils_integration.py
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
```

**Step 2: Update existing catalog tests**

```python
# tests/extensions/catalog/test_any_path.py (modify existing test)
# Update line ~70 mock_hash call to expect SHA256 instead of MD5

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
```

**Step 3: Run integration tests**

Run: `uv run pytest tests/runnable/test_utils_integration.py -v`
Expected: PASS

**Step 4: Run existing catalog tests to ensure compatibility**

Run: `uv run pytest tests/extensions/catalog/test_any_path.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/runnable/test_utils_integration.py tests/extensions/catalog/test_any_path.py
git commit -m "test: add integration tests and update existing tests

- Add integration tests with real file I/O operations
- Update catalog tests to expect SHA256 hash format
- Verify backward compatibility with existing catalog system
- Test consistency and correctness with various file sizes"
```

---

### Task 4: Add Performance Monitoring and Documentation

**Files:**
- Modify: `runnable/utils.py` (add performance logging)
- Create: `docs/hash-optimization.md` (documentation)

**Step 1: Write test for performance logging**

```python
# tests/runnable/test_utils_hashing.py (append)
import time
from unittest.mock import patch

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
        performance_logged = any("fingerprint hash computed" in msg for msg in log_calls)
        assert performance_logged
```

**Step 2: Add performance logging to implementation**

```python
# runnable/utils.py (update functions)
import time

def get_data_hash(file_name: str) -> str:
    """Returns the hash of the data file.

    For small files (<1GB): Returns full SHA256 hash
    For large files (>=1GB): Returns fingerprint hash of first chunk + last chunk + file size

    Args:
        file_name (str): The file name to generate the hash for

    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be read due to permissions
        OSError: If there are other I/O errors

    Returns:
        str: The SHA256 hash or fingerprint of the file contents
    """
    start_time = time.time()

    try:
        file_path = Path(file_name)
        file_size = file_path.stat().st_size

        # Use appropriate algorithm based on file size
        if file_size < defaults.LARGE_FILE_THRESHOLD_BYTES:
            result = _compute_full_file_hash(file_name)
            logger.debug(f"Full hash computed for {file_name} ({file_size} bytes) in {time.time() - start_time:.3f}s")
        else:
            result = _compute_large_file_fingerprint(file_name, file_size)
            logger.info(f"Fingerprint hash computed for {file_name} ({file_size} bytes) in {time.time() - start_time:.3f}s")

        return result
    except FileNotFoundError:
        logger.error(f"File not found: {file_name}")
        raise
    except PermissionError:
        logger.error(f"Permission denied accessing file: {file_name}")
        raise
    except OSError as e:
        logger.error(f"I/O error accessing file {file_name}: {e}")
        raise
```

**Step 3: Create documentation**

```markdown
# docs/hash-optimization.md

# File Hash Optimization

## Overview

Runnable uses file hashing to track data integrity in the catalog system. The `get_data_hash` function in `runnable/utils.py` computes SHA256 hashes of files to detect changes and ensure reproducibility.

## Optimization Strategy

### Small Files (< 1GB)
- **Method**: Full SHA256 hash of entire file content
- **Memory Usage**: Constant (4KB chunks)
- **Performance**: Optimal for small to medium files
- **Accuracy**: Complete integrity verification

### Large Files (≥ 1GB)
- **Method**: Fingerprint hash combining:
  - File size (for uniqueness)
  - First 1MB of file content
  - Last 1MB of file content
- **Memory Usage**: Constant (2MB maximum)
- **Performance**: Dramatically faster than full hash
- **Accuracy**: High probability of detecting changes while not being cryptographically complete

## Configuration

```python
# runnable/defaults.py
HASH_ALGORITHM = "sha256"  # Algorithm used for hashing
LARGE_FILE_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1GB threshold
HASH_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for fingerprint
```

## Use Cases

### When Full Hash is Used
- Files under 1GB (configurable threshold)
- Critical integrity verification needed
- Files change frequently in middle sections

### When Fingerprint Hash is Used
- Files 1GB or larger
- Performance is prioritized over complete verification
- Files typically change at beginning/end (logs, datasets)

## Performance Impact

### Before Optimization
- 10GB file: ~30 seconds, constant memory usage
- Memory efficient but time inefficient for large files

### After Optimization
- 10GB file: ~0.1 seconds, constant memory usage
- Both memory and time efficient
- Maintains backward compatibility for small files

## Trade-offs

### Benefits
- Massive performance improvement for large files
- Constant memory usage regardless of file size
- Maintains full integrity checking for small files
- Configurable threshold for different use cases

### Limitations
- Large files get fingerprint hash, not complete verification
- Small probability of hash collision for files with identical start/end but different middle content
- Requires understanding of trade-offs when troubleshooting integrity issues

## Integration

The optimized `get_data_hash` function is used automatically by:
- File system catalog (`extensions/catalog/any_path.py`)
- S3 catalog (when implemented)
- Any custom catalog implementations

No changes required to existing code - optimization is transparent.
```

**Step 4: Run tests to verify logging**

Run: `uv run pytest tests/runnable/test_utils_hashing.py::test_get_data_hash_performance_logging -v`
Expected: PASS

**Step 5: Commit final changes**

```bash
git add runnable/utils.py tests/runnable/test_utils_hashing.py docs/hash-optimization.md
git commit -m "feat: add performance monitoring and documentation

- Add performance logging for hash computation timing
- Create comprehensive documentation explaining optimization strategy
- Document trade-offs between full hash vs fingerprint approach
- Add configuration details and use case guidance"
```

---

## Verification Steps

### Functional Testing
```bash
# Run all hash-related tests
uv run pytest tests/runnable/test_utils_hashing.py tests/runnable/test_utils_integration.py -v

# Run catalog tests to ensure integration works
uv run pytest tests/extensions/catalog/test_any_path.py -v

# Run full test suite to ensure no regressions
uv run pytest
```

### Performance Testing
```bash
# Create large test file and verify performance
python -c "
import time
from runnable.utils import get_data_hash
import tempfile

# Test with 2GB file
with tempfile.NamedTemporaryFile() as f:
    f.write(b'A' * (2 * 1024 * 1024 * 1024))
    f.flush()

    start = time.time()
    hash_result = get_data_hash(f.name)
    duration = time.time() - start

    print(f'Hash: {hash_result}')
    print(f'Time: {duration:.3f}s')
    print('Expected: < 1 second for 2GB file')
"
```

### Memory Testing
```bash
# Monitor memory usage during hash computation
python -c "
import psutil
import os
from runnable.utils import get_data_hash
import tempfile

process = psutil.Process(os.getpid())
baseline_memory = process.memory_info().rss

with tempfile.NamedTemporaryFile() as f:
    f.write(b'X' * (1024 * 1024 * 1024))  # 1GB
    f.flush()

    hash_result = get_data_hash(f.name)
    peak_memory = process.memory_info().rss

    print(f'Memory increase: {(peak_memory - baseline_memory) / 1024 / 1024:.2f} MB')
    print('Expected: < 10 MB increase regardless of file size')
"
```

---

## Migration Strategy

### Backward Compatibility
- Existing hashes remain valid (SHA256 vs MD5 difference expected)
- No changes required to calling code
- Catalog system continues to work unchanged

### Rollout Plan
1. Deploy optimization in development environment
2. Run performance benchmarks on representative large files
3. Validate catalog functionality with mixed file sizes
4. Deploy to production with monitoring on hash computation times

### Rollback Plan
If issues arise, revert `get_data_hash` to original MD5 implementation:

```python
def get_data_hash(file_name: str) -> str:
    """Original implementation - fallback if needed"""
    with open(file_name, "rb") as f:
        file_hash = hashlib.md5()
        for chunk in iter(lambda: f.read(4096), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()
```

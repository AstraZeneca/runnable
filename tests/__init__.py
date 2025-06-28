"""
Tests package for Runnable
"""

import sys
from pathlib import Path

# Add the project root to the Python path for all tests
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

"""pytest configuration for RePAIR dissertation test suite."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so tests can import
# registration, uncertainty, and scripts packages.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

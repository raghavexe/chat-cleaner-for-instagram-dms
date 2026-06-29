# tests/conftest.py
# Shared fixtures and path setup — pytest picks this up automatically.

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
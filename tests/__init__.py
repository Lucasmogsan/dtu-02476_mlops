"""
Helps find paths during testing.
"""
import os
import toml
import sys
from pathlib import Path

pyproject_data = toml.load(Path("pyproject.toml"))
_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project

_DATA_PATH = os.path.join(_PROJECT_ROOT, "data/")
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models/")

_PROJECT_NAME = pyproject_data["project"]["name"]
_SRC_PATH = os.path.join(_PROJECT_ROOT, _PROJECT_NAME)

sys.path.append(_SRC_PATH)  # add src to path for imports

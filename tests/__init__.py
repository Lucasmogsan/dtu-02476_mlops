"""
Helps find paths during testing.
"""
import os
import toml
from pathlib import Path

pyproject_data = toml.load(Path("pyproject.toml"))
_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_DATA_PATH = os.path.join(_PROJECT_ROOT, "data/")
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models/")
_SRC_PATH = os.path.join(_PROJECT_ROOT, pyproject_data["project"]["name"])
_PROJECT_NAME = pyproject_data["project"]["name"]

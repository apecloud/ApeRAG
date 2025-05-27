import os
import sys

from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

print(f"Added to sys.path: {project_root}")


def load_test_environment():
    """Loads environment variables from .env.test located in the same directory as this conftest.py."""
    # Assuming .env.test is in tests/unit_test/ alongside conftest.py
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env.test')
    print(f"\n--- Attempting to load environment variables from: {dotenv_path} ---")
    # override=True ensures that variables from .env.test take precedence
    loaded = load_dotenv(dotenv_path=dotenv_path, override=True, verbose=True)
    if loaded:
        print(f"Successfully loaded: {dotenv_path}")
    else:
        abs_dotenv_path = os.path.abspath(dotenv_path)
        print(f"Warning: .env.test file not found or empty at {abs_dotenv_path}")


# --- Pytest Hook to Load Environment *Before* Collection ---
def pytest_configure(config):
    """
    Pytest hook called after command line options are parsed and before
    test collection begins. This is the right place to load environment
    variables needed for module-level decorators like skipif.
    """
    config.option.log_cli = True
    config.option.log_cli_level = "INFO"
    print("\npytest_configure: Loading test environment...")
    load_test_environment()


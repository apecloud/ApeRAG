import subprocess


def test_core_python_files_are_ruff_formatted():
    result = subprocess.run(
        ["uvx", "ruff", "format", "--check", "./aperag", "./tests"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr

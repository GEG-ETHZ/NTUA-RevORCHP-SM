"""Check that the models still return the stored reference outputs.

A failure means an output changed. See tests/generate_reference.py before
regenerating the reference files.
"""
from __future__ import annotations

import json

import pytest

from cases import DATA, FUNCTION_CASES, differences, readme_commands, run_function_case, run_readme_command


def load(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def check(expected, actual) -> None:
    problems = differences(expected, actual)
    if problems:
        shown = "\n".join(problems[:20])
        more = f"\n... and {len(problems) - 20} more" if len(problems) > 20 else ""
        pytest.fail(f"{len(problems)} output(s) differ from the reference:\n{shown}{more}")


def test_readme_commands_match_reference_file():
    """Every README example has a stored output, and nothing extra is stored."""
    assert readme_commands(), "no command examples found in README.md"
    assert sorted(readme_commands()) == sorted(load("readme_commands"))


@pytest.mark.parametrize("command", readme_commands())
def test_readme_command(command):
    check(load("readme_commands")[command], run_readme_command(command))


def test_function_cases_match_reference_files():
    for group, cases in FUNCTION_CASES.items():
        assert sorted(cases) == sorted(load(group)), group


@pytest.mark.parametrize(
    "group, name", [(group, name) for group, cases in FUNCTION_CASES.items() for name in cases])
def test_function_case(group, name):
    check(load(group)[name], run_function_case(group, name))

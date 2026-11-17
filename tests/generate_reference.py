"""Rewrite the reference outputs in tests/data/ used by the regression tests.

Usage, from the repository directory:

    python tests/generate_reference.py

The tests compare the current code against these files, so regenerating them
makes the tests pass whatever the code now returns. Only regenerate when:

- the models or the physics were changed on purpose (for example a retrained
  bundle or a corrected equation), and the new numbers have been checked; or
- a README command example or a test case in tests/cases.py was added or
  changed, so there is no stored output for it yet.

Do NOT regenerate to silence a failing test after a refactor, clean-up or
dependency update: in that case the failure means an output changed, and the
change should be understood first. Review the diff of tests/data/ before
committing regenerated files.
"""
from __future__ import annotations

import json

from cases import DATA, FUNCTION_CASES, readme_commands, run_function_case, run_readme_command


def write(name: str, data: dict) -> None:
    path = DATA / f"{name}.json"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    write("readme_commands", {command: run_readme_command(command) for command in readme_commands()})
    for group, cases in FUNCTION_CASES.items():
        write(group, {name: run_function_case(group, name) for name in cases})


if __name__ == "__main__":
    main()

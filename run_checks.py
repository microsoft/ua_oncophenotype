#! /usr/bin/env python
"""
We have a mono-repository with multiple types of code (libraries/experiments/etc.),
so we only run checks on certain directories.  The default checks are used as
pre-commit hooks.  Format the files prior to checking by passing the --format argument.
"""

import argparse
from subprocess import run, PIPE
from subprocess import CalledProcessError
import sys
from pathlib import Path
from typing import List

GIT_REPO_BASE_CMD = ["git", "rev-parse", "--show-toplevel"]


def get_git_repo_base():
    git_output = run(GIT_REPO_BASE_CMD, stdout=PIPE)
    repo_base = git_output.stdout.decode("utf-8").strip()
    return Path(repo_base).resolve()


REPO_BASE = get_git_repo_base()  # the git repository base directory
PYTHON_DIR = REPO_BASE
CONFIG_DIR = (PYTHON_DIR / "scripts").resolve()


def run_check(checks: List[str], directories: List[str]):
    for dir_to_check in directories:
        print(f"### Checking Directory {dir_to_check} ###")
        check_dir(checks, dir_to_check)


def run_format(formatters: List[str], directories: List[str]):
    for dir_to_check in directories:
        print(f"### Checking Directory {dir_to_check} ###")
        format_dir(formatters, dir_to_check)


def format_dir(formatters: List[str], directory: str):
    try:
        if "autoflake" in formatters:
            print("Formatter (autoflake) removing unused imports", flush=True)
            run(f"autoflake -ri {directory}", shell=True, check=True)
            print("autoflake formatting completed")

        if "isort" in formatters:
            print("Formatter (isort) sorting imports", flush=True)
            run(
                f"isort --settings {CONFIG_DIR}/.isort.cfg {directory}",
                shell=True,
                check=True,
            )
            print("isort formatting completed")

        if "black" in formatters:
            print("Formatter (black) formatting source", flush=True)
            run(f"black -v --preview {directory}", shell=True, check=True)
            print("black formatting completed")

    except CalledProcessError as e:
        # squelch the exception stacktrace
        print(str(e))
        sys.exit(1)


def check_dir(checks: List[str], directory: str):
    try:
        if "isort" in checks:
            print("Formatter (isort)", flush=True)
            run(
                f"isort --check-only --settings {CONFIG_DIR}/.isort.cfg --check {directory}",
                shell=True,
                check=True,
            )
            print("isort formatting completed")

        if "black" in checks:
            print("Formatter (black)", flush=True)
            run(f"black -v --preview --check {directory}", shell=True, check=True)
            print("black formatting completed")

        if "flake8" in checks:
            print("Linter (flake8)", flush=True)
            run(
                f"flake8 -v --config={CONFIG_DIR}/.flake8 {directory}",
                shell=True,
                check=True,
            )
            print("flake8 checks passed")

        if "pytest" in checks:
            print("unit tests (pytest)", flush=True)
            run(f"pytest {directory}", shell=True, check=True)
            print("pytest checks passed")

        if "pylint" in checks:
            print(
                "Linter (pylint): NOTE: just informational, not required.", flush=True
            )
            run(
                f"pylint --rcfile={CONFIG_DIR}/.pylintrc {directory}",
                shell=True,
                check=False,  # don't actually require passing
            )
            print("pylint checks completed")

    except CalledProcessError as e:
        # squelch the exception stacktrace
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    all_formatters = ["isort", "autoflake", "black"]
    all_checks = ["isort", "autoflake", "black", "flake8", "pytest", "pylint"]
    default_checks = ["isort", "autoflake", "black", "flake8"]

    default_dirs = [
        PYTHON_DIR / "ua_oncophenotype",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checks", default=default_checks, nargs="+", choices=all_checks
    )
    parser.add_argument("--all-checks", action="store_true", help="run all checks")
    parser.add_argument("--dirs", default=default_dirs, nargs="+")
    parser.add_argument("-f", "--format", action="store_true", help="auto-format code")
    parser.add_argument(
        "-t",
        "--unit_test",
        action="store_true",
        help="run unit tests (add 'pytest' to --checks)",
    )
    parser.add_argument(
        "--formatters", default=all_formatters, nargs="+", choices=all_formatters
    )
    parser.add_argument(
        "-n", "--no-checks", action="store_true", help="skip default checks"
    )

    args = parser.parse_args()
    checks_to_run = list(args.checks)
    if args.all_checks:
        checks_to_run = all_checks

    if args.format:
        run_format(formatters=args.formatters, directories=args.dirs)

    if args.no_checks:
        checks_to_run = []

    if args.unit_test:
        checks_to_run.append("pytest")

    run_check(checks=checks_to_run, directories=args.dirs)

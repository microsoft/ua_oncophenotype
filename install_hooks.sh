#!/bin/bash

# Just run this from anywhere in the git repo and it installs 'run_checks.py' as a
# pre-commit hook. To skip the checks, use the -n / --no-verify flag with git commit
# (for instance to commit a WIP).

repo_base=`git rev-parse --show-toplevel`
ln -s "${repo_base}/run_checks.py" "${repo_base}/.git/hooks/pre-commit"

#!/usr/bin/env bash
#
# Usage: ./generate_requirements.sh
#        Creates/overwrites a requirements.txt in the current directory

# Generate requirements.txt by parsing pip list output:
#  1. Use --format=columns for consistent spacing.
#  2. Skip the first two header lines.
#  3. Convert each line "Package  Version" to "Package==Version".
pip list --format=columns \
  | tail -n +3 \
  | awk '{print $1 "==" $2}' \
  > requirements.txt

echo "Successfully created requirements.txt from pip list."

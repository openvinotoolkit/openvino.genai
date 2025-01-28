#!/bin/bash

# Store the command output:
empty_tree=$(git hash-object -t tree /dev/null)

# Get a list of new files that might have non-ASCII characters:
problem_files=$(git diff --name-only --diff-filter=A -z "$empty_tree" | LC_ALL=C grep -P "[^\x00-\x7F]")

# Count the number of problematic files:
count=$(echo "$problem_files" | wc -w)

# Print necessary info based on the result:
if [ "$count" -ne 0 ]; then
  echo "Error: Non-ASCII characters found in filenames of new files:"
  echo "$problem_files"
  exit 1
else
  echo "Success: No non-ASCII filenames found."
fi
exit 0

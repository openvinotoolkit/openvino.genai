#!/bin/bash

# Define the list of files to check, excluding .md, hidden, and a number of specific files:
files_to_check=$(git ls-files | grep -vE "^\." | grep -vE "\.md$" | grep -vE "^(tests/python_tests|tools/who_what_benchmark/(tests|whowhatbench))" | grep -v "tools/llm_bench/llm_bench_utils/ov_model_classes.py")

# Run git grep to find non-ASCII characters in the selected files and store the results:
results=$(LC_ALL=C git grep -n "[^ -~±�“”]" -- $files_to_check)

# Print the results:
if [ -n "$results" ]; then
  echo "Error: Non-ASCII characters found in files:"
  echo "$results"
  exit 1
else
  echo "Success: No non-ASCII characters found in files."
fi
exit 0

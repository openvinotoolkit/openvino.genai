#!/usr/bin/env bash

set -euo pipefail

# Ensure release data comes from the canonical repository.
expected_upstream_url_pattern='^(https://github\.com/|git@github\.com:)openvinotoolkit/openvino\.genai\.git$'
upstream_url=$(git remote get-url upstream 2>/dev/null || true)

if [[ -z "$upstream_url" ]]; then
  echo "Error: Git remote 'upstream' is not configured." >&2
  exit 1
fi

if [[ ! "$upstream_url" =~ $expected_upstream_url_pattern ]]; then
  echo "Error: Git remote 'upstream' points to '$upstream_url'; expected the canonical openvinotoolkit/openvino.genai GitHub repository." >&2
  exit 1
fi

# Refresh upstream branches and release tags.
git fetch upstream --tags

# Use the newest version-sorted tag as the previous release.
base_tag=$(git tag --list --sort=version:refname | tail -1)

if [[ ! "$base_tag" =~ ^[0-9]{4}\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: latest tag '$base_tag' does not match YYYY.N.N.N." >&2
  exit 1
fi

# Select the newest canonical release branch, excluding suffixed variants.
current_branch=$(git branch -r --sort=version:refname --format='%(refname:short)' |
  grep -E '^upstream/releases/[0-9]{4}/[0-9]+$' |
  tail -1)

if [[ -z "$current_branch" ]]; then
  echo "Error: no upstream release branch matches releases/YYYY/N." >&2
  exit 1
fi

printf 'Base tag: %s\nCurrent release branch: %s\n' "$base_tag" "$current_branch" >&2

# Print release commits as author;hash;subject and omit Dependabot updates.
git log "$base_tag..$current_branch" --pretty=tformat:'%an;%H;%s' |
  sed '/^dependabot\[bot\];/d'

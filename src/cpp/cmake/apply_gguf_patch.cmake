# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Idempotently applies the gguflib security patch to the fetched sources.
# Invoked as the FetchContent PATCH_COMMAND. Re-running configure must not
# fail on an already-patched tree, so we first check whether the patch is
# already present (reverse-apply check) and skip if so.
#
# Required cache/-D variables: GIT_EXECUTABLE, PATCH_FILE

execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --reverse --check --ignore-whitespace "${PATCH_FILE}"
    RESULT_VARIABLE already_applied
    ERROR_QUIET OUTPUT_QUIET)

if(already_applied EQUAL 0)
    message(STATUS "gguflib security patch already applied, skipping")
    return()
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --ignore-whitespace "${PATCH_FILE}"
    RESULT_VARIABLE apply_result)

if(NOT apply_result EQUAL 0)
    message(FATAL_ERROR "Failed to apply gguflib security patch '${PATCH_FILE}'")
endif()

message(STATUS "Applied gguflib security patch")

// Copyright(C) 2025 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0

/** Serialize a JavaScript object to a JSON string
 * with specific formatting to align with Python. */
export function serialize_json(object) {
    return JSON.stringify(object)
        .replaceAll('":', '": ')
        .replaceAll('",', '", ');
}

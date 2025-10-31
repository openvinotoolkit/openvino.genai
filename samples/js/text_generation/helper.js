// Copyright(C) 2025 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0

import { z } from 'zod';

/** Serialize a JavaScript object to a JSON string
 * with specific formatting to align with Python. */
export function serialize_json(object) {
    return JSON.stringify(object)
        .replace(/(:|,)(?! )/g, '$1 ');
}

/** Convert a Zod schema to a JSON Schema
 * with specific formatting to align with Python */
export function toJSONSchema(zodSchema, params) {
    const jsonSchema = z.toJSONSchema(
        zodSchema,
        {
            override: (ctx) => {
                if (params && params.override) {
                    params.override(ctx);
                }
                const keys = Object.keys(ctx.jsonSchema).sort();
                for (const key of keys) {
                    const value = ctx.jsonSchema[key];
                    delete ctx.jsonSchema[key];
                    ctx.jsonSchema[key] = value;
                }
            }
        });
    delete jsonSchema.$schema;
    delete jsonSchema.additionalProperties;
    return jsonSchema;
}
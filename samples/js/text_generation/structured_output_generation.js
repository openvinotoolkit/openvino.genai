// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import readline from 'readline';
import { z } from 'zod';
import { LLMPipeline, StructuredOutputConfig } from 'openvino-genai-node';
import { exit } from 'process';

// Define schemas using zod
const PersonSchema = z.object({
    name: z.string().regex(/^[A-Z][a-z]{1,20}$/),
    surname: z.string().regex(/^[A-Z][a-z]{1,20}$/),
    age: z.number(),
    city: z.enum(["Dublin", "Dubai", "Munich"]),
});

const CarSchema = z.object({
    model: z.string().regex(/^[A-Z][a-z]{1,20} ?[A-Z][a-z]{0,20} ?.?$/),
    year: z.number(),
    engine_type: z.enum(["diesel", "petrol", "electric", "hybrid"]),
});

const TransactionSchema = z.object({
    id: z.number().gte(1000).lte(10_000_000),
    amount: z.number(),
    currency: z.enum(["EUR", "PLN", "RUB", "AED", "CHF", "GBP", "USD"]),
});

const ItemQuantitiesSchema = z.object({
    person: z.number().gte(0).lte(100),
    car: z.number().gte(0).lte(100),
    transaction: z.number().gte(0).lte(100),
});

const itemsMap = {
    person: PersonSchema,
    car: CarSchema,
    transaction: TransactionSchema,
};

const sysMessage = `You generate JSON objects based on the user's request. You can generate JSON objects with different types of objects: person, car, transaction. 
If the user requested a different type, the JSON fields should remain zero. 
Please note that the words 'individual', 'person', 'people', 'man', 'human', 'woman', 'inhabitant', 'citizen' are synonyms and can be used interchangeably. 
E.g. if the user wants 5 houses, then the JSON must be {"person": 0, "car": 0, "transaction": 0}. 
If the user wants 3 people and 1 house, then the JSON must be {"person": 3, "car": 0, "transaction": 0}. 
Make sure that the JSON contains the numbers that the user requested. If the user asks for specific attributes, like 'surname', 'model', etc., 
ignore this information and generate JSON objects with the same fields as in the schema. 
Please use double quotes for JSON keys and values.`;

const sysMessageForItems = `Please try to avoid generating the same JSON objects multiple times.`;

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    const modelDir = process.argv[2];
    if (!modelDir) {
        console.error('Please provide the path to the model directory as the first argument.');
        process.exit(1);
    }

    const device = 'CPU'; // GPU can be used as well
    const pipe = await LLMPipeline(modelDir, device);

    const config = {};
    config.return_decoded_results = true;
    config.max_new_tokens = 300;

    console.log("This is a smart assistant that generates structured output in JSON format. " +
        "You can ask to generate information about a person, car, or bank transaction. " +
        'For example, you can ask: "Please generate jsons for 3 persons and 1 transaction."');

    const pyJsonSchema = {
        'general': '{"properties": {"person": {"maximum": 100, "minimum": 0, "title": "Person", "type": "integer"}, "car": {"maximum": 100, "minimum": 0, "title": "Car", "type": "integer"}, "transaction": {"maximum": 100, "minimum": 0, "title": "Transaction", "type": "integer"}}, "required": ["person", "car", "transaction"], "title": "ItemQuantities", "type": "object"}',
        'person': '{"properties": {"name": {"pattern": "^[A-Z][a-z]{1,20}$", "title": "Name", "type": "string"}, "surname": {"pattern": "^[A-Z][a-z]{1,20}$", "title": "Surname", "type": "string"}, "age": {"title": "Age", "type": "integer"}, "city": {"enum": ["Dublin", "Dubai", "Munich"], "title": "City", "type": "string"}}, "required": ["name", "surname", "age", "city"], "title": "Person", "type": "object"}',
        'car': '{"properties": {"model": {"pattern": "^[A-Z][a-z]{1,20} ?[A-Z][a-z]{0,20} ?.?$", "title": "Model", "type": "string"}, "year": {"title": "Year", "type": "integer"}, "engine_type": {"enum": ["diesel", "petrol", "electric", "hybrid"], "title": "Engine Type", "type": "string"}}, "required": ["model", "year", "engine_type"], "title": "Car", "type": "object"}',
        'transaction': '{"properties": {"id": {"maximum": 10000000, "minimum": 1000, "title": "Id", "type": "integer"}, "amount": {"title": "Amount", "type": "number"}, "currency": {"enum": ["EUR", "PLN", "RUB", "AED", "CHF", "GBP", "USD"], "title": "Currency", "type": "string"}}, "required": ["id", "amount", "currency"], "title": "Transaction", "type": "object"}',
    }

    async function handleInput(prompt) {
        try {
            await pipe.startChat(sysMessage);
            config.structured_output_config = new StructuredOutputConfig({
                json_schema: JSON.stringify(z.toJSONSchema(ItemQuantitiesSchema))
            });
            config.do_sample = false;

            const json_response = await pipe.generate(prompt, config);
            const res = JSON.parse(json_response);
            await pipe.finishChat();
            console.log(`Generated JSON with item quantities: ${JSON.stringify(res)}`);

            config.do_sample = true;
            config.temperature = 0.8;

            await pipe.startChat(sysMessageForItems);
            let generateHasRun = false;

            for (const [item, quantity] of Object.entries(res)) {
                const schema = itemsMap[item];
                if (!schema) continue;
                config.structured_output_config = new StructuredOutputConfig({
                    json_schema: JSON.stringify(z.toJSONSchema(schema))
                    // json_schema: pyJsonSchema[item]
                });
                for (let i = 0; i < quantity; i++) {
                    generateHasRun = true;
                    const jsonStr = await pipe.generate(prompt, config);
                    console.log(JSON.parse(jsonStr));
                }
            }

            await pipe.finishChat();

            if (!generateHasRun) {
                console.log("No items generated. Please try again with a different request.");
            }
        } catch (error) {
            console.error("An error occurred:", error);
        }

    }

    // await handleInput("Generate jsons for 2 persons, 3 cars and 1 transaction.");

    rl.on('line', handleInput);
}

main();

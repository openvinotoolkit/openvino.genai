// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import readline from 'readline';
import { z } from 'zod';
import { LLMPipeline, StreamingStatus, StructuredOutputConfig } from 'openvino-genai-node';
import { serialize_json } from './helper.js';

const getWeatherTool = {
    name: "get_weather",
    schema: z.object({
        city: z.string().describe("City name"),
        country: z.string().describe("Country name"),
        date: z.string().regex(/2\d{3}-[0-1]\d-[0-3]\d/).describe("Date in YYYY-MM-DD format")
    }),
};

const getCurrencyExchangeTool = {
    name: "get_currency_exchange",
    schema: z.object({
        from_currency: z.string().describe("Currency to convert from"),
        to_currency: z.string().describe("Currency to convert to"),
        amount: z.number().describe("Amount to convert")
    }),
};

const tools = [getWeatherTool, getCurrencyExchangeTool];

const sysMessage = "You are a helpful assistant that can provide weather information and currency exchange rates. "
    + `Today is ${new Date().toISOString().split('T')[0]}. `
    + "You can respond in natural language, always start your answer with appropriate greeting, "
    + "If you need additional information to respond you can request it by calling particular tool with structured JSON. "
    + `You can use the following tools:
${tools.map(tool => `<function_name=\"${tool.name}\">, arguments=${serialize_json(tool.schema.keyof().options)}`).join('\n')}
Please, only use the following format for tool calling in your responses:
<function=\"function_name\">{"argument1": "value1", ...}</function>
Use the tool name and arguments as defined in the tool schema.
If you don't know the answer, just say that you don't know, but try to call the tool if it helps to answer the question.`;

const functionPattern = /<function="([^"]+)">(.*?)<\/function>/gs;

/** Parse the tool response from the model output.
    The response should be in the format:
    <function="function_name">{"argument1": "value1", ...}</function>
    */
function parseToolsFromResponse(response) {
    const matches = response.matchAll(functionPattern);
    return Array.from(matches).map(match => {
        const toolName = match[1];
        const args = JSON.parse(match[2]);
        return { toolName, args };
    });
}

function streamer(subword) {
    process.stdout.write(subword);
    return StreamingStatus.RUNNING;
}

async function main() {
    const defaultPrompt = "What is the weather in London today and in Paris yesterday, and how many pounds can I get for 100 euros?";

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const modelDir = process.argv[2];
    if (!modelDir) {
        console.error('Please provide the path to the model directory as the first argument.');
        process.exit(1);
    }

    const prompt = process.argv[3] || defaultPrompt;

    rl.close();

    const device = "CPU"; // GPU can be used as well
    const pipe = await LLMPipeline(modelDir, device);

    console.log(`User prompt: ${prompt} `);

    for (const useStructuralTags of [false, true]) {
        console.log("=".repeat(80));
        console.log(`${useStructuralTags ? "Using structural tags" : "Using no structural tags"} `.padStart(40).padEnd(80));
        console.log("=".repeat(80));

        const generation_config = {};
        generation_config.return_decoded_results = true;
        generation_config.max_new_tokens = 300;

        await pipe.startChat(sysMessage);
        if (useStructuralTags) {
            generation_config.structured_output_config = {
                structural_tags_config: StructuredOutputConfig.TriggeredTags({
                    tags: tools.map(tool => StructuredOutputConfig.Tag({
                        begin: `<function=\"${tool.name}\">`,
                        content: StructuredOutputConfig.JSONSchema(serialize_json(z.toJSONSchema(tool.schema))),
                        end: "</function>"
                    })),
                    triggers: ["<function="]
                })
            };
        };
        generation_config.do_sample = true;

        const response = await pipe.generate(prompt, generation_config, streamer);
        await pipe.finishChat();
        console.log("\n" + "-".repeat(80));

        console.log("Correct tool calls by the model:");
        console.log(parseToolsFromResponse(response.toString()));
    }
}

main();

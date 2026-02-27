// Copyright(C) 2025 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0

import * as https from 'https';
import { LLMPipeline, StreamingStatus } from "openvino-genai-node";
import { serialize_json } from './helper.js';

const llmConfig = {
    'max_new_tokens': 256,
    'return_decoded_results': true,
}

const TOOL_DESC = `{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}`
"get_weather: Call this tool to interact with the get weather API. What is the get weather API useful for? Get the current weather in a given city name. Parameters: [object Object]"
const PROMPT_REACT = `Answer the following questions as best as you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}`;

const tools = [
    {
        "name_for_human": "get weather",
        "name_for_model": "get_weather",
        "description_for_model": "Get the current weather in a given city name.",
        "parameters": [
            {
                "name": "city_name",
                "description": "City name",
                "required": true,
                "schema": { "type": "string" },
            }
        ],
    },
    {
        "name_for_human": "generate image",
        "name_for_model": "generate_image",
        "description_for_model": "AI painting (image generation) service, input text description, and return the image URL drawn based on text information.",
        "parameters": [
            {
                "name": "prompt",
                "description": "describe the image",
                "required": true,
                "schema": { "type": "string" },
            }
        ],
    },
]

function formatTemplate(template, values) {
    const result = template.replace(/{(\w+)}/g, (_, key) => {
        let value = values[key] || '';
        if (typeof value !== "string") {
            value = serialize_json(value);
        }
        return value;
    });
    return result;
}

function buildInputText(tokenizer, chatHistory, listOfToolInfo) {
    const toolsTextList = [];
    for (const toolInfo of listOfToolInfo) {
        let tool = formatTemplate(TOOL_DESC, toolInfo);
        if (toolInfo["args_format"] ?? "json" === "json") {
            tool += " Format the arguments as a JSON object.";
        } else if (toolInfo["args_format"] === "code") {
            tool += " Enclose the code within triple backticks (`) at the beginning and end of the code.";
        } else {
            throw Error(`This args_format: ${args_format} is not supported`);
        }
        toolsTextList.push(tool)
    }
    const toolsText = toolsTextList.join("\n\n");
    const toolsNameText = listOfToolInfo.map(toolInfo => toolInfo["name_for_model"]).join(", ");

    const messages = [{ "role": "system", "content": "You are a helpful assistant." }];
    for (let [query, response] of chatHistory) {
        if (listOfToolInfo) {
            if (chatHistory.length == 1) {
                query = formatTemplate(PROMPT_REACT, {
                    'tools_text': toolsText,
                    'tools_name_text': toolsNameText,
                    query,
                });
            }
        }
        if (query) messages.push({ "role": "user", "content": query })
        if (response) messages.push({ "role": "assistant", "content": response })
    }

    const prompt = tokenizer.applyChatTemplate(messages, true);

    return prompt;
}

function parseFirstToolCall(text) {
    let resultText = text;
    let toolName = "", toolArgs = "";
    const i = resultText.indexOf("\nAction:");
    const j = resultText.indexOf("\nAction Input:");
    let k = resultText.indexOf("\nObservation:");

    if (0 <= i < j) { // If the text has `Action` and `Action input`,
        if (k < j) { // but does not contain `Observation`,
            // then it is likely that `Observation` is omitted by the LLM,
            // because the output text may have discarded the stop word.
            resultText = resultText.trimEnd() + "\nObservation:" // Add it back.
        }
        k = resultText.indexOf("\nObservation:");
        toolName = resultText.slice(i + "\nAction:".length, j).trim();
        toolArgs = resultText.slice(j + "\nAction Input:".length, k).trim();
        resultText = resultText.slice(0, k);
    }
    return [toolName, toolArgs, resultText];
}

async function callTool(toolName, toolArgs) {
    if (toolName === "get_weather") {
        const cityName = JSON.parse(toolArgs)["city_name"];
        const keySelection = {
            "current_condition": [
                "temp_C",
                "FeelsLikeC",
                "humidity",
                "weatherDesc",
                "observation_time",
            ],
        };
        const response = new Promise((resolve, reject) => {
            https.get(`https://wttr.in/${cityName}?format=j1`, {}, (res) => {
                let data = '';

                res.on('data', (chunk) => {
                    data += chunk.toString();
                });

                res.on('end', () => {
                    resolve(JSON.parse(data));
                });

                res.on('error', (err) => {
                    reject(err);
                });
            });
        });
        const data = await response;
        const result = {};
        for (const [key, values] of Object.entries(keySelection)) {
            if (data[key] && Array.isArray(data[key]) && data[key][0]) {
                result[key] = {};
                for (const v of values) {
                    result[key][v] = data[key][0][v];
                }
            }
        }
        return serialize_json(result);
    } else if (toolName === "generate_image") {
        toolArgs = toolArgs.replaceAll('(', '').replaceAll(')', '');
        const parsed = JSON.parse(toolArgs);
        const prompt = encodeURIComponent(parsed.prompt);
        return serialize_json({
            "image_url": `https://image.pollinations.ai/prompt/${prompt}`
        });
    } else {
        throw new Error(`Tool ${toolName} is not supported`);
    }
}

async function llmWithTool(llmPipe, prompt, history, listOfToolInfo) {
    const chatHistory = history.map(x => [x.user, x.bot]).concat([[prompt, ""]])
    const tokenizer = llmPipe.getTokenizer();
    const planningPrompt = buildInputText(tokenizer, chatHistory, listOfToolInfo);

    let text = "";
    while (true) {
        // llm pipe output based planningPrompt and the text (previous output)
        // const llmConfig = llmPipe.getGenerationConfig();
        const generationOutput = await llmPipe.generate(
            planningPrompt + text,
            llmConfig,
            streamer,
        );
        // parse the output to get action
        const [action, actionInput, output] = parseFirstToolCall(generationOutput.toString());
        if (action) {
            const observation = await callTool(action, actionInput);
            const observationTxt = `\nObservation: = ${observation}\nThought:`
            console.log(`\n\n- Getting information from the tool API - ${observationTxt} \n`);
            text += output + observationTxt
        } else {
            text += output;
            break;
        }
    }
    return [text, history]
}

async function main() {
    const llmModelPath = process.argv[2];
    const device = 'CPU' // GPU can be used as well
    const llmPipe = await LLMPipeline(llmModelPath, device);

    const message_history = [];
    const query = "get the weather in London, and create a picture of Big Ben based on the weather information";
    const [response, history] = await llmWithTool(llmPipe, query, message_history, tools);
}

function streamer(subword) {
    process.stdout.write(subword);
    return StreamingStatus.RUNNING;
}

main()

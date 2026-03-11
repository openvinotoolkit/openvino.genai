import { z } from 'zod';
import { ChatHistory, LLMPipeline, StructuredOutputConfig as SOC, StreamingStatus } from 'openvino-genai-node';
import { serialize_json, toJSONSchema } from './helper.js';

function streamer(subword) {
    process.stdout.write(subword);
    return StreamingStatus.RUNNING;
}

const bookFlightTicket = {
    name: "book_flight_ticket",
    schema: z.object({
        origin_airport_code: z.string().describe("The name of Departure airport code"),
        destination_airport_code: z.string().describe("The name of Destination airport code"),
        departure_date: z.string().describe("The date of outbound flight"),
        return_date: z.string().describe("The date of return flight"),
    }).describe("booking flights"),
};

const bookHotel = {
    name: "book_hotel",
    schema: z.object({
        destination: z.string().describe("The name of the city"),
        check_in_date: z.string().describe("The date of check in"),
        checkout_date: z.string().describe("The date of check out"),
    }).describe("booking hotel"),
};

// Helper functions
function toolToDict(tool, withDescription = true) {
    const deleteDescription = (ctx) => delete ctx.jsonSchema['description'];
    const jsonSchema = toJSONSchema(
        tool.schema,
        withDescription
            ? undefined
            : { override: deleteDescription }
    );

    return {
        type: "object",
        properties: {
            name: { type: "string", enum: [tool.name] },
            arguments: jsonSchema,
        },
        required: ["name", "arguments"],
    };
}

function toolsToArraySchema(...tools) {
    return serialize_json({
        type: "array",
        items: {
            anyOf: tools.map(tool => toolToDict(tool, false)),
        },
    });
}

class CustomToolCallParser {
    parse(msg) {
        if (!msg.content) {
            msg.content = "";
        }
        const content = msg.content;

        const startTag = "functools";
        const startIndex = content.indexOf(startTag);
        if (startIndex === -1) {
            return;
        }

        const jsonPart = content.slice(startIndex + startTag.length);
        try {
            const toolCalls = JSON.parse(jsonPart);
            msg.tool_calls = toolCalls;
            return;
        } catch {
            return;
        }
    }
}

function printToolCall(answer) {
    for (const toolCall of answer.parsed[0].tool_calls) {
        const args = Object.keys(toolCall["arguments"])
            .map((key) => `${key}="${toolCall["arguments"][key]}"`);
        console.log(`${toolCall["name"]}(${args.join(", ")})`);
    }
}

// System message
let sysMessage = `You are a helpful AI assistant.
You can answer yes or no to questions, or you can choose to call one or more of the provided functions.

Use the following rule to decide when to call a function:
    * if the response can be generated from your internal knowledge, do so, but use only yes or no as the response
    * if you need external information that can be obtained by calling one or more of the provided functions, generate function calls

If you decide to call functions:
    * prefix function calls with functools marker (no closing marker required)
    * all function calls should be generated in a single JSON list formatted as functools[{"name": [function name], "arguments": [function arguments as JSON]}, ...]
    * follow the provided JSON schema. Do not hallucinate arguments or values. Do not blindly copy values from the provided samples
    * respect the argument type formatting. E.g., if the type is number and format is float, write value 7 as 7.0
    * make sure you pick the right functions that match the user intent
`;

async function main() {
    const modelDir = process.argv[2];
    if (!modelDir) {
        console.error("Please provide the path to the model directory as the first argument.");
        process.exit(1);
    }

    const pipe = await LLMPipeline(modelDir, "CPU");
    const tokenizer = await pipe.getTokenizer();
    const chatHistory = new ChatHistory([{ role: "system", content: sysMessage }]);
    const tools = [bookFlightTicket, bookHotel].map((tool) => toolToDict(tool, true));
    chatHistory.setTools(tools);

    const generationConfig = {
        return_decoded_results: true,
        max_new_tokens: 300,
        do_sample: true,
    };

    const userText1 = "Do dolphins have fingers?";
    console.log("User: ", userText1);
    chatHistory.push({ role: "user", content: userText1 });

    // the example grammar works the same as SOC.Regex("yes|no")
    // but the Union grammar is more flexible and can be extended with more options
    const yesOrNo = SOC.Union(SOC.Regex("yes"), SOC.Regex("no"));
    generationConfig.structured_output_config = new SOC({ structural_tags_config: yesOrNo });
    process.stdout.write("Assistant: ");
    const answer1 = await pipe.generate(chatHistory, generationConfig, streamer);
    chatHistory.push({ role: "assistant", content: answer1.texts[0] });
    console.log();

    const userText2 =
        "book flight ticket from Beijing to Paris(using airport code) in 2025-12-04 to 2025-12-10, "
        + "then book hotel from 2025-12-04 to 2025-12-10 in Paris";
    console.log("User: ", userText2);
    chatHistory.push({ role: "user", content: userText2 });

    const startToolCallTag = SOC.ConstString("functools");
    const toolsJson = SOC.JSONSchema(
        toolsToArraySchema(bookFlightTicket, bookHotel)
    );
    const toolCall = SOC.Concat(startToolCallTag, toolsJson);

    generationConfig.structured_output_config.structural_tags_config = toolCall;
    generationConfig.parsers = [new CustomToolCallParser()];

    process.stdout.write("Assistant: ");
    const answer2 = await pipe.generate(chatHistory, generationConfig);
    console.log("\n\nThe following tool calls were generated:")
    printToolCall(answer2)
    console.log();
}

main();

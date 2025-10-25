import { z } from 'zod';
import { LLMPipeline, StructuredOutputConfig as SOC, StreamingStatus } from 'openvino-genai-node';
import { serialize_json } from './helper.js';

function streamer(subword) {
    process.stdout.write(subword);
    return StreamingStatus.RUNNING;
}

const bookingFlightTickets = {
    name: "booking_flight_tickets",
    schema: z.object({
        origin_airport_code: z.string().describe("The name of Departure airport code"),
        destination_airport_code: z.string().describe("The name of Destination airport code"),
        departure_date: z.string().describe("The date of outbound flight"),
        return_date: z.string().describe("The date of return flight"),
    }),
};

const bookingHotels = {
    name: "booking_hotels",
    schema: z.object({
        destination: z.string().describe("The name of the city"),
        check_in_date: z.string().describe("The date of check in"),
        checkout_date: z.string().describe("The date of check out"),
    }),
};

// Helper functions
function toolToDict(tool, withDescription = true) {
    const deleteDescription = (schema) => delete schema.jsonSchema['description'];
    const jsonSchema = z.toJSONSchema(
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

/** Generate part of the system prompt with available tools */
function generateSystemPromptTools(...tools) {
    return `<|tool|>${serialize_json(tools.map(toolToDict))}</|tool|>`;
}

function toolsToArraySchema(...tools) {
    return serialize_json({
        type: "array",
        items: {
            anyOf: tools.map(tool => toolToDict(tool, false)),
        },
    });
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
sysMessage += generateSystemPromptTools(bookingFlightTickets, bookingHotels);

async function main() {
    const modelDir = process.argv[2];
    if (!modelDir) {
        console.error("Please provide the path to the model directory as the first argument.");
        process.exit(1);
    }

    const pipe = await LLMPipeline(modelDir, "CPU");
    const tokenizer = await pipe.getTokenizer();
    const chatHistory = [{ role: "system", content: sysMessage }];

    const generationConfig = {
        return_decoded_results: true,
        max_new_tokens: 300,
        do_sample: true,
    };

    const userText1 = "Do dolphins have fingers?";
    console.log("User: ", userText1);
    chatHistory.push({ role: "user", content: userText1 });
    const modelInput = tokenizer.applyChatTemplate(chatHistory, true);

    // the example grammar works the same as SOC.Regex("yes|no")
    // but the Union grammar is more flexible and can be extended with more options
    const yesOrNo = SOC.Union(SOC.Regex("yes"), SOC.Regex("no"));
    generationConfig.structured_output_config = new SOC({ structural_tags_config: yesOrNo });
    process.stdout.write("Assistant: ");
    const answer = await pipe.generate(modelInput, generationConfig, streamer);
    chatHistory.push({ role: "assistant", content: answer.texts[0] });
    console.log();

    const userText2 =
        "book flight ticket from Beijing to Paris(using airport code) in 2025-12-04 to 2025-12-10 , "
        + "then book hotel from 2025-12-04 to 2025-12-10 in Paris";
    console.log("User: ", userText2);
    chatHistory.push({ role: "user", content: userText2 });
    const modelInput2 = tokenizer.applyChatTemplate(chatHistory, true);

    const startToolCallTag = SOC.ConstString("functools");
    const toolsJson = SOC.JSONSchema(
        toolsToArraySchema(bookingFlightTickets, bookingHotels)
    );
    const toolCall = SOC.Concat(startToolCallTag, toolsJson);

    generationConfig.structured_output_config.structural_tags_config = toolCall;

    process.stdout.write("Assistant: ");
    await pipe.generate(modelInput2, generationConfig, streamer);
}

main();

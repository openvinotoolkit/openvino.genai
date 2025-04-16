import { LLMPipeline } from 'openvino-genai-node';

main();

async function main() {
    const modelPath = process.argv[2];
    const prompt = process.argv[3];
    
    if (!modelPath) {
        console.error('Please specify path to model directory\n'
                    + 'Run command must be: `node chat_sample.js *path_to_model_dir* *prompt*`');
        process.exit(1);
    }
    if (!prompt) {
        console.error('Please specify prompt\n'
                      + 'Run command must be: `node chat_sample.js *path_to_model_dir* *prompt*`');
        process.exit(1);
    }

    const device = 'CPU'; // GPU can be used as well
    const pipe = await LLMPipeline(modelPath, device);

    const config = {
        'max_new_tokens': 100,
        'do_sample': true,
        'top_p': 0.9,
        'top_k': 30
    };

    // Since the streamer is set, the results will be printed
    // every time a new token is generated and put into the streamer queue.
    for await (const chunk of pipe.stream(prompt, config)) {
        process.stdout.write(chunk);
    }
}
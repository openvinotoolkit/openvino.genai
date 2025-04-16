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

    const config = { 'max_new_tokens': 100 };
    const result = await pipe.generate(prompt, config);

    console.log(result);
}
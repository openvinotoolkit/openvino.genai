import { LLMPipeline } from 'openvino-genai-node';
import { basename } from 'node:path';

main();

async function main() {
    const modelPath = process.argv[2];
    const prompt = process.argv[3];
    
    if (process.argv.length > 4) {
        console.error(`Run command must be: 'node ${basename(process.argv[1])} *path_to_model_dir* *prompt*'`);
        process.exit(1);
    }
    if (!modelPath) {
        console.error('Please specify path to model directory\n'
                    + `Run command must be: 'node ${basename(process.argv[1])} *path_to_model_dir* *prompt*'`);
        process.exit(1);
    }
    if (!prompt) {
        console.error('Please specify prompt\n'
                      + `Run command must be: 'node ${basename(process.argv[1])} *path_to_model_dir* *prompt*'`);
        process.exit(1);
    }

    const device = 'CPU'; // GPU can be used as well
    const pipe = await LLMPipeline(modelPath, device);

    const config = {
        'max_new_tokens': 100,
        'return_decoded_results': true,
    };
    const result = await pipe.generate(prompt, config);

    console.log(result.toString());
}
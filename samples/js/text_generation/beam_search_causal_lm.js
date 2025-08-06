import { LLMPipeline } from 'openvino-genai-node';
import { basename } from 'node:path';

main();

async function main() {
    const modelPath = process.argv[2];
    const prompts = process.argv.slice(3);
    
    if (!modelPath) {
        console.error('Please specify path to model directory\n'
                    + `Run command must be: 'node ${basename(process.argv[1])} *path_to_model_dir* *prompts*'`);
        process.exit(1);
    }
    if (!prompts) {
        console.error('Please specify prompts\n'
                      + `Run command must be: 'node ${basename(process.argv[1])} *path_to_model_dir* *prompts*'`);
        process.exit(1);
    }

    const device = 'CPU'; // GPU can be used as well
    const pipe = await LLMPipeline(modelPath, device);

    const numBeams = 15;
    const config = {
        'max_new_tokens': 20,
        'num_beam_groups': 3,
        'num_beams': numBeams,
        'diversity_penalty': 1,
        'num_return_sequences': numBeams,
        'return_decoded_results': true,

    };
    const beams = await pipe.generate(prompts, config);
    console.log(beams.toString());
}

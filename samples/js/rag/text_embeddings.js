import { TextEmbeddingPipeline, PoolingType } from 'openvino-genai-node';
import { basename } from 'node:path';

main();

async function main() {
    const modelPath = process.argv[2];
    const texts = process.argv.slice(3);

    if (!modelPath) {
        console.error('Please specify path to model directory\n'
                    + `Run command must be: 'node ${basename(process.argv[1])} *path_to_model_dir* *prompt*'`);
        process.exit(1);
    }
    if (!texts) {
        console.error('Please specify prompt\n'
                      + `Run command must be: 'node ${basename(process.argv[1])} *path_to_model_dir* *prompt*'`);
        process.exit(1);
    }

    const device = 'CPU'; // GPU can be used as well
    const config = {
        'pooling_type': PoolingType.MEAN
    };

    const pipeline = await TextEmbeddingPipeline(modelPath, device, config);

    await pipeline.embedDocuments(texts);
}

import { TextEmbeddingPipeline, PoolingType } from 'openvino-genai-node';
import { basename } from 'node:path';

main();

async function main() {
    const modelPath = process.argv[2];
    const texts = process.argv.slice(3);

    const usageCommand = `Usage: node ${basename(process.argv[1])} <MODEL_DIR> '<TEXT 1>' ['<TEXT 2>' ...]`;
    if (!modelPath) {
        console.error('Please specify path to model directory');
        console.error(usageCommand);
        process.exit(1);
    }
    if (!texts.length) {
        console.error('Please specify prompt');
        console.error(usageCommand);
        process.exit(1);
    }

    const device = 'CPU'; // GPU can be used as well
    const config = {
        'pooling_type': PoolingType.MEAN
    };

    const pipeline = await TextEmbeddingPipeline(modelPath, device, config);

    await pipeline.embedDocuments(texts);
}

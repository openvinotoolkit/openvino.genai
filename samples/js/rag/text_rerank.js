// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { TextRerankPipeline } from 'openvino-genai-node';
import { basename } from 'node:path';

main();

async function main() {
    const modelPath = process.argv[2];
    const query = process.argv[3];
    const documents = process.argv.slice(4);

    const usageCommand = `Usage: node ${basename(process.argv[1])} <MODEL_DIR> "<QUERY>" "<TEXT 1>" ["<TEXT 2>" ...]`;
    if (!modelPath) {
        console.error('Please specify path to model directory');
        console.error(usageCommand);
        process.exit(1);
    }
    if (!query) {
        console.error('Please specify query');
        console.error(usageCommand);
        process.exit(1);
    }
    if (!documents.length) {
        console.error('Please specify at least one document');
        console.error(usageCommand);
        process.exit(1);
    }

    const device = 'CPU'; // GPU can be used as well
    const config = { top_n: 3 };

    const pipeline = await TextRerankPipeline(modelPath, { device, config });

    const rerankResult = await pipeline.rerank(query, documents);

    console.log('Reranked documents:');
    for (const [index, score] of rerankResult) {
        console.log(`Document ${index} (score: ${score.toFixed(4)}): ${documents[index]}`);
    }
}

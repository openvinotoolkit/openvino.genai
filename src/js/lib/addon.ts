import { createRequire } from 'module';
import { platform } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import {
  EmbeddingResult,
  EmbeddingResults
} from './utils.js';

export interface TextEmbeddingPipelineWrapper {
  new (): TextEmbeddingPipelineWrapper;
  init(
    modelPath: string,
    device: string,
    config: object,
    ovProperties: object,
    callback: (err: NodeJS.ErrnoException | null) => void,
  ): void;
  embedQuery(
    text: string,
    callback: (err: NodeJS.ErrnoException | null, value: EmbeddingResult) => void,
  ): void;
  embedDocuments(
    documents: string[],
    callback: (err: NodeJS.ErrnoException | null, value: EmbeddingResults) => void,
  ): void;
  embedQuerySync(text: string): EmbeddingResult;
  embedDocumentsSync(documents: string[]): EmbeddingResults;
}

interface OpenVINOGenAIAddon {
  TextEmbeddingPipeline: TextEmbeddingPipelineWrapper,
  LLMPipeline: any,
}

// We need to use delayed import to get an updated Path if required
function getGenAIAddon(): OpenVINOGenAIAddon {
  const require = createRequire(import.meta.url);
  const ovPath = require.resolve('openvino-node');
  if (platform() == 'win32') {
    // Find the openvino binaries that are required for openvino-genai-node
    const pathToOpenVino = join(dirname(ovPath), '../bin');
    if (!process.env.PATH?.includes('openvino-node')) {
      process.env.PATH += ';' + resolve(pathToOpenVino);
    }
  }

  return require('../bin/genai_node_addon.node');
}

export default getGenAIAddon();

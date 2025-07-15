import { LLMPipeline as LLM } from './pipelines/llmPipeline.js';
import {
  TextEmbeddingPipeline as Embedding,
} from './pipelines/textEmbeddingPipeline.js';

class PipelineFactory {
  static async LLMPipeline(modelPath: string, device = 'CPU') {
    const pipeline = new LLM(modelPath, device);
    await pipeline.init();

    return pipeline;
  }
  static async TextEmbeddingPipeline(modelPath: string, device = 'CPU') {
    const pipeline = new Embedding(modelPath, device);
    await pipeline.init();

    return pipeline;
  }
}

export const {LLMPipeline, TextEmbeddingPipeline} = PipelineFactory;
export * from './utils.js';
export * from './addon.js';

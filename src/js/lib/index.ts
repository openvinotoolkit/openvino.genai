import { LLMPipeline as LLM } from './pipelines/llmPipeline.js';

class PipelineFactory {
  static async LLMPipeline(modelPath: string, device = 'CPU') {
    const pipeline = new LLM(modelPath, device);
    await pipeline.init();

    return pipeline;
  }
}

export const {LLMPipeline} = PipelineFactory;

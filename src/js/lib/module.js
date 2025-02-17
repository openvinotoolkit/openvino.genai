import { LLMPipeline as LLM } from "./pipelines/llmPipeline";

class PipelineFactory {
  static async LLMPipeline(modelPath, device = 'CPU') {
    const pipeline = new LLM(modelPath, device);
    await pipeline.init();

    return pipeline;
  }
}

export const LLMPipeline = PipelineFactory.LLMPipeline;

import { LLMPipeline as LLM } from "./pipelines/llmPipeline.js";
import { TextEmbeddingPipeline as Embedding } from "./pipelines/textEmbeddingPipeline.js";
import { SchedulerConfig } from "./utils.js";

class PipelineFactory {
  static async LLMPipeline(modelPath: string, device: string): Promise<any>;
  static async LLMPipeline(
    modelPath: string,
    device: string,
    { schedulerConfig }: { schedulerConfig: SchedulerConfig },
  ): Promise<any>;
  static async LLMPipeline(modelPath: string, device: string = "CPU", properties?: object) {
    const pipeline = new LLM(modelPath, device, properties || {});
    await pipeline.init();
    return pipeline;
  }
  static async TextEmbeddingPipeline(modelPath: string, device = "CPU") {
    const pipeline = new Embedding(modelPath, device);
    await pipeline.init();

    return pipeline;
  }
}

export const { LLMPipeline, TextEmbeddingPipeline } = PipelineFactory;
export { DecodedResults } from "./pipelines/llmPipeline.js";
export * from "./utils.js";
export * from "./addon.js";

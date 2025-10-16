import util from "node:util";
import addon, {
  TextEmbeddingPipelineWrapper,
  EmbeddingResult,
  EmbeddingResults,
  TextEmbeddingConfig,
} from "../addon.js";

export class TextEmbeddingPipeline {
  modelPath: string;
  device: string;
  config: TextEmbeddingConfig;
  ovProperties: object;
  pipeline: TextEmbeddingPipelineWrapper | null = null;
  isInitialized = false;

  constructor(
    modelPath: string,
    device: string,
    config: TextEmbeddingConfig,
    ovProperties?: object,
  ) {
    this.modelPath = modelPath;
    this.device = device;
    this.config = config;
    this.ovProperties = ovProperties || {};
  }

  async init() {
    if (this.pipeline) throw new Error("TextEmbeddingPipeline is already initialized");

    this.pipeline = new addon.TextEmbeddingPipeline();

    const initPromise = util.promisify(this.pipeline.init.bind(this.pipeline));
    await initPromise(this.modelPath, this.device, this.config, this.ovProperties);

    return;
  }

  embedDocumentsSync(texts: string[]): EmbeddingResults {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");

    return this.pipeline.embedDocumentsSync(texts);
  }

  embedQuerySync(text: string): EmbeddingResult {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");

    return this.pipeline.embedQuerySync(text);
  }

  async embedDocuments(documents: string[]): Promise<EmbeddingResults> {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    const embedDocuments = util.promisify(this.pipeline.embedDocuments.bind(this.pipeline));
    const result = await embedDocuments(documents);

    return result;
  }

  async embedQuery(text: string): Promise<EmbeddingResult> {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    const embedQuery = util.promisify(this.pipeline.embedQuery.bind(this.pipeline));
    const result = await embedQuery(text);

    return result;
  }
}

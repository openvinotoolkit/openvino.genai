import { createRequire } from "module";
import { platform } from "node:os";
import { join, dirname, resolve } from "node:path";

export type EmbeddingResult = Float32Array | Int8Array | Uint8Array;
export type EmbeddingResults = Float32Array[] | Int8Array[] | Uint8Array[];
/**
 * Pooling strategy
 */
export enum PoolingType {
  /** First token embeddings */
  CLS,
  /** The average of all token embeddings */
  MEAN,
}
export type TextEmbeddingConfig = {
  /** Maximum length of tokens passed to the embedding model */
  max_length?: number;
  /** If 'true', model input tensors are padded to the maximum length */
  pad_to_max_length?: boolean;
  /** Side to use for padding "left" or "right" */
  padding_side?: "left" | "right";
  /**
   * Batch size of embedding model.
   * Useful for database population. If set, the pipeline will fix model shape for inference optimization.
   * Number of documents passed to pipeline should be equal to batch_size.
   * For query embeddings, batch_size should be set to 1 or not set.
   */
  batch_size?: number;
  /** Pooling strategy applied to model output tensor */
  pooling_type?: PoolingType;
  /** If 'true', L2 normalization is applied to embeddings */
  normalize?: boolean;
  /** Instruction to use for embedding a query */
  query_instruction?: string;
  /** Instruction to use for embedding a document */
  embed_instruction?: string;
};

export interface TextEmbeddingPipelineWrapper {
  new (): TextEmbeddingPipelineWrapper;
  init(
    modelPath: string,
    device: string,
    config: TextEmbeddingConfig,
    ovProperties: object,
    callback: (err: Error | null) => void,
  ): void;
  embedQuery(text: string, callback: (err: Error | null, value: EmbeddingResult) => void): void;
  embedDocuments(
    documents: string[],
    callback: (err: Error | null, value: EmbeddingResults) => void,
  ): void;
  embedQuerySync(text: string): EmbeddingResult;
  embedDocumentsSync(documents: string[]): EmbeddingResults;
}

interface OpenVINOGenAIAddon {
  TextEmbeddingPipeline: TextEmbeddingPipelineWrapper;
  LLMPipeline: any;
}

// We need to use delayed import to get an updated Path if required
function getGenAIAddon(): OpenVINOGenAIAddon {
  const require = createRequire(import.meta.url);
  const ovPath = require.resolve("openvino-node");
  if (platform() == "win32") {
    // Find the openvino binaries that are required for openvino-genai-node
    const pathToOpenVino = join(dirname(ovPath), "../bin");
    if (!process.env.PATH?.includes("openvino-node")) {
      process.env.PATH += ";" + resolve(pathToOpenVino);
    }
  }

  return require("../bin/genai_node_addon.node");
}

export default getGenAIAddon();

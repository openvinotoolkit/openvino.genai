import argparse
import openvino_genai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    args = parser.parse_args()

    device = "CPU"  # GPU can be used as well

    config = openvino_genai.TextEmbeddingPipeline.Config()
    config.pad_to_max_length = True
    config.batch_size = 4
    config.max_length = 128
    config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN

    pipeline = openvino_genai.TextEmbeddingPipeline(args.model_dir, device, config)

    text_embeddings = pipeline.embed_documents(args.texts)
    query_embeddings = pipeline.embed_query("What is the capital of France?")


if "__main__" == __name__:
    main()

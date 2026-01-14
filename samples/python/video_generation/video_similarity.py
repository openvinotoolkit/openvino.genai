# Based on https://huggingface.co/docs/transformers/main/model_doc/xclip#transformers.XCLIPModel.get_video_features
import argparse
import av
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import transformers


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`list[int]`): List of sampled frame indices
    """
    converted_len = int(clip_len * frame_sample_rate)
    # end_idx = np.random.randint(converted_len, seg_len)
    end_idx = seg_len
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def encode_video(path, processor, model):
    container = av.open(path)
    # Sample 8 frames. The paper evaluated the model on 8 and 16 frames.
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)
    with torch.no_grad():
        pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values[0]
        return model(pixel_values).pooler_output.mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left_video")
    parser.add_argument("right_video")
    args = parser.parse_args()

    processor = transformers.AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
    model = transformers.XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32").eval()

    left_embedding = encode_video(args.left_video, processor, model)
    right_embedding = encode_video(args.right_video, processor, model)

    # torch.nn.functional.cosine_similarity(left_embedding, right_embedding, dim=0) doesn't return 1 for identical inputs so use sklearn.
    similarity = cosine_similarity(left_embedding.numpy().reshape(1, -1), right_embedding.numpy().reshape(1, -1))
    print(similarity)


if __name__ == "__main__":
    main()

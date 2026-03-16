# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import cv2


def save_video(filename: str, video_tensor, fps: int = 25):
    batch_size, num_frames, height, width, _ = video_tensor.shape
    video_data = video_tensor.data

    for b in range(batch_size):
        if batch_size == 1:
            output_path = filename
        else:
            base, ext = filename.rsplit(".", 1) if "." in filename else (filename, "avi")
            output_path = f"{base}_b{b}.{ext}"

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for f in range(num_frames):
            frame_bgr = cv2.cvtColor(video_data[b, f], cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        print(f"Wrote {output_path} ({num_frames} frames, {width}x{height} @ {fps} fps)")

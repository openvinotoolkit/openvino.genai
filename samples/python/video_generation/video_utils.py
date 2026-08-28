# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import struct

import cv2
import numpy as np


def _batch_filename(filename: str, batch_size: int, batch_idx: int, default_ext: str) -> str:
    if batch_size == 1:
        return filename
    base, ext = os.path.splitext(filename)
    return f"{base}_b{batch_idx}{ext or '.' + default_ext}"


def save_video(filename: str, video_tensor, fps: int = 25):
    batch_size, num_frames, height, width, _ = video_tensor.shape
    video_data = video_tensor.data

    for b in range(batch_size):
        output_path = _batch_filename(filename, batch_size, b, "avi")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, cv2.CAP_OPENCV_MJPEG, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter failed to open: {output_path}")

        for f in range(num_frames):
            frame_bgr = cv2.cvtColor(video_data[b, f], cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        print(f"Wrote {output_path} ({num_frames} frames, {width}x{height} @ {fps} fps)")


def save_video_with_audio(filename: str, video_tensor, audio_tensor, fps: float, sample_rate: int):
    batch_size, num_frames, height, width, _ = video_tensor.shape
    audio_batch, num_channels, num_samples = audio_tensor.shape
    if batch_size != audio_batch:
        raise RuntimeError("save_video_with_audio(): video and audio batch sizes differ")

    video_data = video_tensor.data
    audio_data = audio_tensor.data
    block_align = num_channels * 2

    for b in range(batch_size):
        output_path = _batch_filename(filename, batch_size, b, "avi")

        # [C, S] -> interleaved [S, C] 16-bit PCM
        pcm = (np.clip(audio_data[b].T, -1.0, 1.0) * 32767.0).astype(np.int16)

        # Interleave one video chunk ("00dc") and one audio chunk ("01wb") per frame.
        movi = bytearray()
        index = []
        max_video_chunk = 0
        max_audio_chunk = 0

        def append_chunk(fourcc: bytes, data: bytes):
            index.append((fourcc, len(movi) + 4, len(data)))
            movi.extend(fourcc)
            movi.extend(struct.pack("<I", len(data)))
            movi.extend(data)
            if len(data) % 2 != 0:
                movi.append(0)

        for f in range(num_frames):
            frame_bgr = cv2.cvtColor(video_data[b, f], cv2.COLOR_RGB2BGR)
            ok, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError(f"Failed to encode video frame for: {output_path}")
            max_video_chunk = max(max_video_chunk, len(jpeg))
            append_chunk(b"00dc", jpeg.tobytes())

            sample_begin, sample_end = f * num_samples // num_frames, (f + 1) * num_samples // num_frames
            audio_bytes = pcm[sample_begin:sample_end].tobytes()
            append_chunk(b"01wb", audio_bytes)
            max_audio_chunk = max(max_audio_chunk, len(audio_bytes))

        avih_size, strh_size, video_strf_size, audio_strf_size = 56, 56, 40, 16
        video_strl_size = 4 + 8 + strh_size + 8 + video_strf_size
        audio_strl_size = 4 + 8 + strh_size + 8 + audio_strf_size
        hdrl_size = 4 + 8 + avih_size + 8 + video_strl_size + 8 + audio_strl_size
        movi_size = 4 + len(movi)
        idx1_size = len(index) * 16
        riff_size = 4 + 8 + hdrl_size + 8 + movi_size + 8 + idx1_size

        header = bytearray()
        header.extend(b"RIFF" + struct.pack("<I", riff_size) + b"AVI ")
        header.extend(b"LIST" + struct.pack("<I", hdrl_size) + b"hdrl")
        header.extend(
            b"avih"
            + struct.pack(
                "<15I",
                avih_size,
                round(1e6 / fps),  # dwMicroSecPerFrame
                0,  # dwMaxBytesPerSec
                0,  # dwPaddingGranularity
                0x110,  # AVIF_HASINDEX | AVIF_ISINTERLEAVED
                num_frames,  # dwTotalFrames
                0,  # dwInitialFrames
                2,  # dwStreams
                max_video_chunk,  # dwSuggestedBufferSize
                width,
                height,
                0,
                0,
                0,
                0,  # dwReserved
            )
        )
        header.extend(b"LIST" + struct.pack("<I", video_strl_size) + b"strl")
        header.extend(
            b"strh"
            + struct.pack(
                "<I4s4sI2H8I4H",
                strh_size,
                b"vids",
                b"MJPG",
                0,  # dwFlags
                0,
                0,  # wPriority, wLanguage
                0,  # dwInitialFrames
                1000,  # dwScale
                round(fps * 1000),  # dwRate
                0,  # dwStart
                num_frames,  # dwLength
                max_video_chunk,  # dwSuggestedBufferSize
                0xFFFFFFFF,  # dwQuality
                0,  # dwSampleSize
                0,
                0,
                width,
                height,  # rcFrame
            )
        )
        header.extend(
            b"strf"
            + struct.pack(
                "<4I2H4s5I",
                video_strf_size,  # BITMAPINFOHEADER
                video_strf_size,  # biSize
                width,
                height,
                1,  # biPlanes
                24,  # biBitCount
                b"MJPG",  # biCompression
                width * height * 3,  # biSizeImage
                0,
                0,
                0,
                0,  # biXPelsPerMeter .. biClrImportant
            )
        )
        header.extend(b"LIST" + struct.pack("<I", audio_strl_size) + b"strl")
        header.extend(
            b"strh"
            + struct.pack(
                "<I4sII2H8I4H",
                strh_size,
                b"auds",
                0,  # fccHandler
                0,  # dwFlags
                0,
                0,  # wPriority, wLanguage
                0,  # dwInitialFrames
                block_align,  # dwScale
                sample_rate * block_align,  # dwRate
                0,  # dwStart
                num_samples,  # dwLength
                max_audio_chunk,  # dwSuggestedBufferSize
                0xFFFFFFFF,  # dwQuality
                block_align,  # dwSampleSize
                0,
                0,
                0,
                0,  # rcFrame
            )
        )
        header.extend(
            b"strf"
            + struct.pack(
                "<I2H2I2H",
                audio_strf_size,  # PCMWAVEFORMAT
                1,  # WAVE_FORMAT_PCM
                num_channels,
                sample_rate,
                sample_rate * block_align,
                block_align,
                16,  # wBitsPerSample
            )
        )
        header.extend(b"LIST" + struct.pack("<I", movi_size) + b"movi")

        idx1 = bytearray(b"idx1" + struct.pack("<I", idx1_size))
        for fourcc, offset, size in index:
            idx1.extend(fourcc + struct.pack("<3I", 0x10, offset, size))  # AVIIF_KEYFRAME

        with open(output_path, "wb") as file:
            file.write(header)
            file.write(movi)
            file.write(idx1)

        print(
            f"Wrote {output_path} ({num_frames} frames, {width}x{height} @ {fps} fps, "
            f"audio {num_samples} samples @ {sample_rate} Hz)"
        )

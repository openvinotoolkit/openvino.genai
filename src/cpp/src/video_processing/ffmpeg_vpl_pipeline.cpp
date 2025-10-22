// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_processing/ffmpeg_vpl_pipeline.hpp"
#include <iostream>
#include <fstream>
#include <cstring>

#ifdef ENABLE_FFMPEG_VPL
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}
#include <vpl/mfxvideo.h>
#include <vpl/mfxdefs.h>
#endif

namespace ov {
namespace genai {

class FFmpegVPLPipeline::Impl {
public:
    Impl(const VideoProcessingConfig& config) : m_config(config) {
#ifdef ENABLE_FFMPEG_VPL
        m_initialized = false;
#else
        throw std::runtime_error("FFmpeg and VPL support not enabled. Please build with ENABLE_FFMPEG_VPL=ON");
#endif
    }
    
    ~Impl() {
#ifdef ENABLE_FFMPEG_VPL
        cleanup();
#endif
    }
    
    bool process() {
#ifdef ENABLE_FFMPEG_VPL
        if (!initialize_ffmpeg()) {
            std::cerr << "Failed to initialize FFmpeg decoder" << std::endl;
            return false;
        }
        
        if (!initialize_vpl()) {
            std::cerr << "Failed to initialize VPL" << std::endl;
            return false;
        }
        
        // Process frames
        while (true) {
            AVFrame* decoded_frame = decode_frame();
            if (!decoded_frame) {
                break; // No more frames
            }
            
            // Process with VPL
            if (!process_frame_with_vpl(decoded_frame)) {
                av_frame_free(&decoded_frame);
                std::cerr << "Failed to process frame with VPL" << std::endl;
                return false;
            }
            
            av_frame_free(&decoded_frame);
        }
        
        return true;
#else
        std::cerr << "FFmpeg and VPL support not enabled" << std::endl;
        return false;
#endif
    }
    
    bool get_next_frame(VideoFrame& frame) {
#ifdef ENABLE_FFMPEG_VPL
        if (!m_initialized) {
            if (!initialize_ffmpeg()) {
                return false;
            }
            if (!initialize_vpl()) {
                return false;
            }
            m_initialized = true;
        }
        
        AVFrame* decoded_frame = decode_frame();
        if (!decoded_frame) {
            return false;
        }
        
        // Convert frame to VideoFrame structure
        frame.width = decoded_frame->width;
        frame.height = decoded_frame->height;
        frame.timestamp = decoded_frame->pts;
        frame.format = 0; // NV12 by default
        
        // Calculate frame size
        int frame_size = av_image_get_buffer_size(
            (AVPixelFormat)decoded_frame->format,
            decoded_frame->width,
            decoded_frame->height,
            1
        );
        
        frame.data.resize(frame_size);
        
        av_image_copy_to_buffer(
            frame.data.data(),
            frame_size,
            decoded_frame->data,
            decoded_frame->linesize,
            (AVPixelFormat)decoded_frame->format,
            decoded_frame->width,
            decoded_frame->height,
            1
        );
        
        av_frame_free(&decoded_frame);
        return true;
#else
        return false;
#endif
    }
    
    std::string get_metadata() const {
#ifdef ENABLE_FFMPEG_VPL
        if (!m_format_ctx) {
            return "No metadata available";
        }
        
        std::string metadata;
        metadata += "Duration: " + std::to_string(m_format_ctx->duration / AV_TIME_BASE) + "s\n";
        
        if (m_video_stream) {
            metadata += "Width: " + std::to_string(m_codec_ctx->width) + "\n";
            metadata += "Height: " + std::to_string(m_codec_ctx->height) + "\n";
            AVRational fps = m_video_stream->avg_frame_rate;
            metadata += "FPS: " + std::to_string(fps.num / (double)fps.den) + "\n";
        }
        
        return metadata;
#else
        return "FFmpeg support not enabled";
#endif
    }

private:
#ifdef ENABLE_FFMPEG_VPL
    bool initialize_ffmpeg() {
        // Open input file
        if (avformat_open_input(&m_format_ctx, m_config.input_file.c_str(), nullptr, nullptr) < 0) {
            std::cerr << "Could not open input file: " << m_config.input_file << std::endl;
            return false;
        }
        
        // Retrieve stream information
        if (avformat_find_stream_info(m_format_ctx, nullptr) < 0) {
            std::cerr << "Could not find stream information" << std::endl;
            return false;
        }
        
        // Find the video stream
        m_video_stream_idx = -1;
        for (unsigned int i = 0; i < m_format_ctx->nb_streams; i++) {
            if (m_format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                m_video_stream_idx = i;
                m_video_stream = m_format_ctx->streams[i];
                break;
            }
        }
        
        if (m_video_stream_idx == -1) {
            std::cerr << "Could not find video stream" << std::endl;
            return false;
        }
        
        // Get decoder
        const AVCodec* decoder = avcodec_find_decoder(m_video_stream->codecpar->codec_id);
        if (!decoder) {
            std::cerr << "Codec not found" << std::endl;
            return false;
        }
        
        // Allocate codec context
        m_codec_ctx = avcodec_alloc_context3(decoder);
        if (!m_codec_ctx) {
            std::cerr << "Could not allocate codec context" << std::endl;
            return false;
        }
        
        // Copy codec parameters
        if (avcodec_parameters_to_context(m_codec_ctx, m_video_stream->codecpar) < 0) {
            std::cerr << "Could not copy codec parameters" << std::endl;
            return false;
        }
        
        // Open codec
        if (avcodec_open2(m_codec_ctx, decoder, nullptr) < 0) {
            std::cerr << "Could not open codec" << std::endl;
            return false;
        }
        
        m_packet = av_packet_alloc();
        m_frame = av_frame_alloc();
        
        return true;
    }
    
    bool initialize_vpl() {
        // Initialize VPL session
        mfxVersion version = {0, 1};
        mfxStatus sts = MFXInit(MFX_IMPL_AUTO_ANY, &version, &m_vpl_session);
        if (sts != MFX_ERR_NONE) {
            std::cerr << "Failed to initialize VPL session" << std::endl;
            return false;
        }
        
        // Configure VPP parameters
        memset(&m_vpp_params, 0, sizeof(m_vpp_params));
        m_vpp_params.vpp.In.FourCC = MFX_FOURCC_NV12;
        m_vpp_params.vpp.In.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
        m_vpp_params.vpp.In.Width = m_codec_ctx->width;
        m_vpp_params.vpp.In.Height = m_codec_ctx->height;
        m_vpp_params.vpp.In.CropW = m_codec_ctx->width;
        m_vpp_params.vpp.In.CropH = m_codec_ctx->height;
        m_vpp_params.vpp.In.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
        m_vpp_params.vpp.In.FrameRateExtN = m_video_stream->avg_frame_rate.num;
        m_vpp_params.vpp.In.FrameRateExtD = m_video_stream->avg_frame_rate.den;
        
        // Output parameters
        int out_width = m_config.target_width > 0 ? m_config.target_width : m_codec_ctx->width;
        int out_height = m_config.target_height > 0 ? m_config.target_height : m_codec_ctx->height;
        
        m_vpp_params.vpp.Out.FourCC = m_config.output_format == 1 ? MFX_FOURCC_RGB4 : MFX_FOURCC_NV12;
        m_vpp_params.vpp.Out.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
        m_vpp_params.vpp.Out.Width = out_width;
        m_vpp_params.vpp.Out.Height = out_height;
        m_vpp_params.vpp.Out.CropW = out_width;
        m_vpp_params.vpp.Out.CropH = out_height;
        m_vpp_params.vpp.Out.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
        m_vpp_params.vpp.Out.FrameRateExtN = m_video_stream->avg_frame_rate.num;
        m_vpp_params.vpp.Out.FrameRateExtD = m_video_stream->avg_frame_rate.den;
        
        m_vpp_params.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
        
        // Enable filters if requested
        if (m_config.denoise || m_config.detail_enhance) {
            m_vpp_params.ExtParam = new mfxExtBuffer*[2];
            m_vpp_params.NumExtParam = 0;
            
            if (m_config.denoise) {
                // Add denoise filter configuration
                m_vpp_params.NumExtParam++;
            }
            if (m_config.detail_enhance) {
                // Add detail enhancement filter configuration
                m_vpp_params.NumExtParam++;
            }
        }
        
        // Initialize VPP
        sts = MFXVideoVPP_Init(m_vpl_session, &m_vpp_params);
        if (sts != MFX_ERR_NONE) {
            std::cerr << "Failed to initialize VPP" << std::endl;
            return false;
        }
        
        // Open output file if specified
        if (!m_config.output_file.empty()) {
            m_output_file.open(m_config.output_file, std::ios::binary);
            if (!m_output_file.is_open()) {
                std::cerr << "Failed to open output file" << std::endl;
                return false;
            }
        }
        
        return true;
    }
    
    AVFrame* decode_frame() {
        while (av_read_frame(m_format_ctx, m_packet) >= 0) {
            if (m_packet->stream_index == m_video_stream_idx) {
                int ret = avcodec_send_packet(m_codec_ctx, m_packet);
                av_packet_unref(m_packet);
                
                if (ret < 0) {
                    std::cerr << "Error sending packet for decoding" << std::endl;
                    return nullptr;
                }
                
                ret = avcodec_receive_frame(m_codec_ctx, m_frame);
                if (ret == 0) {
                    AVFrame* frame_copy = av_frame_clone(m_frame);
                    return frame_copy;
                } else if (ret == AVERROR(EAGAIN)) {
                    continue;
                } else {
                    std::cerr << "Error receiving frame" << std::endl;
                    return nullptr;
                }
            }
            av_packet_unref(m_packet);
        }
        
        // Flush decoder
        avcodec_send_packet(m_codec_ctx, nullptr);
        int ret = avcodec_receive_frame(m_codec_ctx, m_frame);
        if (ret == 0) {
            AVFrame* frame_copy = av_frame_clone(m_frame);
            return frame_copy;
        }
        
        return nullptr;
    }
    
    bool process_frame_with_vpl(AVFrame* frame) {
        // For simplicity, we'll write the processed frame to output
        // In a real implementation, we would convert the AVFrame to mfxFrameSurface1
        // and process it through VPP
        
        if (m_output_file.is_open()) {
            // Write frame data to output file
            int data_size = av_image_get_buffer_size(
                (AVPixelFormat)frame->format,
                frame->width,
                frame->height,
                1
            );
            
            std::vector<uint8_t> buffer(data_size);
            av_image_copy_to_buffer(
                buffer.data(),
                data_size,
                frame->data,
                frame->linesize,
                (AVPixelFormat)frame->format,
                frame->width,
                frame->height,
                1
            );
            
            m_output_file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
        }
        
        return true;
    }
    
    void cleanup() {
        if (m_vpl_session) {
            MFXVideoVPP_Close(m_vpl_session);
            MFXClose(m_vpl_session);
            m_vpl_session = nullptr;
        }
        
        if (m_vpp_params.ExtParam) {
            delete[] m_vpp_params.ExtParam;
            m_vpp_params.ExtParam = nullptr;
        }
        
        if (m_output_file.is_open()) {
            m_output_file.close();
        }
        
        if (m_frame) {
            av_frame_free(&m_frame);
        }
        
        if (m_packet) {
            av_packet_free(&m_packet);
        }
        
        if (m_codec_ctx) {
            avcodec_free_context(&m_codec_ctx);
        }
        
        if (m_format_ctx) {
            avformat_close_input(&m_format_ctx);
        }
    }
    
    VideoProcessingConfig m_config;
    bool m_initialized = false;
    
    // FFmpeg members
    AVFormatContext* m_format_ctx = nullptr;
    AVCodecContext* m_codec_ctx = nullptr;
    AVStream* m_video_stream = nullptr;
    AVPacket* m_packet = nullptr;
    AVFrame* m_frame = nullptr;
    int m_video_stream_idx = -1;
    
    // VPL members
    mfxSession m_vpl_session = nullptr;
    mfxVideoParam m_vpp_params;
    
    // Output file
    std::ofstream m_output_file;
#endif
};

FFmpegVPLPipeline::FFmpegVPLPipeline(const VideoProcessingConfig& config)
    : m_impl(std::make_unique<Impl>(config)) {}

FFmpegVPLPipeline::~FFmpegVPLPipeline() = default;

bool FFmpegVPLPipeline::process() {
    return m_impl->process();
}

bool FFmpegVPLPipeline::get_next_frame(VideoFrame& frame) {
    return m_impl->get_next_frame(frame);
}

std::string FFmpegVPLPipeline::get_metadata() const {
    return m_impl->get_metadata();
}

} // namespace genai
} // namespace ov

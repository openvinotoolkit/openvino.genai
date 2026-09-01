// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

namespace ov::genai {

/// Trims already-decoded audio from the front of `audio_accum` in place, once it exceeds
/// `window_chunk_num` chunks, keeping `window_rollback_chunk_num` chunks of rollback margin
/// plus the newest, not-yet-decoded chunk that was just merged in. No-op when
/// `window_chunk_num == 0` or `audio_accum` already fits the window.
///
/// Call this AFTER merging the newest chunk into `audio_accum` (matching the SGLang
/// reference's ordering), not before — trimming first would double-count the "room for the
/// newest chunk" margin once in the keep target and again via the append that follows.
///
/// `already_inferred_samples` is `audio_accum`'s size immediately before this call's merge —
/// i.e. how much of it has definitely already been through a decode pass. The drop is capped
/// at this value so a burst larger than one chunk (e.g. a caller pushing several chunks worth
/// of audio in a single push_chunk() call) can never cause never-decoded audio to be dropped.
///
/// Returns the number of samples actually dropped from the front (0 if no-op). Callers that
/// track text state keyed to absolute audio position (e.g. which chunk's audio grounds a given
/// piece of committed text) need this to know how far the window's front edge just advanced,
/// so they can evict/reset any state whose grounding audio no longer exists in the window --
/// mirroring SGLang's roll-triggered `start_new_window()` reset rather than a fixed heuristic.
size_t apply_sliding_window_drop(std::vector<float>& audio_accum,
                                  size_t already_inferred_samples,
                                  size_t chunk_size_samples,
                                  size_t window_chunk_num,
                                  size_t window_rollback_chunk_num);

}  // namespace ov::genai

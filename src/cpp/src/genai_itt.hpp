// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#ifdef ENABLE_PROFILING_ITT
#    include <ittnotify.h>
#    include <mutex>
#    include <unordered_map>
#endif

namespace ov::genai::itt {

enum class Domain : std::uint8_t {
    GenAI,
    Metrics,
    LLM,
    VLM,
    Whisper,
    Tokenizer,
    ContinuousBatching
};

enum class MarkerScope : std::uint8_t {
    Global,
    Process,
    Thread,
    Task,
    Marker
};

#ifdef ENABLE_PROFILING_ITT
namespace detail {

inline __itt_scope to_itt_scope(MarkerScope scope) {
    switch (scope) {
    case MarkerScope::Global:
        return __itt_scope_global;
    case MarkerScope::Process:
        return __itt_scope_track_group;
    case MarkerScope::Thread:
        return __itt_scope_track;
    case MarkerScope::Task:
        return __itt_scope_task;
    case MarkerScope::Marker:
    default:
        return __itt_scope_marker;
    }
}

inline __itt_domain* domain_by_name(const char* name) {
    static std::mutex mutex;
    static std::unordered_map<std::string, __itt_domain*> domains;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = domains.find(name);
    if (it != domains.end()) {
        return it->second;
    }

    auto* created = __itt_domain_create(name);
    domains.emplace(name, created);
    return created;
}

inline __itt_string_handle* string_handle(const char* name) {
    static std::mutex mutex;
    static std::unordered_map<std::string, __itt_string_handle*> handles;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = handles.find(name);
    if (it != handles.end()) {
        return it->second;
    }

    auto* created = __itt_string_handle_create(name);
    handles.emplace(name, created);
    return created;
}

inline __itt_id next_id() {
    static std::atomic<std::uint64_t> next{1};
    __itt_id id = __itt_null;
    id.d1 = next.fetch_add(1, std::memory_order_relaxed);
    return id;
}

inline std::vector<__itt_id>& active_entities() {
    static thread_local std::vector<__itt_id> entities;
    return entities;
}

inline __itt_id current_parent_id() {
    const auto& entities = active_entities();
    return entities.empty() ? __itt_null : entities.back();
}

inline bool is_enabled(const __itt_domain* domain) {
    return domain != nullptr && domain->flags != 0;
}

}  // namespace detail
#endif

inline const char* default_domain_name() {
    return "ov::genai";
}

inline const char* domain_name(Domain domain) {
    switch (domain) {
    case Domain::Metrics:
        return "ov::genai::metrics";
    case Domain::LLM:
        return "ov::genai::llm";
    case Domain::VLM:
        return "ov::genai::vlm";
    case Domain::Whisper:
        return "ov::genai::whisper";
    case Domain::Tokenizer:
        return "ov::genai::tokenizer";
    case Domain::ContinuousBatching:
        return "ov::genai::continuous_batching";
    case Domain::GenAI:
    default:
        return default_domain_name();
    }
}

#ifdef ENABLE_PROFILING_ITT
inline __itt_domain* domain(Domain name = Domain::GenAI) {
    return detail::domain_by_name(domain_name(name));
}
#endif

class ScopedTask {
public:
    explicit ScopedTask(const char* name, Domain domain = Domain::GenAI) {
#ifdef ENABLE_PROFILING_ITT
        m_itt_domain = itt::domain(domain);
        if (!detail::is_enabled(m_itt_domain)) {
            return;
        }
        auto* task_handle = detail::string_handle(name);
        if (task_handle == nullptr) {
            return;
        }
        m_id = detail::next_id();
        const auto parent_id = detail::current_parent_id();
        detail::active_entities().push_back(m_id);
        __itt_id_create(m_itt_domain, m_id);
        __itt_task_begin(m_itt_domain, m_id, parent_id, task_handle);
        m_active = true;
#else
        (void)name;
#endif
    }

    ~ScopedTask() {
#ifdef ENABLE_PROFILING_ITT
        if (m_active) {
            __itt_task_end(m_itt_domain);
            detail::active_entities().pop_back();
            __itt_id_destroy(m_itt_domain, m_id);
        }
#endif
    }

    ScopedTask(const ScopedTask&) = delete;
    ScopedTask& operator=(const ScopedTask&) = delete;

    void add_metadata(const char* key, std::uint64_t value) const {
#ifdef ENABLE_PROFILING_ITT
        if (m_active) {
            auto metadata_value = value;
            auto* metadata_key = detail::string_handle(key);
            if (metadata_key == nullptr) {
                return;
            }
            __itt_metadata_add(m_itt_domain,
                               m_id,
                               metadata_key,
                               __itt_metadata_u64,
                               1,
                               &metadata_value);
        }
#else
        (void)key;
        (void)value;
#endif
    }

private:
#ifdef ENABLE_PROFILING_ITT
    __itt_domain* m_itt_domain = nullptr;
    __itt_id m_id = __itt_null;
    bool m_active = false;
#endif
};

class ScopedRegion {
public:
    explicit ScopedRegion(const char* name, Domain domain = Domain::GenAI) {
#ifdef ENABLE_PROFILING_ITT
        m_itt_domain = itt::domain(domain);
        if (!detail::is_enabled(m_itt_domain)) {
            return;
        }
        auto* region_handle = detail::string_handle(name);
        if (region_handle == nullptr) {
            return;
        }
        m_id = detail::next_id();
        const auto parent_id = detail::current_parent_id();
        detail::active_entities().push_back(m_id);
        __itt_id_create(m_itt_domain, m_id);
        __itt_region_begin(m_itt_domain, m_id, parent_id, region_handle);
        m_active = true;
#else
        (void)name;
#endif
    }

    ~ScopedRegion() {
#ifdef ENABLE_PROFILING_ITT
        if (m_active) {
            __itt_region_end(m_itt_domain, m_id);
            detail::active_entities().pop_back();
            __itt_id_destroy(m_itt_domain, m_id);
        }
#endif
    }

    ScopedRegion(const ScopedRegion&) = delete;
    ScopedRegion& operator=(const ScopedRegion&) = delete;

private:
#ifdef ENABLE_PROFILING_ITT
    __itt_domain* m_itt_domain = nullptr;
    __itt_id m_id = __itt_null;
    bool m_active = false;
#endif
};

inline bool domain_enabled(Domain domain) {
#ifdef ENABLE_PROFILING_ITT
    return detail::is_enabled(itt::domain(domain));
#else
    (void)domain;
    return false;
#endif
}

inline bool cb_trace_enabled() {
    return domain_enabled(Domain::ContinuousBatching);
}

inline void marker(const char* name, MarkerScope scope = MarkerScope::Task, Domain domain = Domain::GenAI) {
#ifdef ENABLE_PROFILING_ITT
    auto* itt_domain = itt::domain(domain);
    if (detail::is_enabled(itt_domain)) {
        auto* event_name = detail::string_handle(name);
        if (event_name != nullptr) {
            __itt_marker(itt_domain, __itt_null, event_name, detail::to_itt_scope(scope));
        }
    }
#else
    (void)name;
    (void)scope;
    (void)domain;
#endif
}

inline void counter_add(const char* name, std::uint64_t delta, Domain domain = Domain::Metrics) {
#ifdef ENABLE_PROFILING_ITT
    auto* itt_domain = itt::domain(domain);
    if (detail::is_enabled(itt_domain)) {
        auto* counter_name = detail::string_handle(name);
        if (counter_name != nullptr) {
            __itt_counter_inc_delta_v3(itt_domain,
                                       counter_name,
                                       static_cast<unsigned long long>(delta));
        }
    }
#else
    (void)name;
    (void)delta;
    (void)domain;
#endif
}

inline void counter_inc(const char* name, Domain domain = Domain::Metrics) {
    counter_add(name, 1U, domain);
}

}  // namespace ov::genai::itt

// Skip metadata formatting unless an ITT collector is listening.
#define GENAI_CB_TRACE_ENABLED() ::ov::genai::itt::cb_trace_enabled()

#define GENAI_ITT_CONCAT_INNER(a, b) a##b
#define GENAI_ITT_CONCAT(a, b) GENAI_ITT_CONCAT_INNER(a, b)

// Domain helpers.
#define GENAI_ITT_DEFAULT_DOMAIN() ::ov::genai::itt::Domain::GenAI
#define GENAI_ITT_DOMAIN(name) (name)
#define GENAI_ITT_METRICS_DOMAIN() ::ov::genai::itt::Domain::Metrics
#define GENAI_ITT_LLM_DOMAIN() ::ov::genai::itt::Domain::LLM
#define GENAI_ITT_VLM_DOMAIN() ::ov::genai::itt::Domain::VLM
#define GENAI_ITT_WHISPER_DOMAIN() ::ov::genai::itt::Domain::Whisper
#define GENAI_ITT_TOKENIZER_DOMAIN() ::ov::genai::itt::Domain::Tokenizer
#define GENAI_ITT_CB_DOMAIN() ::ov::genai::itt::Domain::ContinuousBatching

// Task helpers.
#define GENAI_ITT_SCOPED_TASK(name) ::ov::genai::itt::ScopedTask GENAI_ITT_CONCAT(genai_itt_task_, __LINE__)(name)
#define GENAI_ITT_SCOPED_TASK_D(domain, name) ::ov::genai::itt::ScopedTask GENAI_ITT_CONCAT(genai_itt_task_, __LINE__)((name), (domain))
#define GENAI_ITT_SCOPED_TASK_LLM(name) GENAI_ITT_SCOPED_TASK_D(GENAI_ITT_LLM_DOMAIN(), name)
#define GENAI_ITT_SCOPED_TASK_VLM(name) GENAI_ITT_SCOPED_TASK_D(GENAI_ITT_VLM_DOMAIN(), name)
#define GENAI_ITT_SCOPED_TASK_WHISPER(name) GENAI_ITT_SCOPED_TASK_D(GENAI_ITT_WHISPER_DOMAIN(), name)
#define GENAI_ITT_SCOPED_TASK_TOKENIZER(name) GENAI_ITT_SCOPED_TASK_D(GENAI_ITT_TOKENIZER_DOMAIN(), name)
#define GENAI_ITT_SCOPED_TASK_CB(name) GENAI_ITT_SCOPED_TASK_D(GENAI_ITT_CB_DOMAIN(), name)

// Region helpers.
#define GENAI_ITT_SCOPED_REGION(name) ::ov::genai::itt::ScopedRegion GENAI_ITT_CONCAT(genai_itt_region_, __LINE__)(name)
#define GENAI_ITT_SCOPED_REGION_D(domain, name) ::ov::genai::itt::ScopedRegion GENAI_ITT_CONCAT(genai_itt_region_, __LINE__)((name), (domain))
#define GENAI_ITT_SCOPED_REGION_LLM(name) GENAI_ITT_SCOPED_REGION_D(GENAI_ITT_LLM_DOMAIN(), name)
#define GENAI_ITT_SCOPED_REGION_VLM(name) GENAI_ITT_SCOPED_REGION_D(GENAI_ITT_VLM_DOMAIN(), name)

// Marker helpers.
#define GENAI_ITT_MARKER(name) ::ov::genai::itt::marker((name), ::ov::genai::itt::MarkerScope::Task)
#define GENAI_ITT_MARKER_D(domain, name) ::ov::genai::itt::marker((name), ::ov::genai::itt::MarkerScope::Task, (domain))
#define GENAI_ITT_MARKER_S(name, scope) ::ov::genai::itt::marker((name), (scope))
#define GENAI_ITT_MARKER_DS(domain, name, scope) ::ov::genai::itt::marker((name), (scope), (domain))
#define GENAI_ITT_MARKER_METRICS(name) GENAI_ITT_MARKER_D(GENAI_ITT_METRICS_DOMAIN(), name)

// Counter helpers.
#define GENAI_ITT_COUNTER_ADD(name, delta) ::ov::genai::itt::counter_add((name), static_cast<std::uint64_t>(delta))
#define GENAI_ITT_COUNTER_INC(name) ::ov::genai::itt::counter_inc((name))
#define GENAI_ITT_COUNTER_ADD_D(domain, name, delta) ::ov::genai::itt::counter_add((name), static_cast<std::uint64_t>(delta), (domain))
#define GENAI_ITT_COUNTER_INC_D(domain, name) ::ov::genai::itt::counter_inc((name), (domain))
#define GENAI_ITT_COUNTER_ADD_METRICS(name, delta) GENAI_ITT_COUNTER_ADD_D(GENAI_ITT_METRICS_DOMAIN(), name, delta)
#define GENAI_ITT_COUNTER_INC_METRICS(name) GENAI_ITT_COUNTER_INC_D(GENAI_ITT_METRICS_DOMAIN(), name)

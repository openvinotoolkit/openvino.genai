// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#ifdef ENABLE_PROFILING_ITT
#    include <ittnotify.h>
#    include <mutex>
#    include <unordered_map>
#endif

namespace ov::genai::itt {

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

}  // namespace detail
#endif

inline bool cb_trace_enabled() {
#ifdef ENABLE_PROFILING_ITT
    const auto* domain = detail::domain_by_name("ov.genai");
    return domain != nullptr && domain->flags != 0;
#else
    return false;
#endif
}

inline const char* default_domain_name() {
    return "ov.genai";
}

inline const char* domain_name(const char* name) {
    return name ? name : default_domain_name();
}

#ifdef ENABLE_PROFILING_ITT
inline __itt_domain* domain(const char* name = nullptr) {
    return detail::domain_by_name(domain_name(name));
}
#endif

class ScopedTask {
public:
    explicit ScopedTask(const char* name, const char* domain = nullptr) : m_domain(domain_name(domain)) {
#ifdef ENABLE_PROFILING_ITT
        __itt_task_begin(itt::domain(m_domain), __itt_null, __itt_null, detail::string_handle(name));
#else
        (void)name;
#endif
    }

    ~ScopedTask() {
#ifdef ENABLE_PROFILING_ITT
        __itt_task_end(itt::domain(m_domain));
#endif
    }

private:
    const char* m_domain = default_domain_name();
};

class ScopedRegion {
public:
    explicit ScopedRegion(const char* name, const char* domain = nullptr) : m_domain(domain_name(domain)) {
#ifdef ENABLE_PROFILING_ITT
        __itt_task_begin(itt::domain(m_domain), __itt_null, __itt_null, detail::string_handle(name));
#else
        (void)name;
#endif
    }

    ~ScopedRegion() {
#ifdef ENABLE_PROFILING_ITT
        __itt_task_end(itt::domain(m_domain));
#endif
    }

private:
    const char* m_domain;
};

inline void region_begin(const char* name, const char* domain = nullptr) {
#ifdef ENABLE_PROFILING_ITT
    __itt_task_begin(itt::domain(domain), __itt_null, __itt_null, detail::string_handle(name));
#else
    (void)name;
    (void)domain;
#endif
}

inline void region_end(const char* domain = nullptr) {
#ifdef ENABLE_PROFILING_ITT
    __itt_task_end(itt::domain(domain));
#else
    (void)domain;
#endif
}

inline void marker(const char* name, MarkerScope scope = MarkerScope::Task, const char* domain = nullptr) {
#ifdef ENABLE_PROFILING_ITT
    __itt_marker(itt::domain(domain), __itt_null, detail::string_handle(name), detail::to_itt_scope(scope));
#else
    (void)name;
    (void)scope;
    (void)domain;
#endif
}

inline void counter_add(const char* name, std::uint64_t delta, const char* domain = nullptr) {
#ifdef ENABLE_PROFILING_ITT
    __itt_counter_inc_delta_v3(itt::domain(domain), detail::string_handle(name), static_cast<unsigned long long>(delta));
#else
    (void)name;
    (void)delta;
    (void)domain;
#endif
}

inline void counter_inc(const char* name, const char* domain = nullptr) {
    counter_add(name, 1U, domain);
}

}  // namespace ov::genai::itt

// Skip metadata formatting unless an ITT collector is listening.
#define GENAI_CB_TRACE_ENABLED() ::ov::genai::itt::cb_trace_enabled()

#define GENAI_ITT_CONCAT_INNER(a, b) a##b
#define GENAI_ITT_CONCAT(a, b) GENAI_ITT_CONCAT_INNER(a, b)

// Domain helpers.
#define GENAI_ITT_DEFAULT_DOMAIN() ::ov::genai::itt::default_domain_name()
#define GENAI_ITT_DOMAIN(name) (name)

// Task helpers.
#define GENAI_ITT_SCOPED_TASK(name) ::ov::genai::itt::ScopedTask GENAI_ITT_CONCAT(genai_itt_task_, __LINE__)(name)
#define GENAI_ITT_SCOPED_TASK_D(domain, name) ::ov::genai::itt::ScopedTask GENAI_ITT_CONCAT(genai_itt_task_, __LINE__)((name), (domain))

// Region helpers.
#define GENAI_ITT_SCOPED_REGION(name) ::ov::genai::itt::ScopedRegion GENAI_ITT_CONCAT(genai_itt_region_, __LINE__)(name)
#define GENAI_ITT_SCOPED_REGION_D(domain, name) ::ov::genai::itt::ScopedRegion GENAI_ITT_CONCAT(genai_itt_region_, __LINE__)((name), (domain))
#define GENAI_ITT_REGION_BEGIN(name) ::ov::genai::itt::region_begin((name))
#define GENAI_ITT_REGION_END() ::ov::genai::itt::region_end()
#define GENAI_ITT_REGION_BEGIN_D(domain, name) ::ov::genai::itt::region_begin((name), (domain))
#define GENAI_ITT_REGION_END_D(domain) ::ov::genai::itt::region_end((domain))

// Marker helpers.
#define GENAI_ITT_MARKER(name) ::ov::genai::itt::marker((name), ::ov::genai::itt::MarkerScope::Task)
#define GENAI_ITT_MARKER_D(domain, name) ::ov::genai::itt::marker((name), ::ov::genai::itt::MarkerScope::Task, (domain))
#define GENAI_ITT_MARKER_S(name, scope) ::ov::genai::itt::marker((name), (scope))
#define GENAI_ITT_MARKER_DS(domain, name, scope) ::ov::genai::itt::marker((name), (scope), (domain))

// Counter helpers.
#define GENAI_ITT_COUNTER_ADD(name, delta) ::ov::genai::itt::counter_add((name), static_cast<std::uint64_t>(delta))
#define GENAI_ITT_COUNTER_INC(name) ::ov::genai::itt::counter_inc((name))
#define GENAI_ITT_COUNTER_ADD_D(domain, name, delta) ::ov::genai::itt::counter_add((name), static_cast<std::uint64_t>(delta), (domain))
#define GENAI_ITT_COUNTER_INC_D(domain, name) ::ov::genai::itt::counter_inc((name), (domain))

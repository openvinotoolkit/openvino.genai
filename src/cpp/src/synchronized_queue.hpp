// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class SynchronizedQueue
{
    std::queue<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;

public:
    SynchronizedQueue() = default;
    SynchronizedQueue(const SynchronizedQueue&) = delete;
    SynchronizedQueue(const SynchronizedQueue&&) = delete;
    SynchronizedQueue& operator=(const SynchronizedQueue&) = delete;

    T back() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this]{return !m_queue.empty(); });
        return m_queue.back();
    }

    T pull() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this]{return !m_queue.empty();});
        auto val = m_queue.front();
        m_queue.pop();
        return val;
    }

    void push(const T& item) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_queue.push(item);
        m_cv.notify_one();
    }

    bool empty() {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_queue.empty();
    }
};

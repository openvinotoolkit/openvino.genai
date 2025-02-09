// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <queue>
#include <mutex>
#include <future>
#include <algorithm>
#include <atomic>

namespace ov::genai {

// From OVMS:
// https://github.com/openvinotoolkit/model_server/blob/d73e85cbb8ac1d761754cb2064a00551a9ffc655/src/queue.hpp#L34
template <typename T>
class CircularBufferQueue
{
    int m_front_idx;
    std::atomic<int> m_back_idx;
    std::vector<int> m_values;
    std::queue<std::promise<int>> m_promises;
    std::vector<T> m_data;
    std::mutex m_front_mut;
    std::mutex m_queue_mutex;

public:

    CircularBufferQueue(size_t length, const std::function<T()>& create_fn) :
        m_values(length),
        m_front_idx{0},
        m_back_idx{0} {
        std::iota(m_values.begin(), m_values.end(), 0);
        m_data.reserve(length);
        for (size_t i = 0; i < length; i++) {
            m_data.emplace_back(std::move(create_fn()));
        }
    }

    CircularBufferQueue(const CircularBufferQueue&) = delete;
    CircularBufferQueue(const CircularBufferQueue&&) = delete;
    CircularBufferQueue& operator=(const CircularBufferQueue&) = delete;

    T& get(int value) {
        return m_data[value];
    }

    std::future<int> get_idle() {
        int value;
        std::promise<int> idle_promise;
        std::future<int> idle_future = idle_promise.get_future();
        std::unique_lock<std::mutex> lk(m_front_mut);
        if (m_values[m_front_idx] < 0) {
            std::unique_lock<std::mutex> queueLock(m_queue_mutex);
            m_promises.push(std::move(idle_promise));
        } else {
            value = m_values[m_front_idx];
            m_values[m_front_idx] = -1;
            m_front_idx = (m_front_idx + 1) % m_values.size();
            lk.unlock();
            idle_promise.set_value(value);
        }
        return idle_future;
    }

    void return_to(int value) {
        std::unique_lock<std::mutex> lk(m_queue_mutex);
        if (m_promises.size()) {
            std::promise<int> promise = std::move(m_promises.front());
            m_promises.pop();
            lk.unlock();
            promise.set_value(value);
            return;
        }
        int old_back = m_back_idx.load();
        while (!m_back_idx.compare_exchange_weak(
            old_back,
            (old_back + 1) % m_values.size(),
            std::memory_order_relaxed)) {
        }
        m_values[old_back] = value;
    }
};

template <typename T>
class CircularBufferQueueElementGuard {
    CircularBufferQueue<T>* m_queue;
    int m_value;
public:
    CircularBufferQueueElementGuard(CircularBufferQueue<T>* queue) : m_queue(queue) {
        m_value = m_queue->get_idle().get();   // blocking until we get the element
    }

    T& get() {
        return m_queue->get(m_value);
    }

    ~CircularBufferQueueElementGuard() {
        m_queue->return_to(m_value);
    }
};

}

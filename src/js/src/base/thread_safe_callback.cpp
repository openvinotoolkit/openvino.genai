// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/base/thread_safe_callback.hpp"

#include <algorithm>
#include <iostream>
#include <mutex>
#include <vector>

#include "include/addon.hpp"
#include "openvino/core/except.hpp"

namespace {

class CallbackEnvironmentClosing : public std::runtime_error {
public:
    CallbackEnvironmentClosing() : std::runtime_error("Node.js environment is closing") {}
};

class JavaScriptCallbackError : public std::runtime_error {
public:
    JavaScriptCallbackError(std::string message, std::string stack)
        : std::runtime_error(std::move(message)),
          stack(std::move(stack)) {}

    std::string stack;
};

}  // namespace

Napi::Value call_js_callback(const Napi::Function& callback,
                            const std::initializer_list<napi_value>& arguments,
                            napi_value receiver) {
    const auto env = callback.Env();
    if (!receiver) {
        receiver = env.Undefined();
    }
    napi_value result = nullptr;
    // Inspect raw statuses before node-addon-api constructs an error during shutdown.
    const napi_status status =
        napi_call_function(env, receiver, callback, arguments.size(), arguments.begin(), &result);
    if (status == napi_ok) {
        return Napi::Value(env, result);
    }
    if (status == napi_cannot_run_js || status == napi_closing) {
        throw CallbackEnvironmentClosing();
    }
    OPENVINO_ASSERT(status == napi_pending_exception, "Failed to invoke a JavaScript callback. Status: ", status);
    napi_value exception = nullptr;
    const napi_status exception_status = napi_get_and_clear_last_exception(env, &exception);
    OPENVINO_ASSERT(exception_status == napi_ok, "Failed to retrieve a callback exception. Status: ", exception_status);
    const Napi::Value value(env, exception);
    if (!value.IsObject() && !value.IsFunction() && !value.IsSymbol()) {
        // Coercion only probes whether V8 can run; preserve the original thrown value.
        napi_value string = nullptr;
        const napi_status conversion_status = napi_coerce_to_string(env, exception, &string);
        if (conversion_status == napi_pending_exception || conversion_status == napi_cannot_run_js ||
            conversion_status == napi_closing) {
            throw CallbackEnvironmentClosing();
        }
        OPENVINO_ASSERT(conversion_status == napi_ok,
                        "Failed to inspect a primitive callback exception. Status: ",
                        conversion_status);
    }
    throw Napi::Error(env, exception);
}

class CallbackEnvironment {
public:
    static std::shared_ptr<CallbackEnvironment> get(Napi::Env env);
    void add(const std::shared_ptr<ThreadSafeCallback::State>& callback);
    void finalized();

private:
    static void cleanup(napi_async_cleanup_hook_handle handle, void* data);
    void finish_cleanup();

    std::mutex m_mutex;
    std::vector<std::weak_ptr<ThreadSafeCallback::State>> m_callbacks;
    size_t m_active_callbacks = 0;
    bool m_closing = false;
    napi_async_cleanup_hook_handle m_cleanup_handle = nullptr;
};

struct ThreadSafeCallback::State {
    std::mutex mutex;
    napi_threadsafe_function function = nullptr;
    bool closing = false;
    bool released = false;
    std::vector<std::weak_ptr<CallbackWaiter>> waiters;
    std::shared_ptr<CallbackEnvironment> environment;
    std::function<void()> on_finalize;
    Napi::ObjectReference receiver;

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex);
        closing = true;
        const auto exception = std::make_exception_ptr(std::runtime_error("Node.js environment is closing"));
        for (const auto& weak_waiter : waiters) {
            if (auto waiter = weak_waiter.lock()) {
                waiter->reject(exception);
            }
        }
    }

    void release(napi_threadsafe_function_release_mode mode = napi_tsfn_release) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!released && function) {
            released = true;
            const napi_status status = napi_release_threadsafe_function(function, mode);
            if (status != napi_ok) {
                std::cerr << "Failed to release a GenAI thread-safe callback. Status: " << status << std::endl;
            }
        }
    }

    static void invoke(napi_env env, napi_value callback, void*, void* data) {
        std::unique_ptr<Invocation> invocation(static_cast<Invocation*>(data));
        if (!env || !callback) {
            return;
        }
        try {
            (*invocation)(Napi::Env(env), Napi::Function(env, callback));
        } catch (const CallbackEnvironmentClosing&) {
            return;
        } catch (const Napi::Error& error) {
            report_exception(env, error.Value());
        } catch (const std::exception& error) {
            report_exception(env, Napi::Error::New(env, error.what()).Value());
        }
    }

    static void report_exception(napi_env env, napi_value error) {
        const napi_status status = napi_fatal_exception(env, error);
        if (status != napi_ok) {
            std::cerr << "Failed to report a GenAI callback exception. Status: " << status << std::endl;
        }
    }

    static void finalize(napi_env, void* data, void*) {
        std::unique_ptr<std::shared_ptr<State>> owner(static_cast<std::shared_ptr<State>*>(data));
        const auto& state = *owner;
        state->receiver.Reset();
        auto on_finalize = std::move(state->on_finalize);
        if (on_finalize) {
            on_finalize();
        }
        state->environment->finalized();
    }
};

std::shared_ptr<CallbackEnvironment> CallbackEnvironment::get(Napi::Env env) {
    auto& environment = env.GetInstanceData<AddonData>()->callback_environment;
    if (!environment) {
        auto created = std::make_shared<CallbackEnvironment>();
        auto owner = std::make_unique<std::shared_ptr<CallbackEnvironment>>(created);
        const napi_status status =
            napi_add_async_cleanup_hook(env, cleanup, owner.get(), &created->m_cleanup_handle);
        OPENVINO_ASSERT(status == napi_ok, "Failed to register GenAI environment cleanup. Status: ", status);
        owner.release();
        environment = std::move(created);
    }
    return environment;
}

void CallbackEnvironment::add(const std::shared_ptr<ThreadSafeCallback::State>& callback) {
    std::lock_guard<std::mutex> lock(m_mutex);
    OPENVINO_ASSERT(!m_closing, "Node.js environment is closing");
    m_callbacks.erase(std::remove_if(m_callbacks.begin(),
                                    m_callbacks.end(),
                                    [](const auto& item) {
                                        return item.expired();
                                    }),
                      m_callbacks.end());
    m_callbacks.emplace_back(callback);
    ++m_active_callbacks;
}

void CallbackEnvironment::cleanup(napi_async_cleanup_hook_handle, void* data) {
    std::unique_ptr<std::shared_ptr<CallbackEnvironment>> owner(
        static_cast<std::shared_ptr<CallbackEnvironment>*>(data));
    const auto& environment = *owner;
    std::vector<std::shared_ptr<ThreadSafeCallback::State>> callbacks;
    {
        std::lock_guard<std::mutex> lock(environment->m_mutex);
        environment->m_closing = true;
        for (const auto& item : environment->m_callbacks) {
            if (auto callback = item.lock()) {
                callbacks.emplace_back(std::move(callback));
            }
        }
    }
    // Wake every native waiter before a TSFN finalizer can join an inference thread.
    for (const auto& callback : callbacks) {
        callback->cancel();
    }
    for (const auto& callback : callbacks) {
        callback->release(napi_tsfn_abort);
    }
    environment->finish_cleanup();
}

void CallbackEnvironment::finalized() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        --m_active_callbacks;
    }
    finish_cleanup();
}

void CallbackEnvironment::finish_cleanup() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_closing && m_active_callbacks == 0 && m_cleanup_handle) {
        const napi_status status = napi_remove_async_cleanup_hook(m_cleanup_handle);
        if (status != napi_ok) {
            std::cerr << "Failed to finish GenAI environment cleanup. Status: " << status << std::endl;
        }
        m_cleanup_handle = nullptr;
    }
}

ThreadSafeCallback::ThreadSafeCallback(Napi::Env env,
                                       const Napi::Function& callback,
                                       const std::string& name,
                                       bool keep_alive,
                                       std::function<void()> on_finalize,
                                       Napi::Object receiver)
    : m_state(std::make_shared<State>()) {
    m_state->environment = CallbackEnvironment::get(env);
    m_state->on_finalize = std::move(on_finalize);
    if (!receiver.IsEmpty()) {
        m_state->receiver = Napi::Persistent(receiver);
    }
    const auto resource_name = Napi::String::New(env, name);
    auto owner = std::make_unique<std::shared_ptr<State>>(m_state);
    m_state->environment->add(m_state);
    const napi_status status = napi_create_threadsafe_function(env,
                                                               callback,
                                                               nullptr,
                                                               resource_name,
                                                               0,
                                                               1,
                                                               owner.get(),
                                                               State::finalize,
                                                               nullptr,
                                                               State::invoke,
                                                               &m_state->function);
    if (status != napi_ok) {
        m_state->environment->finalized();
    }
    OPENVINO_ASSERT(status == napi_ok, "Failed to create a GenAI thread-safe callback. Status: ", status);
    owner.release();
    if (!keep_alive) {
        const napi_status unref_status = napi_unref_threadsafe_function(env, m_state->function);
        if (unref_status != napi_ok) {
            m_state->release();
        }
        OPENVINO_ASSERT(unref_status == napi_ok, "Failed to unref a GenAI callback. Status: ", unref_status);
    }
}

ThreadSafeCallback::~ThreadSafeCallback() {
    release();
}

napi_status ThreadSafeCallback::call(Invocation invocation) const {
    auto data = std::make_unique<Invocation>(std::move(invocation));
    std::lock_guard<std::mutex> lock(m_state->mutex);
    if (m_state->closing || m_state->released) {
        return napi_closing;
    }
    const napi_status status = napi_call_threadsafe_function(m_state->function, data.get(), napi_tsfn_nonblocking);
    if (status == napi_ok) {
        data.release();
    }
    return status;
}

void ThreadSafeCallback::release() const {
    m_state->release();
}

Napi::Object ThreadSafeCallback::receiver() const {
    return m_state->receiver.Value();
}

void ThreadSafeCallback::add_waiter(const std::shared_ptr<CallbackWaiter>& waiter) const {
    std::lock_guard<std::mutex> lock(m_state->mutex);
    OPENVINO_ASSERT(!m_state->closing && !m_state->released, "Node.js environment is closing");
    auto& waiters = m_state->waiters;
    waiters.erase(std::remove_if(waiters.begin(),
                                waiters.end(),
                                [](const auto& item) {
                                    const auto waiter = item.lock();
                                    return !waiter || waiter->settled();
                                }),
                  waiters.end());
    waiters.emplace_back(waiter);
}

std::exception_ptr capture_callback_exception(Napi::Env, const Napi::Value& value) {
    try {
        std::string message;
        std::string stack;
        if (value.IsObject()) {
            const auto object = value.As<Napi::Object>();
            const auto message_value = object.Get("message");
            message = message_value.IsUndefined() ? value.ToString().Utf8Value() : message_value.ToString().Utf8Value();
            const auto stack_value = object.Get("stack");
            if (stack_value.IsString()) {
                stack = stack_value.As<Napi::String>().Utf8Value();
            }
        } else {
            message = value.ToString().Utf8Value();
        }
        return std::make_exception_ptr(JavaScriptCallbackError(std::move(message), std::move(stack)));
    } catch (const Napi::Error& error) {
        return std::make_exception_ptr(
            std::runtime_error("Failed to read a JavaScript callback error: " + std::string(error.what())));
    }
}

Napi::Error callback_error(Napi::Env env, const std::exception_ptr& exception, const std::string& prefix) {
    try {
        std::rethrow_exception(exception);
    } catch (const JavaScriptCallbackError& error) {
        auto result = Napi::Error::New(env, prefix + error.what());
        auto cause = Napi::Error::New(env, error.what());
        if (!error.stack.empty()) {
            cause.Set("stack", error.stack);
        }
        result.Set("cause", cause.Value());
        return result;
    } catch (const std::exception& error) {
        return Napi::Error::New(env, prefix + error.what());
    }
}

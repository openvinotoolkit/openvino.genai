// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <napi.h>

#include "openvino/core/except.hpp"

class CallbackEnvironment;

std::exception_ptr capture_callback_exception(Napi::Env env, const Napi::Value& value);
Napi::Error callback_error(Napi::Env env, const std::exception_ptr& exception, const std::string& prefix = {});

struct CallbackWaiter {
    virtual ~CallbackWaiter() = default;
    virtual bool settled() const = 0;
    virtual void reject(std::exception_ptr exception) = 0;
};

template <typename Result>
class CallbackResult final : public CallbackWaiter {
public:
    std::future<Result> get_future() {
        return m_promise.get_future();
    }

    bool settled() const override {
        return m_settled.load();
    }

    void resolve(Result result) {
        if (!m_settled.exchange(true)) {
            m_promise.set_value(std::move(result));
        }
    }

    void reject(std::exception_ptr exception) override {
        if (!m_settled.exchange(true)) {
            m_promise.set_exception(std::move(exception));
        }
    }

private:
    std::atomic<bool> m_settled{false};
    std::promise<Result> m_promise;
};

class ThreadSafeCallback {
public:
    using Invocation = std::function<void(Napi::Env, Napi::Function)>;

    ThreadSafeCallback(Napi::Env env,
                       const Napi::Function& callback,
                       const std::string& name,
                       bool keep_alive = true,
                       std::function<void()> on_finalize = {},
                       Napi::Object receiver = {});
    ~ThreadSafeCallback();

    ThreadSafeCallback(const ThreadSafeCallback&) = delete;
    ThreadSafeCallback& operator=(const ThreadSafeCallback&) = delete;

    napi_status call(Invocation invocation) const;
    void release() const;
    Napi::Object receiver() const;

    template <typename Result, typename Invoke, typename Convert>
    Result call_and_wait(Invoke invoke, Convert convert, bool allow_promise = false) const {
        auto result = std::make_shared<CallbackResult<Result>>();
        auto future = result->get_future();
        add_waiter(result);
        const napi_status status = call([result, invoke = std::move(invoke), convert = std::move(convert), allow_promise](
                                            Napi::Env env,
                                            Napi::Function callback) {
            try {
                Napi::Value value = invoke(env, callback);
                if (value.IsPromise()) {
                    auto fulfilled = Napi::Function::New(env, [result, convert](const Napi::CallbackInfo& info) {
                        if (result->settled()) {
                            return;
                        }
                        try {
                            result->resolve(convert(info[0]));
                        } catch (const Napi::Error& error) {
                            result->reject(capture_callback_exception(info.Env(), error.Value()));
                        } catch (const std::exception&) {
                            result->reject(std::current_exception());
                        }
                    });
                    auto rejected = Napi::Function::New(env, [result](const Napi::CallbackInfo& info) {
                        if (!result->settled()) {
                            result->reject(capture_callback_exception(info.Env(), info[0]));
                        }
                    });
                    auto then = env.Global()
                                    .Get("Promise")
                                    .As<Napi::Function>()
                                    .Get("prototype")
                                    .As<Napi::Object>()
                                    .Get("then")
                                    .As<Napi::Function>();
                    then.Call(value, {fulfilled, rejected});
                    OPENVINO_ASSERT(allow_promise, "Callback must return synchronously");
                } else {
                    result->resolve(convert(value));
                }
            } catch (const Napi::Error& error) {
                result->reject(capture_callback_exception(env, error.Value()));
            } catch (const std::exception&) {
                result->reject(std::current_exception());
            }
        });
        if (status != napi_ok) {
            result->reject(std::make_exception_ptr(
                std::runtime_error("Thread-safe callback could not be queued. Status: " + std::to_string(status))));
        }
        return future.get();
    }

private:
    struct State;
    std::shared_ptr<State> m_state;
    void add_waiter(const std::shared_ptr<CallbackWaiter>& waiter) const;

    friend class CallbackEnvironment;
};

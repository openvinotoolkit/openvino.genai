// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// string

inline void debugPrint(const std::string& message) {
    const char* debugEnv = std::getenv("DEBUG");
    if (debugEnv != nullptr) {
        std::cout << message << std::endl;
    }
}

template <typename... Args>
void debugPrint(const std::string& format_str, Args&&... args) {
    const char* debugEnv = std::getenv("DEBUG");
    if (debugEnv != nullptr) {
        std::ostringstream oss;
        ((oss << args << " "), ...);
        std::cout << format_str << oss.str() << std::endl;
    }
}

// vector
template <typename T>
void debugPrint(const std::vector<T>& vec) {
    const char* debugEnv = std::getenv("DEBUG");
    if (debugEnv != nullptr) {
        std::cout << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << vec[i];
            if (i < vec.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

// tuple
template <typename Tuple, std::size_t Index>
void printTupleElement(const Tuple& t, std::integral_constant<std::size_t, Index>) {
    std::cout << std::get<Index>(t);
    if constexpr (Index + 1 != std::tuple_size<Tuple>::value) {
        std::cout << ", ";
    }
}

template <typename Tuple, std::size_t... Is>
void printTuple(const Tuple& t, std::index_sequence<Is...>) {
    (..., printTupleElement(t, std::integral_constant<std::size_t, Is>{}));
}

template <typename... Args>
void printTuple(const std::tuple<Args...>& t) {
    std::cout << "(";
    printTuple(t, std::index_sequence_for<Args...>{});
    std::cout << ")";
}

template <typename... Args>
void debugPrint(const std::tuple<Args...>& t) {
    const char* debugEnv = std::getenv("DEBUG");
    if (debugEnv != nullptr) {
        printTuple(t);
        std::cout << std::endl;
    }
}
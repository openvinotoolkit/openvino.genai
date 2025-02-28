// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "openvino/genai/visibility.hpp"


/**
 * @enum ov_genai_status_e
 * @brief This enum contains codes for all possible return values of the interface functions
 */
typedef enum {
    OK = 0,  //!< SUCCESS
    /*
     * @brief map exception to C++ interface
     */
    GENERAL_ERROR = -1,       //!< GENERAL_ERROR
    NOT_IMPLEMENTED = -2,     //!< NOT_IMPLEMENTED
    NETWORK_NOT_LOADED = -3,  //!< NETWORK_NOT_LOADED
    PARAMETER_MISMATCH = -4,  //!< PARAMETER_MISMATCH
    NOT_FOUND = -5,           //!< NOT_FOUND
    OUT_OF_BOUNDS = -6,       //!< OUT_OF_BOUNDS
    /*
     * @brief exception not of std::exception derived type was thrown
     */
    UNEXPECTED = -7,          //!< UNEXPECTED
    REQUEST_BUSY = -8,        //!< REQUEST_BUSY
    RESULT_NOT_READY = -9,    //!< RESULT_NOT_READY
    NOT_ALLOCATED = -10,      //!< NOT_ALLOCATED
    INFER_NOT_STARTED = -11,  //!< INFER_NOT_STARTED
    NETWORK_NOT_READ = -12,   //!< NETWORK_NOT_READ
    INFER_CANCELLED = -13,    //!< INFER_CANCELLED
    /*
     * @brief exception in C wrapper
     */
    INVALID_C_PARAM = -14,         //!< INVALID_C_PARAM
    UNKNOWN_C_ERROR = -15,         //!< UNKNOWN_C_ERROR
    NOT_IMPLEMENT_C_METHOD = -16,  //!< NOT_IMPLEMENT_C_METHOD
    UNKNOW_EXCEPTION = -17,        //!< UNKNOW_EXCEPTION
} ov_genai_status_e;
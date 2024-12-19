# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(cpp)

if(ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(ENABLE_JS)
    add_subdirectory(js)
endif()

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h
    ${CMAKE_BINARY_DIR}/stb_image.h
    EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image_resize2.h
    ${CMAKE_BINARY_DIR}/stb_image_resize2.h
    EXPECTED_HASH MD5=9293a158ddf9e44dd186c446372784d8)

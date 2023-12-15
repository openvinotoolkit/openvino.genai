// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

class ProgressBar {
public:
    ProgressBar(int total, int width = 40);

    void progress(int current);

    void finish();

private:
    void print_progress(int progress);

    int total;
    int width;
    int last_progress;
};

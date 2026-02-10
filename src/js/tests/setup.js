// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { downloadModel } from "./utils.js";
import { models } from "./models.js";

for (const model of Object.values(models)) {
  await downloadModel(model);
}

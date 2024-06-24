#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

GENAI_CPP_DOCKER_IMAGE ?= openvino_llm
GENAI_CPP_IMAGE_TAG ?= latest
HTTP_PROXY := "$(http_proxy)"
HTTPS_PROXY := "$(https_proxy)"
NO_PROXY := "$(no_proxy)"

ifeq ($(shell uname),Darwin)
    # MacOS
    CORES_TOTAL := $(shell sysctl -n hw.physicalcpu)
else
    # Ubuntu & Redhat
    CORES_PER_SOCKET := $(shell lscpu | awk '/^Core\(s\) per socket:/ {print $$NF}')
    SOCKETS := $(shell lscpu | awk '/^Socket\(s\):/ {print $$NF}')
    CORES_TOTAL := $$(($(SOCKETS) * $(CORES_PER_SOCKET)))
endif
JOBS ?= $(CORES_TOTAL)

.PHONY: default docker_build \

default: docker_build

.PHONY: docker_build
docker_build:
	docker build --build-arg http_proxy="$(http_proxy)" --build-arg no_proxy="$(no_proxy)" --build-arg https_proxy="$(https_proxy)" --build-arg JOBS=$(JOBS) -t $(GENAI_CPP_DOCKER_IMAGE):$(GENAI_CPP_IMAGE_TAG) .
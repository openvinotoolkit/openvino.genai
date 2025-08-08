// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const std::pair<std::string, std::string> chat_template_fallback_map[] = {
{
    // THUDM/chatglm3-6b
    "{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n {{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
    "{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n{{ ' ' }}{{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n{{ ' ' }}{{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
},
{
    // Qwen/Qwen2-VL-2B, Qwen/Qwen2-VL-7B
    "{% if messages is string %}{{ messages }}{% else %}{% for content in messages %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}",
    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
},
};

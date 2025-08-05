// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const std::pair<std::string, std::string> chat_template_fallback_map[] = {
{
    // THUDM/chatglm3-6b
    "{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n {{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
    "{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n{{ ' ' }}{{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n{{ ' ' }}{{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
}, 
{
    // tiiuae/falcon-7b-instruct
    "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message.strip() }}{% endif %}{% if message['role'] == 'user' %}{{ '\n\nUser: ' + message['content'].strip().replace('\r\n', '\n').replace('\n\n', '\n') }}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant: ' + message['content'].strip().replace('\r\n', '\n').replace('\n\n', '\n') }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\n\nAssistant:' }}{% endif %}",
    "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message.strip() }}{% endif %}{% if message['role'] == 'user' %}{{ '\n\nUser: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant: ' + message['content'].strip() }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\n\nAssistant:' }}{% endif %}"
},
{
    // Qwen/Qwen2-VL-2B, Qwen/Qwen2-VL-7B
    "{% if messages is string %}{{ messages }}{% else %}{% for content in messages %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}",
    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
},
{
    // llava-hf/llava-1.5-7b-hf
    "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}",
    "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'] | upper + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
},
};

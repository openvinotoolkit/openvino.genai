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
};

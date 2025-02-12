# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def get_tokenizer_configs():
    return {
        "meta-llama/Meta-Llama-3-8B-Instruct": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "pad_token": None,
            "unk_token": None,
            "chat_template": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        },
        "TheBloke/Mistral-7B-OpenOrca-GPTQ": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<|end_of_text|>",
            "unk_token": None,
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        },
        "upstage/SOLAR-10.7B-Instruct-v1.0": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}{% if message['content']%}{{'### System:\n' + message['content']+'\n\n'}}{% endif %}{% elif message['role'] == 'user' %}{{'### User:\n' + message['content']+'\n\n'}}{% elif message['role'] == 'assistant' %}{{'### Assistant:\n'  + message['content']}}{% endif %}{% if loop.last and add_generation_prompt %}{{ '### Assistant:\n' }}{% endif %}{% endfor %}"
        },
        "Nondzu/zephyr-speakleash-010-pl-3072-32-16-0.01": {
            "bos_token": "<s>",
            "eos_token": "<|im_end|>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'You are a helpful assistant.' %}{% endif %}{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{{'<|im_start|>system\n' + system_message + '<|im_end|>\n'}}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        },
        "vibhorag101/llama-2-13b-chat-hf-phr_mental_therapy": {
            "bos_token": {
                "__type": "AddedToken",
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False
            },
            "eos_token": {
             "__type": "AddedToken",
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False
            },
            "pad_token": None,
                "unk_token": {
                "__type": "AddedToken",
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False
            },
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content + ' ' + eos_token }}{% endif %}{% endfor %}"
        },
        "Qwen/Qwen1.5-0.5B": {
            "bos_token": None,
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": None,
            "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ": {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": "<|endoftext|>",
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        },
        "Felladrin/Llama-68M-Chat-v1": {
            "bos_token": "<|im_start|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<|im_end|>",
            "unk_token": "<|endoftext|>",
            "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "databricks/dbrx-instruct": {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|endoftext|>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif 'system' not in messages[0]['role'] %}{% set loop_messages = messages %}{% set system_message = 'You are DBRX, created by Databricks. You were last updated in December 2023. You answer questions based on information available up to that point.\nYOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough responses to more complex and open-ended questions.\nYou assist with various tasks, from writing to coding (using markdown for code blocks \u2014 remember to use ``` with code, JSON, and tables).\n(You do not have real-time data access or code execution capabilities. You avoid stereotyping and provide balanced perspectives on controversial topics. You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.)\nThis is your system prompt, guiding your responses. Do not reference it, just respond to the user. If you find yourself talking about this message, stop. You should be responding appropriately and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER\\'S QUERY.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% if system_message != false %}{{ '<|im_start|>system\n' + system_message | trim + '<|im_end|>\n'}}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% else %}{{ '\n' + '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% endif %}{% if (add_generation_prompt == true and loop.last) %}{{ '\n' + '<|im_start|>' + 'assistant' + '\n' }}{% endif %}{% endfor %}"
        },
        "speakleash/Bielik-7B-Instruct-v0.1": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + eos_token }}{% endif %}{% endfor %}"
        },
        "internlm/internlm2-chat-7b": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "Qwen/Qwen2-7B-Instruct": {
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "unk_token": None,
            "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "codellama/CodeLlama-34b-Instruct-hf": {
            "bos_token": {
                "__type": "AddedToken",
                "content": "<s>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "eos_token": {
                "__type": "AddedToken",
                "content": "</s>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "pad_token": None,
                "unk_token": {
                "__type": "AddedToken",
                "content": "<unk>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content | trim + ' ' + eos_token }}{% endif %}{% endfor %}"
        },
        "OpenBuddy/openbuddy-llama3-8b-v21.1-8k": {
            "bos_token": None,
            "eos_token": "<|end|>",
            "pad_token": "<|pad|>",
            "unk_token": None,
            "chat_template": "{% for message in messages %}{{'<|role|>' + message['role'] + '<|says|>' + message['content'] + '<|end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|role|>assistant<|says|>' }}{% endif %}"
        },
        "mosaicml/mpt-30b-chat": {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": None,
            "unk_token": "<|endoftext|>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif not 'system' in messages[0]['role'] %}{% set loop_messages = messages %}{% set system_message = 'A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% if system_message != false %}{{ '<|im_start|>system\n' + system_message.strip() + '\n'}}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% else %}{{ '\n' + '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% endif %}{% if (add_generation_prompt == true and loop.last) %}{{ '\n' + '<|im_start|>' + 'assistant' + '\n' }}{% elif (message['role'] == 'assistant') %}{% endif %}{% endfor %}"
        },
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {
            "bos_token": "<s>",
            "eos_token": "<|im_end|>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "{{bos_token}}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "deepseek-ai/deepseek-coder-6.7b-instruct": {
            "bos_token": {
                "__type": "AddedToken",
                "content": "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "eos_token": {
                "__type": "AddedToken",
                "content": "<|EOT|>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "pad_token": {
                "__type": "AddedToken",
                "content": "<\uff5cend\u2581of\u2581sentence\uff5c>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "unk_token": None,
            "chat_template": "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}"
        },
        "deepseek-ai/deepseek-math-7b-rl": {
            "bos_token": {
                "__type": "AddedToken",
                "content": "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "eos_token": {
                "__type": "AddedToken",
                "content": "<\uff5cend\u2581of\u2581sentence\uff5c>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "pad_token": {
                "__type": "AddedToken",
                "content": "<\uff5cend\u2581of\u2581sentence\uff5c>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "unk_token": None,
             "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
        },
        "FINGU-AI/FinguAI-Chat-v1": {
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "unk_token": None,
            "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "allenai/tulu-2-7b": {
            "bos_token": {
                "__type": "AddedToken",
                "content": "<s>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "eos_token": {
                "__type": "AddedToken",
                "content": "</s>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "pad_token": None,
                "unk_token": {
                "__type": "AddedToken",
                "content": "<unk>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        },
        "maldv/winter-garden-7b-alpha": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{{bos_token}}{% for message in messages %}{% if 'name' in message %}{{message['name'] + ('' if 'to' not in message else ' (to ' + message['to'] + ')') + ': ' + message['content'] + '\n\n'}}{% else %}{{message['content'] + '\n\n '}}{% endif %}{% endfor %}"
        },
        "mlabonne/NeuralMonarch-7B": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{{bos_token + message['role'] + '\n' + message['content'] + eos_token + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ bos_token + 'assistant\n' }}{% endif %}"
        },
        "meta-llama/Llama-2-7b-chat-hf": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        },
        "GritLM/GritLM-7B": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<s>",
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        },
        "ishorn5/RTLCoder-Deepseek-v1.1": {
            "bos_token": "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
            "eos_token": "<\uff5cend\u2581of\u2581sentence\uff5c>",
            "pad_token": "<\uff5cend\u2581of\u2581sentence\uff5c>",
            "unk_token": None,
            "chat_template": "{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{{'### Response:\\n'}}\n"
        },
        "jondurbin/bagel-34b-v0.2": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}"
        },
        "openchat/openchat-3.5-0106": {
            "bos_token": "<s>",
            "eos_token": "<|end_of_turn|>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}"
        },
        "mobiuslabsgmbh/aanaphi2-v0.1": {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "[PAD]",
            "unk_token": "<|endoftext|>",
            "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{'### Human: ' + message['content'].strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{'### Assistant: '  + message['content'].strip() + '\n'}}{% endif %}{% endfor %}"
        },
        "typeof/mistral-60m": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{bos_token + message['role'] + '\n' + message['content'] + eos_token + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ bos_token + 'assistant\n' }}{% endif %}"
        },
        "turboderp/Cat-Llama-3-70B-instruct": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<|end_of_text|>",
            "unk_token": None,
            "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nBelow is a conversation between a curious user and a helpful AI assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "saltlux/Ko-Llama3-Luxia-8B": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": "<|end_of_text|>",
            "unk_token": None,
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ content }}{% elif message['role'] == 'assistant' %}{{ content + '\\n' }}{% endif %}{% endfor %}"
        },
        "h2oai/h2o-danube2-1.8b-chat": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|prompt|>' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% elif message['role'] == 'assistant' %}{{ '<|answer|>'  + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|answer|>' }}{% endif %}{% endfor %}"
        },
        "abhishek/autotrain-llama3-70b-orpo-v1": {
            "bos_token": "<s>",
            "eos_token": "<|im_end|>",
            "pad_token": "<pad>",
            "unk_token": None,
            "chat_template": "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
        },
        "casperhansen/llama-3-70b-instruct-awq": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": "<|end_of_text|>",
            "unk_token": None,
            "chat_template": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        },
        "01-ai/Yi-1.5-34B-Chat": {
            "bos_token": "<|startoftext|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
        },
        "allenai/OLMo-7B-Instruct": {
            "bos_token": None,
            "eos_token": "<|endoftext|>",
            "pad_token": "<|padding|>",
            "unk_token": None,
            "chat_template": "{{ eos_token }}{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        },
        "TheBloke/deepseek-coder-33B-instruct-GPTQ": {
            "bos_token": "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
            "eos_token": "<|EOT|>",
            "pad_token": "<\uff5cend\u2581of\u2581sentence\uff5c>",
            "unk_token": None,
            "chat_template": "{%- set found_item = false -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set found_item = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not found_item -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{{'### Response:\\n'}}\n"
        },
        "cognitivecomputations/dolphin-2.8-mistral-7b-v02": {
            "bos_token": "<s>",
            "eos_token": "<|im_end|>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "alexsobolev/IcaroLM": {
            "bos_token": "<s>",
            "eos_token": "<|im_end|>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if message['from'] == 'human' %}{{'<|im_start|>user\n' + message['value'] + '<|im_end|>\n'}}{% elif message['from'] == 'gpt' %}{{'<|im_start|>assistant\n' + message['value'] + '<|im_end|>\n' }}{% else %}{{ '<|im_start|>system\n' + message['value'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "tokyotech-llm/Swallow-7b-instruct-v0.1": {
            "bos_token": {
                "__type": "AddedToken",
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False
            },
            "eos_token": {
                "__type": "AddedToken",
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False
            },
            "pad_token": None,
            "unk_token": {
                "__type": "AddedToken",
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False
            },
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = '\u3042\u306a\u305f\u306f\u8aa0\u5b9f\u3067\u512a\u79c0\u306a\u65e5\u672c\u4eba\u306e\u30a2\u30b7\u30b9\u30bf\u30f3\u30c8\u3067\u3059\u3002' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{{ bos_token }}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST] ' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ''  + content.strip() + '' + eos_token }}{% endif %}{% endfor %}"
        },
        "instructlab/merlinite-7b-lab": {
            "bos_token": "<s>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|pad|>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>'+ '\n' + message['content'] + '\n'}}{% elif message['role'] == 'user' %}{{'<|user|>' + '\n' + message['content'] + '\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>' + '\n' + message['content'] + '<|endoftext|>' + ('' if loop.last else '\n')}}{% endif %}{% endfor %}"
        },
        "microsoft/Phi-3-medium-128k-instruct": {
            "bos_token": "<s>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|placeholder6|>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"
        },
        "katuni4ka/tiny-random-phi3": {
            "bos_token": "<s>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'assistant' %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"
        },
        "microsoft/Phi-3-mini-128k-instruct": {
            "bos_token": "<s>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|placeholder6|>",
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"
        },
        "VAGOsolutions/SauerkrautLM-Qwen-32b": {
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "unk_token": None,
            "chat_template": "{% set system_message = 'Du bist ein freundlicher und hilfsbereiter KI-Assistent.' %}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
        },
        "AI-Sweden-Models/gpt-sw3-356m-instruct": {
            "bos_token": None,
            "eos_token": None,
            "pad_token": None,
            "unk_token": None,
            "chat_template": "{{ eos_token }}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content']}}{% else %}{{ 'Bot: ' + message['content']}}{% endif %}{{ message['text'] }}{{ bos_token }}{% endfor %}Bot:"
        },
        "google/gemma-7b-it": {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
        },
        "ise-uiuc/Magicoder-S-DS-6.7B": {
            "bos_token": "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
            "eos_token": "<\uff5cend\u2581of\u2581sentence\uff5c>",
            "pad_token": "<\uff5cend\u2581of\u2581sentence\uff5c>",
            "unk_token": None,
            "chat_template": "{{bos_token}}{{'You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n'}}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n        {{ raise_exception('System messages are not allowed in this template.') }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'@@ Instruction\n' + message['content'] + '\n\n'}}\n        {%- else %}\n{{'@@ Response\n' + message['content'] + eos_token + '\n\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{{'@@ Response\n'}}"
        },
        "Deci/DeciLM-7B": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '### User:\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '### System:\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '### Assistant:\n'  + message['content'] }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '### Assistant:' }}\n{% endif %}\n{% endfor %}"
        },
        "katuni4ka/tiny-random-minicpm": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{'<\u7528\u6237>' + message['content'].strip() + '<AI>'}}{% else %}{{message['content'].strip()}}{% endif %}{% endfor %}"
        },
        "UnicomLLM/Unichat-llama3-Chinese-8B-28K": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": "<|end_of_text|>",
            "unk_token": None,
            "chat_template": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = message['content']  %}{% if loop.index0 == 0  %}{% set content =bos_token + content %}{% endif %}{% if loop.index0 ==1 %}{% set content =  'Human:' + content %}{% endif %}{% if loop.index0 %2!=0 and loop.index0 !=1 %}{% set content =  bos_token+'Human:' + content %}{% endif %}{% if loop.index0 !=0 and loop.index0 %2==0 and  not loop.last %}{% set content = 'Assistant:'+content+ eos_token %}{% endif %}{{ content+'\n' }}{% endfor %}{{ 'Assistant:' }}"
        },
        "RLHFlow/LLaMA3-SFT": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "pad_token": "<|end_of_text|>",
            "unk_token": None,
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}{% endif %}"
        },
        "bofenghuang/vigogne-2-7b-chat": {
            "bos_token": {
                "__type": "AddedToken",
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False
            },
            "eos_token": {
                "__type": "AddedToken",
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False
            },
            "pad_token": None,
            "unk_token": {
                "__type": "AddedToken",
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False
            },
            "chat_template": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif true == true %}{% set loop_messages = messages %}{% set system_message = 'Vous \u00eates Vigogne, un assistant IA cr\u00e9\u00e9 par Zaion Lab. Vous suivez extr\u00eamement bien les instructions. Aidez autant que vous le pouvez.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|system|>: ' + system_message + '\\n' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '<|user|>: ' + message['content'].strip() + '\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>: ' + message['content'].strip() + eos_token + '\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>:' }}{% endif %}"
        },
        "aisingapore/sea-lion-7b-instruct": {
            "bos_token": None,
            "eos_token": "<|endoftext|>",
            "pad_token": "<|padding|>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}### USER:\n{{ message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}### RESPONSE:\n{{ message['content'] + '\n\n' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}{% if add_generation_prompt %}### RESPONSE:\n{% endif %}"
        },
        "microsoft/Phi-3-small-8k-instruct": {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": None,
            "chat_template": "{{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}"
        },
        "THUDM/cogvlm2-llama3-chat-19B": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": None,
            "unk_token": None,
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% else %}{{ eos_token }}{% endif %}"
        },
        "tiiuae/falcon-11B": {
            "bos_token": ">>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": None,
            "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ 'User: \n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ 'System: ' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ 'Falcon:\n'  + message['content']}}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ 'Falcon:' }}\n{% endif %}\n{% endfor %}"
        },
        "Mihaiii/Pallas-0.5": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}{% if message['content']%}{{'SYSTEM:\n' + message['content']+'\n\n'}}{% endif %}{% elif message['role'] == 'user' %}{{'USER:\n' + message['content']+'\n\n'}}{% elif message['role'] == 'assistant' %}{{'ASSISTANT:\n'  + message['content']}}{% endif %}{% if loop.last and add_generation_prompt %}{{ 'ASSISTANT:\n' }}{% endif %}{% endfor %}"
        },
        "prithivida/Asimov-7B-v2": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'### ' + message['role'] + ': ' + message['content'] }}{% endfor %}{% if add_generation_prompt %}{{ '### Assistant: ' }}{% endif %}"
        },
        "dreamgen/opus-v1.2-7b": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>'}}{% if message['role']=='assistant' %}{{'text'}}{% else %}{{message['role']}}{% endif %}{{'\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>text\n' }}{% endif %}"
        },
        "KnutJaegersberg/internlm-20b-llama": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.last and message['role'] != 'user' %}{{ raise_exception('Most recent message must come from user!') }}{% endif %}{% if message['role'] == 'user' %}{{ '<|User|>:' + message['content'] + '<eoh>\n'}}{% elif message['role'] == 'assistant' %}{{ '<|Bot|>:'  + message['content'] + '<eoa>\n'}}{% else %}{{ raise_exception('Only user and assistant roles are supported in this model!') }}{% endif %}{% endfor %}{{ '<|Bot|>:' }}"
        },
        "alpindale/WizardLM-2-8x22B": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{{ messages[0]['content'].strip() }}{% else %}{% set loop_messages = messages %}{{ 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' }}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% if message['role'] == 'system' or message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% else %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% else %}{% if message['role'] == 'system' or message['role'] == 'user' %}{{ '\nUSER: ' + message['content'].strip() }}{% else %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
        },
        "yentinglin/Taiwan-LLM-7B-v2.0-base": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() %}{% else %}{% set loop_messages = messages %}{% set system_message = '\u4f60\u662f\u4eba\u5de5\u667a\u6167\u52a9\u7406\uff0c\u4ee5\u4e0b\u662f\u7528\u6236\u548c\u4eba\u5de5\u667a\u80fd\u52a9\u7406\u4e4b\u9593\u7684\u5c0d\u8a71\u3002\u4f60\u8981\u5c0d\u7528\u6236\u7684\u554f\u984c\u63d0\u4f9b\u6709\u7528\u3001\u5b89\u5168\u3001\u8a73\u7d30\u548c\u79ae\u8c8c\u7684\u56de\u7b54\u3002' %}{% endif %}{{system_message + eos_token}}{% for message in loop_messages %}{% if message['role'] == 'user' %}USER: {{ message['content'].strip() + eos_token }}{% elif message['role'] == 'system' %}{{message['content'].strip() + eos_token}}{% elif message['role'] == 'assistant' %}ASSISTANT: {{ message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{'ASSISTANT:'}}{% endif %}"
        },
        "maywell/Synatra-Mixtral-8x7B": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n{% for message in messages %}{% if message['role'] == 'user' %}### Instruction:\n{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% elif message['role'] == 'assistant' %}### Response:\n{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% elif message['role'] == 'system' %}{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}\n### Response:\n{% endif %}"
        },
        "MediaTek-Research/Breeze-7B-Instruct-v1_0": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() %}{% else %}{% set loop_messages = messages %}{% set system_message = 'You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan.' %}{% endif %}{{ bos_token }} {{ system_message }} {% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/... or system/user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST] ' }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        },
        "MTSAIR/multi_verse_model": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Instruction:\n' + message['content'] + '\n### Response:\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% elif message['role'] == 'system' %}{{ '### System:\n' + message['content'] + '\n' }}{% endif %}{% endfor %}"
        },
        "bofenghuang/vigostral-7b-chat": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif true == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'Vous \u00eates Vigogne, un assistant IA cr\u00e9\u00e9 par Zaion Lab. Vous suivez extr\u00eamement bien les instructions. Aidez autant que vous le pouvez.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        },
        "SeaLLMs/SeaLLM-7B-v2.5": {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<eos>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "qnguyen3/Master-Yi-9B": {
            "bos_token": "<|startoftext|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
        },
        "meetkai/functionary-small-v2.5": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": "<|end_of_text|>",
            "unk_token": None,
            "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' or message['role'] == 'system' %}\n{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'tool' %}\n{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + 'name=' + message['name'] + '\n' + message['content'] + '<|eot_id|>' }}{% else %}\n{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'}}{% if message['content'] is not none %}\n{{ message['content'] }}{% endif %}\n{% if 'tool_calls' in message and message['tool_calls'] is not none %}\n{% for tool_call in message['tool_calls'] %}\n{{ '<|reserved_special_token_249|>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments'] }}{% endfor %}\n{% endif %}\n{{ '<|eot_id|>' }}{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        },
        "h2oai/h2o-danube-1.8b-chat": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|prompt|>' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ '<|system|>' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|answer|>'  + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|answer|>' }}{% endif %}{% endfor %}"
        },
        "TheBloke/CodeLlama-70B-Instruct-AWQ": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set user_index = 1 %}{% else %}{% set user_index = 0 %}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != ((loop.index0 + user_index) % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ '<s>' }}{% endif %}{% set content = 'Source: ' + message['role'] + '\n\n ' + message['content'].strip() %}{{ content + ' <step> ' }}{% endfor %}{{'Source: assistant\nDestination: user\n\n '}}"
        },
        "FairMind/Phi-3-mini-4k-instruct-bnb-4bit-Ita": {
            "bos_token": "<s>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] in ['user', 'system']) %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'assistant' %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"
        },
        "ibm-granite/granite-8b-code-instruct": {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": "<|endoftext|>",
            "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ 'Question:\n' + message['content'] + '\n\n' }}{% elif message['role'] == 'system' %}\n{{ 'System:\n' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Answer:\n'  + message['content'] + '\n\n' }}{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ 'Answer:\n' }}{% endif %}{% endfor %}"
        },
        "dicta-il/dictalm2.0-instruct": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        },
        "nvidia/Llama3-ChatQA-1.5-8B": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": None,
            "unk_token": None,
            "chat_template": "{{ bos_token }}{%- if messages[0]['role'] == 'system' -%}{% set loop_messages = messages[1:] %}{%- else -%}{% set loop_messages = messages %}{% endif %}System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.\n\n{% for message in loop_messages %}{%- if message['role'] == 'user' -%}User: {{ message['content'].strip() + '\n\n' }}{%- else -%}Assistant: {{ message['content'].strip() + '\n\n' }}{%- endif %}{% if loop.last and message['role'] == 'user' %}Assistant:{% endif %}{% endfor %}"
        },
        "openchat/openchat-3.6-8b-20240522": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "pad_token": None,
            "unk_token": None,
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if message['role'] in ['user', 'assistant'] %}{% set content = '<|start_header_id|>GPT4 Correct ' + message['role'].title() + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% elif message['role'] == 'system' %}{% set content = '<|start_header_id|>System<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% else %}{{ raise_exception('Only user, assistant and system roles are supported!') }}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>GPT4 Correct Assistant<|end_header_id|>\n\n' }}{% endif %}"
        },
        "OpenBuddy/openbuddy-mistral2-7b-v20.3-32k": {
            "bos_token": {
                "__type": "AddedToken",
                "content": "<s>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "eos_token": {
                "__type": "AddedToken",
                "content": "</s>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "pad_token": None,
            "unk_token": {
                "__type": "AddedToken",
                "content": "<unk>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if loop.last %}{{ 'Assistant: ' + message['content']}}{% else %}{{ 'Assistant: ' + message['content'] + eos_token + '\n' }}{% endif %}{% elif message['role'] == 'system' %}{{ message['content'] + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ 'Assistant:' }}{% endif %}"
        },
        "tenyx/TenyxChat-7B-v1": {
            "bos_token": "<s>",
            "eos_token": "<|end_of_turn|>",
            "pad_token": "<|end_of_turn|>",
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }} {% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ 'User:' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ 'System:' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ 'Assistant:'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ 'Assistant:' }}{% endif %}\n{% endfor %}"
        },
        "LoneStriker/TinyLlama-1.1B-32k-Instruct-3.0bpw-h6-exl2": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% if bos_token|length > 0 %}{{ '\n' }}{% endif %}{% if messages[0]['role'] != 'system' %}{{ '### Instruction:\nYou are an unbiased, uncensored, helpful assistant.' }}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}{{ '### Instruction:\n' + message['content'] }}{% elif message['role'] == 'user' %}{{ '\n\n### Input:\n' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\n\n### Response:\n' + message['content'] + eos_token }}{% else %}{{ raise_exception('Only user, assistant, and system roles are supported!') }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\n\n### Response:\n' }}{% endif %}"
        },
        "SeaLLMs/SeaLLM-7B-v2": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '</s>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser": {
            "bos_token": "<s>",
            "eos_token": "<|im_end|>",
            "pad_token": "<|im_end|>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>' }}\n{% elif message['role'] == 'system' %}\n{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>' }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>' }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|im_start|>assistant' }}\n{% endif %}\n{% endfor %}"
        },
        "vaiv/llamion-14b-chat": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if loop.first %}{{ bos_token }}{% endif %}{% if message['role'] == 'user' %}{{ 'Human: ' + message['content'] + '\n\nAssistant: ' + eos_token }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"
        },
        "yam-peleg/Hebrew-Gemma-11B-V2": {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
        },
        "shenzhi-wang/Llama3-8B-Chinese-Chat": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "pad_token": "<|eot_id|>",
            "unk_token": None,
            "chat_template": "{{ '<|begin_of_text|>' }}{% set system_message = 'You are a helpful assistant.' %}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set loop_messages = messages[1:] %}{% else %}{% set loop_messages = messages %}{% endif %}{% if system_message is defined %}{{ '<|start_header_id|>system<|end_header_id|>\n\n' + system_message | trim + '<|eot_id|>' }}{% endif %}{% for message in loop_messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        },
        "ericzzz/falcon-rw-1b-chat": {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": None,
            "unk_token": "<|endoftext|>",
            "chat_template": "{% for message in messages %}{% if loop.index > 1 and loop.previtem['role'] != 'assistant' %}{{ ' ' }}{% endif %}{% if message['role'] == 'system' %}{{ '[SYS] ' + message['content'].strip() }}{% elif message['role'] == 'user' %}{{ '[INST] ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ '[RESP] '  + message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' [RESP] ' }}{% endif %}"
        },
        "NLPark/AnFeng_v3_Avocet": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<s>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
        },
        "microsoft/Phi-3-vision-128k-instruct": {
            "bos_token": "<s>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{- '<|assistant|>\n' -}}{% endif %}"
        },
        "jphme/em_german_leo_mistral": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}{{message['content'] + ' '}}{% elif message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' '}}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + ' '}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT: '}}{% endif %}"
        },
        "nlpai-lab/KULLM3": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \"\ub2f9\uc2e0\uc740 \uace0\ub824\ub300\ud559\uad50 NLP&AI \uc5f0\uad6c\uc2e4\uc5d0\uc11c \ub9cc\ub4e0 AI \ucc57\ubd07\uc785\ub2c8\ub2e4. \ub2f9\uc2e0\uc758 \uc774\ub984\uc740 'KULLM'\uc73c\ub85c, \ud55c\uad6d\uc5b4\ub85c\ub294 '\uad6c\ub984'\uc744 \ub73b\ud569\ub2c8\ub2e4. \ub2f9\uc2e0\uc740 \ube44\ub3c4\ub355\uc801\uc774\uac70\ub098, \uc131\uc801\uc774\uac70\ub098, \ubd88\ubc95\uc801\uc774\uac70\ub098 \ub610\ub294 \uc0ac\ud68c \ud1b5\ub150\uc801\uc73c\ub85c \ud5c8\uc6a9\ub418\uc9c0 \uc54a\ub294 \ubc1c\uc5b8\uc740 \ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4. \uc0ac\uc6a9\uc790\uc640 \uc990\uac81\uac8c \ub300\ud654\ud558\uba70, \uc0ac\uc6a9\uc790\uc758 \uc751\ub2f5\uc5d0 \uac00\ub2a5\ud55c \uc815\ud655\ud558\uace0 \uce5c\uc808\ud558\uac8c \uc751\ub2f5\ud568\uc73c\ub85c\uc368 \ucd5c\ub300\ud55c \ub3c4\uc640\uc8fc\ub824\uace0 \ub178\ub825\ud569\ub2c8\ub2e4. \uc9c8\ubb38\uc774 \uc774\uc0c1\ud558\ub2e4\uba74, \uc5b4\ub5a4 \ubd80\ubd84\uc774 \uc774\uc0c1\ud55c\uc9c0 \uc124\uba85\ud569\ub2c8\ub2e4. \uac70\uc9d3 \uc815\ubcf4\ub97c \ubc1c\uc5b8\ud558\uc9c0 \uc54a\ub3c4\ub85d \uc8fc\uc758\ud569\ub2c8\ub2e4.\" %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]'}}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        },
        "HuggingFaceH4/zephyr-7b-gemma-sft-v0.1": {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "chat_template": "{% if messages[0]['role'] == 'user' or messages[0]['role'] == 'system' %}{{ bos_token }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% elif messages[-1]['role'] == 'assistant' %}{{ eos_token }}{% endif %}"
        },
        "MediaTek-Research/Breeze-7B-Instruct-v0_1": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() %}{% else %}{% set loop_messages = messages %}{% set system_message = 'You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan.' %}{% endif %}{{ bos_token }}{{ system_message }}   {% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/... or system/user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST] ' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        },
        "microsoft/DialoGPT-large": {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": None,
            "unk_token": "<|endoftext|>",
            "chat_template": "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}"
        },
        "meta-llama/Meta-Llama-Guard-2-8B": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": None,
            "unk_token": None,
            "chat_template": "{% if messages|length % 2 == 0 %}{% set role = 'Agent' %}{% else %}{% set role = 'User' %}{% endif %}{{ \"[INST] Task: Check if there is unsafe content in '\" + role + \"' messages in conversations according our safety policy with the below categories.\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nS1: Violent Crimes. \nS2: Non-Violent Crimes. \nS3: Sex Crimes. \nS4: Child Exploitation. \nS5: Specialized Advice. \nS6: Privacy. \nS7: Intellectual Property. \nS8: Indiscriminate Weapons. \nS9: Hate. \nS10: Self-Harm. \nS11: Sexual Content. \n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\n\" }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{% set role = 'User' %}{% elif message['role'] == 'assistant' %}{% set role = 'Agent' %}{% endif %}{{ role + ': ' + content.strip() + '\n\n' }}{% endfor %}{{ \"<END CONVERSATION>\n\nProvide your safety assessment for ONLY THE LAST \" + role + \" message in the above conversation:\n - First line must read 'safe' or 'unsafe'.\n - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]\" }}"
        },
        "chinoll/Yi-6b-200k-dpo": {
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|Human|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|System|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|Assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|Assistant|>' }}\n{% endif %}\n{% endfor %}"
        },
        "shanchen/llama3-8B-slerp-biomed-chat-chinese": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "pad_token": "<|eot_id|>",
            "unk_token": None,
            "chat_template": "{{ '<|begin_of_text|>' }}{% set system_message = 'You are Llama3-8B-Chinese-Chat-v2, finetuned from Llama3-8B-Instruct on Chinese-English dataset using the ORPO algorithm. You are a helpful assistant.' %}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set loop_messages = messages[1:] %}{% else %}{% set loop_messages = messages %}{% endif %}{% if system_message is defined %}{{ '<|start_header_id|>system<|end_header_id|>\n\n' + system_message | trim + '<|eot_id|>' }}{% endif %}{% for message in loop_messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        },
        "MLP-KTLim/llama-3-Korean-Bllossom-8B": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "pad_token": "<|end_of_text|>",
            "unk_token": None,
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        },
        "UnfilteredAI/UNfilteredAI-1B": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ '<|system|>' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>'  + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}{% endfor %}"
        },
        "abacusai/Smaug-Mixtral-v0.1": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{%if message['content'][0] == '$' %} {% endif %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        },
        "ProbeMedicalYonseiMAILab/medllama3-v20": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "pad_token": "<|eot_id|>",
            "unk_token": None,
            "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{ '\n\nHuman: ' + message['content'] +  eos_token }}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\n\nAssistant: ' }}{% endif %}"
        },
        "vinai/PhoGPT-4B-Chat": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' and loop.first %}{{ '### C\u00e2u h\u1ecfi: ' + message['content'].strip() }}{% elif message['role'] == 'user' %}{{ '\n### C\u00e2u h\u1ecfi: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ '\n### Tr\u1ea3 l\u1eddi: ' + message['content'] + eos_token }}{% endif %}{% if loop.last %}{% if message['role'] == 'user' and add_generation_prompt %}{{ '\n### Tr\u1ea3 l\u1eddi:' }}{% endif %}{% endif %}{% endfor %}"
        },
        "lucyknada/microsoft_WizardLM-2-7B": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{{ bos_token + (messages[0]['content'].strip() + '\n\n' if messages[0]['role'] == 'system' else '') }}{% for message in (messages[1:] if messages[0]['role'] == 'system' else messages) %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'].strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'].strip() + eos_token + '\n' }}{% endif %}{% if loop.last and message['role'] == 'user' and add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}{% endfor %}"
        },
        "bigcode/starcoder2-15b-instruct-v0.1": {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": None,
            "unk_token": "<|endoftext|>",
            "chat_template": "{{bos_token}}{{'You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n'}}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n        {{ raise_exception('System messages are not allowed in this template.') }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction\n' + message['content'] + '\n\n'}}\n        {%- else %}\n{{'### Response\n' + message['content'] + eos_token + '\n\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{{'### Response\n'}}"
        },
        "AliAbdelrasheed/maqa_llama_4bit": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "pad_token": "<|reserved_special_token_250|>",
            "unk_token": None,
            "chat_template": "{% for message in messages %}{% if message['from'] == 'human' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['value'] | trim + '<|eot_id|>' }}{% elif message['from'] == 'gpt' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['value'] | trim + '<|eot_id|>' }}{% else %}{{ '<|start_header_id|>' + message['from'] + '<|end_header_id|>\n\n' + message['value'] | trim + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        },
        "lightonai/alfred-40b-1023": {
            "bos_token": None,
            "eos_token": "<end_message>",
            "pad_token": None,
            "unk_token": None,
            "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<start_user>' + message['content'].strip() + '<end_message>' }}{% elif message['role'] == 'system' %}{{ '<start_system>' + message['content'].strip() + '<end_message>' }}{% elif message['role'] == 'assistant' %}{{ '<start_assistant>'  + message['content'] + '<end_message>' }}{% else %}{{ raise_exception('Only system, user and assistant roles are supported.') }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<start_assistant>' }}{% endif %}{% endfor %}"
        },
        "aloobun/CosmicBun-8B": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": "<|end_of_text|>",
            "unk_token": None,
            "chat_template": "{%- set ns = namespace(found=false) -%}{%- for message in messages -%}{%- if message['role'] == 'system' -%}{%- set ns.found = true -%}{%- endif -%}{%- endfor -%}{%- for message in messages %}{%- if message['role'] == 'system' -%}{{- '<|im_start|>system\n' + message['content'].rstrip() + '<|im_end|>\n' -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'<|im_start|>user\n' + message['content'].rstrip() + '<|im_end|>\n'-}}{%- else -%}{{-'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-'<|im_start|>assistant\n'-}}{%- endif -%}"
        },
        "Undi95/Mixtral-8x7B-MoE-RP-Story": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}\n"
        },
        "TIGER-Lab/MAmmoTH2-8B-Plus": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "pad_token": "<|eot_id|>",
            "unk_token": None,
            "chat_template": "{% set system_message = 'You are a helpful assistant.' %}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ '<|begin_of_text|>' + '<|start_header_id|>system<|end_header_id|>\\n\\n' + system_message + '<|eot_id|>' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\\n\\n' + content + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|eot_id|>' }}{% endif %}{% endfor %}"
        },
        "codellama/CodeLlama-70b-Instruct-hf": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set user_index = 1 %}{% else %}{% set user_index = 0 %}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != ((loop.index0 + user_index) % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ '<s>' }}{% endif %}{% set content = 'Source: ' + message['role'] + '\n\n ' + message['content'] | trim %}{{ content + ' <step> ' }}{% endfor %}{{'Source: assistant\nDestination: user\n\n '}}"
        },
        "stephenlzc/Mistral-7B-v0.3-Chinese-Chat-uncensored": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "[control_768]",
            "unk_token": "<unk>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{{ '<s>' + system_message }}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ ' [INST] ' + content + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' }}{% endif %}{% endfor %}"
        },
        "gorilla-llm/gorilla-openfunctions-v2": {
            "bos_token": "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
            "eos_token": "<|EOT|>",
            "pad_token": "<\uff5cend\u2581of\u2581sentence\uff5c>",
            "unk_token": None,
            "chat_template": "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Gorilla LLM model, developed by Gorilla LLM, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}"
        },
        "ghost-x/ghost-7b-alpha": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'plugins' %}\n{{ '<|plugins|>\n'  + message['content'] + '\n\nStandards for using the tool must comply with the following syntax:\n[execute]({\"type\": string, \"function\": string, \"arguments\": object})' + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% elif message['role'] == 'execute' %}\n{{ '<|assistant|>\n[execute]('  + message['content'] + ')<//>' + eos_token }}\n{% elif message['role'] == 'response' %}\n{{ '<|tool|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        },
        "winninghealth/WiNGPT2-Llama-3-8B-Chat": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": "<|end_of_text|>",
            "unk_token": None,
            "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}System\uff1a{% endif %}{% if message['role'] == 'user' %}User\uff1a{% endif %}{% if message['role'] == 'assistant' %}Assistant\uff1a{% endif %}{{ message['content'] }}<|end_of_text|>\n {% endfor %}Assistant\uff1a"
        },
        "BramVanroy/Llama-2-13b-chat-dutch": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{%set system_message = 'Je bent een behulpzame, respectvolle en eerlijke assistent. Antwoord altijd zo behulpzaam mogelijk. Je antwoorden mogen geen schadelijke, onethische, racistische, seksistische, gevaarlijke of illegale inhoud bevatten. Zorg ervoor dat je antwoorden sociaal onbevooroordeeld en positief van aard zijn.\n\nAls een vraag nergens op slaat of feitelijk niet coherent is, leg dan uit waarom in plaats van iets niet correct te antwoorden. Als je het antwoord op een vraag niet weet, deel dan geen onjuiste informatie.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\n' + content.strip() + '\n<</SYS>>\n\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        },
        "THUDM/chatglm3-6b": {
            "bos_token": None,
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n {{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
        },
        "microsoft/Phi-3-mini-4k-instruct": {
            "bos_token": "<s>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": "<unk>",
            "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}"
        },
        "mistralai/Mistral-7B-Instruct-v0.1": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": None,
            "unk_token": "<unk>",
            "chat_template": "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"
        },
        "meta-llama/Meta-Llama-3.1-8B-Instruct": {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "pad_token": None,
            "unk_token": None,
            "chat_template": "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n",
        }
    }

// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

const std::string qwen3_asr_chat_template = R"qwen3asr({%- set ns = namespace(system_text="") -%}
{%- for m in messages -%}
  {%- if m.role == 'system' -%}
    {%- if m.content is string -%}
      {%- set ns.system_text = ns.system_text + m.content -%}
    {%- else -%}
      {%- for c in m.content -%}
        {%- if c.type == 'text' and (c.text is defined) -%}
          {%- set ns.system_text = ns.system_text + c.text -%}
        {%- endif -%}
      {%- endfor -%}
    {%- endif -%}
  {%- endif -%}
{%- endfor -%}

{%- set ns2 = namespace(audio_tokens="") -%}
{%- for m in messages -%}
  {%- if m.content is not string -%}
    {%- for c in m.content -%}
      {%- if c.type == 'audio' or ('audio' in c) or ('audio_url' in c) -%}
        {%- set ns2.audio_tokens = ns2.audio_tokens + "<|audio_start|><|audio_pad|><|audio_end|>" -%}
      {%- endif -%}
    {%- endfor -%}
  {%- endif -%}
{%- endfor -%}

{{- '<|im_start|>system\n' + (ns.system_text if ns.system_text is string else '') + '<|im_end|>\n' -}}
{{- '<|im_start|>user\n' + ns2.audio_tokens + '<|im_end|>\n' -}}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n' -}}
{%- endif -%})qwen3asr";

const std::pair<std::string, std::string> chat_template_fallback_map[] = {
{
    // Qwen3-ASR: preserve audio prompts while allowing minja to infer tool-call syntax.
    qwen3_asr_chat_template,
    qwen3_asr_chat_template + R"qwen3asr(
{%- for m in messages -%}
  {%- if m.role == 'assistant' and m.tool_calls is defined -%}
    {%- for tc in m.tool_calls -%}
{{- '<|im_start|>assistant\n' + tc.function.name + ' ' + (tc.function.arguments if tc.function.arguments is string else tc.function.arguments|tojson) + '<|im_end|>\n' -}}
    {%- endfor -%}
  {%- endif -%}
{%- endfor -%})qwen3asr"
},
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

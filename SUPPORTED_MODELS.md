# OpenVINO™ GenAI: Supported Models

## Large language models

<table>
  <tbody style="vertical-align: top;">
    <tr>
      <th>Architecture</th>
      <th>Models</th>
      <th>Example HuggingFace Models</th>
    </tr>
    <tr>
      <td><code>AquilaModel</code></td>
      <td>Aquila</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/BAAI/Aquila-7B"><code>BAAI/Aquila-7B</code></a></li>
          <li><a href="https://huggingface.co/BAAI/AquilaChat-7B"><code>BAAI/AquilaChat-7B</code></a></li>
          <li><a href="https://huggingface.co/BAAI/Aquila2-7B"><code>BAAI/Aquila2-7B</code></a></li>
          <li><a href="https://huggingface.co/BAAI/AquilaChat2-7B"><code>BAAI/AquilaChat2-7B</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>ArcticForCausalLM</code></td>
      <td>Snowflake</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/Snowflake/snowflake-arctic-instruct"><code>Snowflake/snowflake-arctic-instruct</code></a></li>
          <li><a href="https://huggingface.co/Snowflake/snowflake-arctic-base"><code>Snowflake/snowflake-arctic-base</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>BaichuanForCausalLM</code></td>
      <td>Baichuan2</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat"><code>baichuan-inc/Baichuan2-7B-Chat</code></a></li>
          <li><a href="https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat"><code>baichuan-inc/Baichuan2-13B-Chat</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td rowspan="2"><code>BloomForCausalLM</code></td>
      <td>Bloom</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/bigscience/bloom-560m"><code>bigscience/bloom-560m</code></a></li>
          <li><a href="https://huggingface.co/bigscience/bloom-1b1"><code>bigscience/bloom-1b1</code></a></li>
          <li><a href="https://huggingface.co/bigscience/bloom-1b7"><code>bigscience/bloom-1b7</code></a></li>
          <li><a href="https://huggingface.co/bigscience/bloom-3b"><code>bigscience/bloom-3b</code></a></li>
          <li><a href="https://huggingface.co/bigscience/bloom-7b1"><code>bigscience/bloom-7b1</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td> Bloomz</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/bigscience/bloomz-560m"><code>bigscience/bloomz-560m</code></a></li>
          <li><a href="https://huggingface.co/bigscience/bloomz-1b1"><code>bigscience/bloomz-1b1</code></a></li>
          <li><a href="https://huggingface.co/bigscience/bloomz-1b7"><code>bigscience/bloomz-1b7</code></a></li>
          <li><a href="https://huggingface.co/bigscience/bloomz-3b"><code>bigscience/bloomz-3b</code></a></li>
          <li><a href="https://huggingface.co/bigscience/bloomz-7b1"><code>bigscience/bloomz-7b1</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>ChatGLMModel</code></td>
      <td>ChatGLM</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/THUDM/chatglm2-6b"><code>THUDM/chatglm2-6b</code></a></li>
          <li><a href="https://huggingface.co/THUDM/chatglm3-6b"><code>THUDM/chatglm3-6b</code></a></li>
          <li><a href="https://huggingface.co/THUDM/glm-4-9b"><code>THUDM/glm-4-9b</code></a></li>
          <li><a href="https://huggingface.co/THUDM/glm-4-9b-chat"><code>THUDM/glm-4-9b-chat</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>CodeGenForCausalLM</code></td>
      <td>CodeGen</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/Salesforce/codegen-350m-multi"><code>Salesforce/codegen-350m-multi</code></a></li>
          <li><a href="https://huggingface.co/Salesforce/codegen-2B-multi"><code>Salesforce/codegen-2B-multi</code></a></li>
          <li><a href="https://huggingface.co/Salesforce/codegen-6B-multi"><code>Salesforce/codegen-6B-multi</code></a></li>
          <li><a href="https://huggingface.co/Salesforce/codegen-16B-multi"><code>Salesforce/codegen-16B-multi</code></a></li>
          <li><a href="https://huggingface.co/Salesforce/codegen-350m-mono"><code>Salesforce/codegen-350m-mono</code></a></li>
          <li><a href="https://huggingface.co/Salesforce/codegen-2B-mono"><code>Salesforce/codegen-2B-mono</code></a></li>
          <li><a href="https://huggingface.co/Salesforce/codegen-6B-mono"><code>Salesforce/codegen-6B-momo</code></a></li>
          <li><a href="https://huggingface.co/Salesforce/codegen-16B-mono"><code>Salesforce/codegen-16B-mono</code></a></li>
          <li><a href="https://huggingface.co/Salesforce/codegen2-1B_P"><code>Salesforce/codegen2-1B_P</code></a></li>
          <li><a href="https://huggingface.co/Salesforce/codegen2-3_7B_P"><code>Salesforce/codegen2-3_7B_P</code></a></li>
          <li><a href="https://huggingface.co/Salesforce/codegen2-7B_P"><code>Salesforce/codegen2-7B_P</code></a></li>
          <li><a href="https://huggingface.co/Salesforce/codegen2-16B_P"><code>Salesforce/codegen2-16B_P</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td rowspan="2"><code>CohereForCausalLM</code></td>
      <td>Aya</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/CohereForAI/aya-23-8B"><code>CohereForAI/aya-23-8B</code></a></li>
          <li><a href="https://huggingface.co/CohereForAI/aya-expanse-8b"><code>CohereForAI/aya-expanse-8b</code></a></li>
          <li><a href="https://huggingface.co/CohereForAI/aya-23-35B"><code>CohereForAI/aya-23-35B</code></a></li>
          <li><a href="https://huggingface.co/CohereForAI/aya-expanse-35b"><code>CohereForAI/aya-expanse-35b</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
    <td>C4AI Command R</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024"><code>CohereForAI/c4ai-command-r7b-12-2024</code></a></li>
          <li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-v01"><code>CohereForAI/c4ai-command-r-v01</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>DbrxForCausalLM</code></td>
      <td>DBRX</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/databricks/dbrx-instruct"><code>databricks/dbrx-instruct</code></a></li>
          <li><a href="https://huggingface.co/databricks/dbrx-base"><code>databricks/dbrx-base</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>DeciLMForCausalLM</code></td>
      <td>DeciLM</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/Deci/DeciLM-7B"><code>Deci/DeciLM-7B</code></a></li>
          <li><a href="https://huggingface.co/Deci/DeciLM-7B-instruct"><code>Deci/DeciLM-7B-instruct</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>DeepseekForCausalLM</code></td>
      <td>DeepSeek-MoE</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/deepseek-ai/deepseek-moe-16b-base"><code>deepseek-ai/deepseek-moe-16b-base</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat"><code>deepseek-ai/deepseek-moe-16b-chat</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>DeepseekV2ForCausalLM</code></td>
      <td>DeepSeekV2</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite"><code>deepseek-ai/DeepSeek-V2-Lite</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat"><code>deepseek-ai/DeepSeek-V2-Lite-Chat</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite"><code>deepseek-ai/DeepSeek-Coder-V2-Lite</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Chat"><code>deepseek-ai/DeepSeek-Coder-V2-Lite-Chat</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>DeepseekV3ForCausalLM</code></td>
      <td>DeepSeekV3</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3"><code>deepseek-ai/DeepSeek-V3</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-Base"><code>deepseek-ai/DeepSeek-V3-Base</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1"><code>deepseek-ai/DeepSeek-R1</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero"><code>deepseek-ai/DeepSeek-R1-Zero</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>ExaoneForCausalLM</code></td>
      <td>Exaone</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"><code>LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"><code>LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct"><code>LGAI-EXAONE/EXAONE-3.5-32B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"><code>LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>FalconForCausalLM</code></td>
      <td>Falcon</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/tiiuae/falcon-11B"><code>tiiuae/falcon-11B</code></a></li>
          <li><a href="https://huggingface.co/tiiuae/falcon-7b"><code>tiiuae/falcon-7b</code></a></li>
          <li><a href="https://huggingface.co/tiiuae/falcon-7b-instruct"><code>tiiuae/falcon-7b-instruct</code></a></li>
          <li><a href="https://huggingface.co/tiiuae/falcon-40b"><code>tiiuae/falcon-40b</code></a></li>
          <li><a href="https://huggingface.co/tiiuae/falcon-40b-instruct"><code>tiiuae/falcon-40b-instruct</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>GemmaForCausalLM</code></td>
      <td>Gemma</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/google/gemma-2b"><code>google/gemma-2b</code></a></li>
          <li><a href="https://huggingface.co/google/gemma-2b-it"><code>google/gemma-2b-it</code></a></li>
          <li><a href="https://huggingface.co/google/gemma-1.1-2b-it"><code>google/gemma-1.1-2b-it</code></a></li>
          <li><a href="https://huggingface.co/google/codegemma-2b"><code>google/codegemma-2b</code></a></li>
          <li><a href="https://huggingface.co/google/codegemma-1.1-2b"><code>google/codegemma-1.1-2b</code></a></li>
          <li><a href="https://huggingface.co/google/gemma-7b"><code>google/gemma-7b</code></a></li>
          <li><a href="https://huggingface.co/google/gemma-7b-it"><code>google/gemma-7b-it</code></a></li>
          <li><a href="https://huggingface.co/google/gemma-1.1-7b-it"><code>google/gemma-1.1-7b-it</code></a></li>
          <li><a href="https://huggingface.co/google/codegemma-7b"><code>google/codegemma-7b</code></a></li>
          <li><a href="https://huggingface.co/google/codegemma-1.1-7b"><code>google/codegemma-1.1-7b</code></a></li>
          <li><a href="https://huggingface.co/google/codegemma-7b-it"><code>google/codegemma-7b-it</code></a></li>
          <li><a href="https://huggingface.co/google/codegemma-1.1-7b-it"><code>google/codegemma-1.1-7b-it</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>Gemma2ForCausalLM</code></td>
      <td>Gemma2</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/google/gemma-2-2b"><code>google/gemma-2-2b</code></a></li>
          <li><a href="https://huggingface.co/google/gemma-2-2b-it"><code>google/gemma-2-2b-it</code></a></li>
          <li><a href="https://huggingface.co/google/gemma-2-9b"><code>google/gemma-2-9b</code></a></li>
          <li><a href="https://huggingface.co/google/gemma-2-9b-it"><code>google/gemma-2-9b-it</code></a></li>
          <li><a href="https://huggingface.co/google/gemma-2-27b"><code>google/gemma-2-27b</code></a></li>
          <li><a href="https://huggingface.co/google/gemma-2-27b-it"><code>google/gemma-2-27b-it</code></a></li>
        </ul>
      </td>
    </tr>
   <tr>
      <td><code>GlmForCausalLM</code></td>
      <td>GLM</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/THUDM/glm-edge-1.5b-chat"><code>THUDM/glm-edge-1.5b-chat</code></a></li>
          <li><a href="https://huggingface.co/THUDM/glm-edge-4b-chat"><code>THUDM/glm-edge-4b-chat</code></a></li>
          <li><a href="https://huggingface.co/THUDM/glm-4-9b-hf"><code>THUDM/glm-4-9b-hf</code></a></li>
          <li><a href="https://huggingface.co/THUDM/glm-4-9b-chat-hf"><code>THUDM/glm-4-9b-chat-hf</code></a></li>
          <li><a href="https://huggingface.co/THUDM/glm-4-9b-chat-1m-hf"><code>THUDM/glm-4-9b-chat-1m-hf</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td rowspan="2"><code>GPT2LMHeadModel</code></td>
      <td>GPT2</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/openai-community/gpt2"><code>openai-community/gpt2</code></a></li>
          <li><a href="https://huggingface.co/openai-community/gpt2-medium"><code>openai-community/gpt2-medium</code></a></li>
          <li><a href="https://huggingface.co/openai-community/gpt2-large"><code>openai-community/gpt2-large</code></a></li>
          <li><a href="https://huggingface.co/openai-community/gpt2-xl"><code>openai-community/gpt2-xl</code></a></li>
          <li><a href="https://huggingface.co/distilbert/distilgpt2"><code>distilbert/distilgpt2</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
    <td>CodeParrot</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/codeparrot/codeparrot-small"><code>codeparrot/codeparrot-small</code></a></li>
          <li><a href="https://huggingface.co/codeparrot/codeparrot-small-code-to-text"><code>codeparrot/codeparrot-small-code-to-text</code></a></li>
          <li><a href="https://huggingface.co/codeparrot/codeparrot-small-text-to-code"><code>codeparrot/codeparrot-small-text-to-code</code></a></li>
          <li><a href="https://huggingface.co/codeparrot/codeparrot-small-multi"><code>codeparrot/codeparrot-small-multi</code></a></li>
          <li><a href="https://huggingface.co/codeparrot/codeparrot"><code>codeparrot/codeparrot</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>GPTBigCodeForCausalLM</code></td>
      <td>StarCoder</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/bigcode/starcoderbase-1b"><code>bigcode/starcoderbase-1b</code></a></li>
          <li><a href="https://huggingface.co/bigcode/starcoderbase-3b"><code>bigcode/starcoderbase-3b</code></a></li>
          <li><a href="https://huggingface.co/bigcode/starcoderbase-7b"><code>bigcode/starcoderbase-7b</code></a></li>
          <li><a href="https://huggingface.co/bigcode/starcoderbase"><code>bigcode/starcoderbase</code></a></li>
          <li><a href="https://huggingface.co/bigcode/starcoder"><code>bigcode/starcoder</code></a></li>
          <li><a href="https://huggingface.co/bigcode/octocoder"><code>bigcode/octocoder</code></a></li>
          <li><a href="https://huggingface.co/HuggingFaceH4/starchat-alpha"><code>HuggingFaceH4/starchat-alpha</code></a></li>
          <li><a href="https://huggingface.co/HuggingFaceH4/starchat-beta"><code>HuggingFaceH4/starchat-beta</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>GPTJForCausalLM</code></td>
      <td>GPT-J</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/EleutherAI/gpt-j-6b"><code>EleutherAI/gpt-j-6b</code></a></li>
          <li><a href="https://huggingface.co/crumb/Instruct-GPT-J"><code>crumb/Instruct-GPT-J</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>GPTNeoForCausalLM</code></td>
      <td>GPT Neo</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/EleutherAI/gpt-neo-1.3B"><code>EleutherAI/gpt-neo-1.3B</code></a></li>
          <li><a href="https://huggingface.co/EleutherAI/gpt-neo-2.7B"><code>EleutherAI/gpt-neo-2.7B</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td rowspan="3"><code>GPTNeoXForCausalLM</code></td>
      <td>GPT NeoX</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/EleutherAI/gpt-neox-20b"><code>EleutherAI/gpt-neox-20b</code></a></li>
        </ul>
      </td>
      </tr>
      <tr>
      <td>Dolly</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/databricks/dolly-v2-3b"><code>databricks/dolly-v2-3b</code></a></li>
          <li><a href="https://huggingface.co/databricks/dolly-v2-7b"><code>databricks/dolly-v2-7b</code></a></li>
          <li><a href="https://huggingface.co/databricks/dolly-v2-12b"><code>databricks/dolly-v2-12b</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>RedPajama</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/ikala/redpajama-3b-chat"><code>ikala/redpajama-3b-chat</code></a></li>
          <li><a href="https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1"><code>togethercomputer/RedPajama-INCITE-Chat-3B-v1</code></a></li>
          <li><a href="https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1"><code>togethercomputer/RedPajama-INCITE-Instruct-3B-v1</code></a></li>
          <li><a href="https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat"><code>togethercomputer/RedPajama-INCITE-7B-Chat</code></a></li>
          <li><a href="https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Instruct"><code>togethercomputer/RedPajama-INCITE-7B-Instruct</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>GPTNeoXJapaneseForCausalLM</code></td>
      <td>GPT NeoX Japanese</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/abeja/gpt-neox-japanese-2.7b"><code>abeja/gpt-neox-japanese-2.7b</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>GraniteForCausalLM</code></td>
      <td>Granite</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/ibm-granite/granite-3.2-2b-instruct"><code>ibm-granite/granite-3.2-2b-instruct</code></a></li>
          <li><a href="https://huggingface.co/ibm-granite/granite-3.2-8b-instruct"><code>ibm-granite/granite-3.2-8b-instruct</code></a></li>
          <li><a href="https://huggingface.co/ibm-granite/granite-3.1-2b-instruct"><code>ibm-granite/granite-3.1-2b-instruct</code></a></li>
          <li><a href="https://huggingface.co/ibm-granite/granite-3.1-8b-instruct"><code>ibm-granite/granite-3.1-8b-instruct</code></a></li>
          <li><a href="https://huggingface.co/ibm-granite/granite-3.0-2b-instruct"><code>ibm-granite/granite-3.0-2b-instruct</code></a></li>
          <li><a href="https://huggingface.co/ibm-granite/granite-3.0-8b-instruct"><code>ibm-granite/granite-3.0-8b-instruct</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>GraniteMoeForCausalLM</code></td>
      <td>GraniteMoE</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct"><code>ibm-granite/granite-3.1-1b-a400m-instruct</code></a></li>
          <li><a href="https://huggingface.co/iibm-granite/granite-3.1-3b-a800m-instruct"><code>ibm-granite/granite-3.1-3b-a800m-instruct</code></a></li>
          <li><a href="https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct"><code>ibm-granite/granite-3.0-1b-a400m-instruct</code></a></li>
          <li><a href="https://huggingface.co/iibm-granite/granite-3.0-3b-a800m-instruct"><code>ibm-granite/granite-3.0-3b-a800m-instruct</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>InternLMForCausalLM</code></td>
      <td>InternLM</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/internlm/internlm-chat-7b"><code>internlm/internlm-chat-7b</code></a></li>
          <li><a href="https://huggingface.co/internlm/internlm-7b"><code>internlm/internlm-7b</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>InternLM2ForCausalLM</code></td>
      <td>InternLM2</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/internlm/internlm2-chat-1_8b"><code>internlm/internlm2-chat-1_8b</code></a></li>
          <li><a href="https://huggingface.co/internlm/internlm2-1_8b"><code>internlm/internlm2-1_8b</code></a></li>
          <li><a href="https://huggingface.co/internlm/internlm2-chat-7b"><code>internlm/internlm2-chat-7b</code></a></li>
          <li><a href="https://huggingface.co/internlm/internlm2-7b"><code>internlm/internlm2-7b</code></a></li>
          <li><a href="https://huggingface.co/internlm/internlm2-chat-20b"><code>internlm/internlm2-chat-20b</code></a></li>
          <li><a href="https://huggingface.co/internlm/internlm2-20b"><code>internlm/internlm2-20b</code></a></li>
          <li><a href="https://huggingface.co/internlm/internlm2_5-chat-1_8b"><code>internlm/internlm2_5-chat-1_8b</code></a></li>
          <li><a href="https://huggingface.co/internlm/internlm2_5-1_8b"><code>internlm/internlm2_5-1_8b</code></a></li>
          <li><a href="https://huggingface.co/internlm/internlm2_5-chat-7b"><code>internlm/internlm2_5-chat-7b</code></a></li>
          <li><a href="https://huggingface.co/internlm/internlm2_5-7b"><code>internlm/internlm2_5-7b</code></a></li>
          <li><a href="https://huggingface.co/internlm/internlm2_5-chat-20b"><code>internlm/internlm2_5-chat-20b</code></a></li>
          <li><a href="https://huggingface.co/internlm/internlm2_5-20b"><code>internlm/internlm2_5-20b</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>JAISLMHeadModel</code></td>
      <td>Jais</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/inceptionai/jais-13b-chat"><code>inceptionai/jais-13b-chat</code></a></li>
          <li><a href="https://huggingface.co/inceptionai/jais-13b"><code>inceptionai/jais-13b</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td rowspan="5" vertical-align="top"><code>LlamaForCausalLM</code></td>
      <td>Llama 3</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/meta-llama/Llama-3.2-1B"><code>meta-llama/Llama-3.2-1B</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"><code>meta-llama/Llama-3.2-1B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-3.2-3B"><code>meta-llama/Llama-3.2-3B</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"><code>meta-llama/Llama-3.2-3B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-3.1-8B"><code>meta-llama/Llama-3.1-8B</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"><code>meta-llama/Llama-3.1-8B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B"><code>meta-llama/Meta-Llama-3-8B</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"><code>meta-llama/Meta-Llama-3-8B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-3.2-70B-Instruct"><code>meta-llama/Llama-3.2-70B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-3.1-70B"><code>meta-llama/Llama-3.1-70B</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct"><code>meta-llama/Llama-3.1-70B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B"><code>meta-llama/Meta-Llama-3-70B</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct"><code>meta-llama/Meta-Llama-3-70B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"><code>deepseek-ai/DeepSeek-R1-Distill-Llama-8B</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"><code>deepseek-ai/DeepSeek-R1-Distill-Llama-70B</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <!-- <td><code>LlamaForCausalLM</code></td> -->
      <td>Llama 2</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/meta-llama/Llama-2-13b-chat-hf"><code>meta-llama/Llama-2-13b-chat-hf</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-2-13b-hf"><code>meta-llama/Llama-2-13b-hf</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"><code>meta-llama/Llama-2-7b-chat-hf</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-2-7b-hf"><code>meta-llama/Llama-2-7b-hf</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-2-70b-chat-hf"><code>meta-llama/Llama-2-70b-chat-hf</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Llama-2-70b-hf"><code>meta-llama/Llama-2-70b-hf</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
    <td>Falcon3</td>
    <td>
        <ul>
          <li><a href="https://huggingface.co/tiiuae/Falcon3-1B-Instruct"><code>tiiuae/Falcon3-1B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/tiiuae/Falcon3-1B-Base"><code>tiiuae/Falcon3-1B-Base</code></a></li>
          <li><a href="https://huggingface.co/tiiuae/Falcon3-3B-Instruct"><code>tiiuae/Falcon3-3B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/tiiuae/Falcon3-3B-Base"><code>tiiuae/Falcon3-3B-Base</code></a></li>
          <li><a href="https://huggingface.co/tiiuae/Falcon3-7B-Instruct"><code>tiiuae/Falcon3-7B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/tiiuae/Falcon3-7B-Base"><code>tiiuae/Falcon3-7B-Base</code></a></li>
          <li><a href="https://huggingface.co/tiiuae/Falcon3-10B-Instruct"><code>tiiuae/Falcon3-10B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/tiiuae/Falcon3-10B-Base"><code>tiiuae/Falcon3-10B-Base</code></a></li>
        </ul>
    </td>
    </tr>
    <tr>
      <td>OpenLLaMA</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/openlm-research/open_llama_13b"><code>openlm-research/open_llama_13b</code></a></li>
          <li><a href="https://huggingface.co/openlm-research/open_llama_3b"><code>openlm-research/open_llama_3b</code></a></li>
          <li><a href="https://huggingface.co/openlm-research/open_llama_3b_v2"><code>openlm-research/open_llama_3b_v2</code></a></li>
          <li><a href="https://huggingface.co/openlm-research/open_llama_7b"><code>openlm-research/open_llama_7b</code></a></li>
          <li><a href="https://huggingface.co/openlm-research/open_llama_7b_v2"><code>openlm-research/open_llama_7b_v2</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>TinyLlama</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0"><code>TinyLlama/TinyLlama-1.1B-Chat-v1.0</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>MPTForCausalLM</code></td>
      <td>MPT</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/mosaicml/mpt-7b"><code>mosaicml/mpt-7b</code></a></li>
          <li><a href="https://huggingface.co/mosaicml/mpt-7b-instruct"><code>mosaicml/mpt-7b-instruct</code></a></li>
          <li><a href="https://huggingface.co/mosaicml/mpt-7b-chat"><code>mosaicml/mpt-7b-chat</code></a></li>
          <li><a href="https://huggingface.co/mosaicml/mpt-30b"><code>mosaicml/mpt-30b</code></a></li>
          <li><a href="https://huggingface.co/mosaicml/mpt-30b-instruct"><code>mosaicml/mpt-30b-instruct</code></a></li>
          <li><a href="https://huggingface.co/mosaicml/mpt-30b-chat"><code>mosaicml/mpt-30b-chat</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>MiniCPMForCausalLM</code></td>
      <td>MiniCPM</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16"><code>openbmb/MiniCPM-1B-sft-bf16</code></a></li>
          <li><a href="https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16"><code>openbmb/MiniCPM-2B-dpo-fp16</code></a></li>
          <li><a href="https://huggingface.co/openbmb/MiniCPM-2B-sft-fp32"><code>openbmb/MiniCPM-2B-sft-fp32</code></a></li>
          <li><a href="https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp32"><code>openbmb/MiniCPM-2B-dpo-fp32</code></a></li>
          <li><a href="https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16"><code>openbmb/MiniCPM-2B-sft-bf16</code></a></li>
          <li><a href="https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16"><code>openbmb/MiniCPM-2B-dpo-bf16</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>MiniCPM3ForCausalLM</code></td>
      <td>MiniCPM3</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/openbmb/MiniCPM3-4B"><code>openbmb/MiniCPM3-4B</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td rowspan="4"><code>MistralForCausalLM</code></td>
      <td>Mistral</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1"><code>mistralai/Mistral-7B-Instruct-v0.1</code></a></li>
          <li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"><code>mistralai/Mistral-7B-Instruct-v0.2</code></a></li>
          <li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"><code>mistralai/Mistral-7B-Instruct-v0.3</code></a></li>
          <li><a href="https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407"><code>mistralai/Mistral-Nemo-Instruct-2407</code></a></li>
          <li><a href="https://huggingface.co/mistralai/Mistral-Nemo-Base-2407"><code>mistralai/Mistral-Nemo-Base-2407</code></a></li>
          <li><a href="https://huggingface.co/mistralai/Mistral-7B-v0.1"><code>mistralai/Mistral-7B-v0.1</code></a></li>
          <li><a href="https://huggingface.co/mistralai/Mistral-7B-v0.3"><code>mistralai/Mistral-7B-v0.3</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Notus</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/argilla/notus-7b-v1"><code>argilla/notus-7b-v1</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Zephyr</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-beta"><code>HuggingFaceH4/zephyr-7b-beta</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Neural Chat</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/Intel/neural-chat-7b-v3-3"><code>Intel/neural-chat-7b-v3-3</code></a></li>
          <li><a href="https://huggingface.co/Intel/neural-chat-7b-v3-2"><code>Intel/neural-chat-7b-v3-2</code></a></li>
          <li><a href="https://huggingface.co/Intel/neural-chat-7b-v3-1"><code>Intel/neural-chat-7b-v3-1</code></a></li>
          <li><a href="https://huggingface.co/Intel/neural-chat-7b-v3"><code>Intel/neural-chat-7b-v3</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>MixtralForCausalLM</code></td>
      <td>Mixtral</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1"><code>mistralai/Mixtral-8x7B-Instruct-v0.1</code></a></li>
          <li><a href="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1"><code>mistralai/Mixtral-8x7B-v0.1</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>OlmoForCausalLM</code></td>
      <td>OLMo</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/allenai/OLMo-1B-hf"><code>allenai/OLMo-1B-hf</code></a></li>
          <li><a href="https://huggingface.co/allenai/OLMo-7B-hf"><code>allenai/OLMo-7B-hf</code></a></li>
          <li><a href="https://huggingface.co/allenai/OLMo-7B-Twin-2T-hf"><code>allenai/OLMo-7B-Twin-2T-hf</code></a></li>
          <li><a href="https://huggingface.co/allenai/OLMo-7B-Instruct-hf"><code>allenai/OLMo-7B-Instruct-hf</code></a></li>
          <li><a href="https://huggingface.co/allenai/OLMo-7B-0724-Instruct-hf"><code>allenai/OLMo-7B-0724-Instruct-hf</code></a></li>
          <li><a href="https://huggingface.co/allenai/OLMo-7B-0724-SFT-hf"><code>allenai/OLMo-7B-0724-SFT-hf</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>OPTForCausalLM</code></td>
      <td>OPT</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/facebook/opt-125m"><code>facebook/opt-125m</code></a></li>
          <li><a href="https://huggingface.co/facebook/opt-350m"><code>facebook/opt-350m</code></a></li>
          <li><a href="https://huggingface.co/facebook/opt-1.3b"><code>facebook/opt-1.3b</code></a></li>
          <li><a href="https://huggingface.co/facebook/opt-2.7b"><code>facebook/opt-2.7b</code></a></li>
          <li><a href="https://huggingface.co/facebook/opt-6.7b"><code>facebook/opt-6.7b</code></a></li>
          <li><a href="https://huggingface.co/facebook/opt-13b"><code>facebook/opt-13b</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>OrionForCausalLM</code></td>
      <td>Orion</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/OrionStarAI/Orion-14B-Chat"><code>OrionStarAI/Orion-14B-Chat</code></a></li>
          <li><a href="https://huggingface.co/OrionStarAI/Orion-14B-LongChat"><code>OrionStarAI/Orion-14B-LongChat</code></a></li>
          <li><a href="https://huggingface.co/OrionStarAI/Orion-14B-Base"><code>OrionStarAI/Orion-14B-Base</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>PhiForCausalLM</code></td>
      <td>Phi</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/microsoft/phi-2"><code>microsoft/phi-2</code></a></li>
          <li><a href="https://huggingface.co/microsoft/phi-1_5"><code>microsoft/phi-1_5</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>Phi3ForCausalLM</code></td>
      <td>Phi3</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct"><code>microsoft/Phi-3-mini-4k-instruct</code></a></li>
          <li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct"><code>microsoft/Phi-3-mini-128k-instruct</code></a></li>
          <li><a href="https://huggingface.co/microsoft/Phi-3-medium-4k-instruct"><code>microsoft/Phi-3-medium-4k-instruct</code></a></li>
          <li><a href="https://huggingface.co/microsoft/Phi-3-medimum-128k-instruct"><code>microsoft/Phi-3-medium-128k-instruct</code></a></li>
          <li><a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct"><code>microsoft/Phi-3.5-mini-instruct</code></a></li>
          <li><a href="https://huggingface.co/microsoft/Phi-4-mini-instruct"><code>microsoft/Phi-4-mini-instruct</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>QWenLMHeadModel</code></td>
      <td>Qwen</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/Qwen/Qwen-1_8B-Chat"><code>Qwen/Qwen-1_8B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen-1_8B-Chat-Int4"><code>Qwen/Qwen-1_8B-Chat-Int4</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen-1_8B"><code>Qwen/Qwen-1_8B</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen-7B-Chat"><code>Qwen/Qwen-7B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen-7B-Chat-Int4"><code>Qwen/Qwen-7B-Chat-Int4</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen-7B"><code>Qwen/Qwen-7B</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen-14B-Chat"><code>Qwen/Qwen-14B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen-14B-Chat-Int4"><code>Qwen/Qwen-14B-Chat-Int4</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen-14B"><code>Qwen/Qwen-14B</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen-72B-Chat"><code>Qwen/Qwen-72B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen-72B-Chat-Int4"><code>Qwen/Qwen-72B-Chat-Int4</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen-72B"><code>Qwen/Qwen-72B</code></a></li>
        </ul>
      </td>
    </tr>
      <tr>
      <td><code>Qwen2ForCausalLM</code></td>
      <td>Qwen2</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"><code>Qwen/Qwen2.5-0.5B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct"><code>Qwen/Qwen2.5-1.5B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct"><code>Qwen/Qwen2.5-3B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"><code>Qwen/Qwen2.5-7B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct"><code>Qwen/Qwen2.5-14B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen2.5-32B-Instruct"><code>Qwen/Qwen2.5-32B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct"><code>Qwen/Qwen2.5-72B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen2-0.5B-Instruct"><code>Qwen/Qwen2-0.5B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen2-1.5B-Instruct"><code>Qwen/Qwen2-1.5B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen2-7B-Instruct"><code>Qwen/Qwen2-7B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen2-72B-Instruct"><code>Qwen/Qwen2-72B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat"><code>Qwen/Qwen1.5-0.5B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat"><code>Qwen/Qwen1.5-1.8B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen1.5-4B-Chat"><code>Qwen/Qwen1.5-4B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen1.5-7B-Chat"><code>Qwen/Qwen1.5-7B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen1.5-14B-Chat"><code>Qwen/Qwen1.5-14B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen1.5-32B-Chat"><code>Qwen/Qwen1.5-32B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/QwQ-32B"><code>Qwen/QwQ-32B</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"><code>deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"><code>deepseek-ai/DeepSeek-R1-Distill-Qwen-7B</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"><code>deepseek-ai/DeepSeek-R1-Distill-Qwen-14B</code></a></li>
          <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"><code>deepseek-ai/DeepSeek-R1-Distill-Qwen-32B</code></a></li>
        </ul>
      </td>
    </tr>
    </tr>
      <tr>
      <td><code>Qwen2MoeForCausalLM</code></td>
      <td>Qwen2MoE</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct"><code>Qwen/Qwen2-57B-A14B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen2-57B-A14B"><code>Qwen/Qwen2-57B-A14B</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat"><code>Qwen/Qwen1.5-MoE-A2.7B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B"><code>Qwen/Qwen1.5-MoE-A2.7B</code></a></li>
        </ul>
      </td>
    </tr>
    </tr>
      <tr>
      <td><code>StableLmForCausalLM</code></td>
      <td>StableLM</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/stabilityai/stablelm-zephyr-3b"><code>stabilityai/stablelm-zephyr-3b</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stablelm-2-1_6b"><code>stabilityai/stablelm-2-1_6b</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stablelm-2-12b"><code>stabilityai/stablelm-2-12b</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b"><code>stabilityai/stablelm-2-zephyr-1_6b</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stablelm-3b-4e1t"><code>stabilityai/stablelm-3b-4e1t</code></a></li>
        </ul>
      </td>
    </tr>
    </tr>
      <tr>
      <td><code>Starcoder2ForCausalLM</code></td>
      <td>Startcoder2</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/bigcode/starcoder2-3b"><code>bigcode/starcoder2-3b</code></a></li>
          <li><a href="https://huggingface.co/bigcode/starcoder2-7b"><code>bigcode/starcoder2-7b</code></a></li>
          <li><a href="https://huggingface.co/bigcode/starcoder2-15b"><code>bigcode/starcoder2-15b</code></a></li>
        </ul>
      </td>
    </tr>
    </tr>
      <tr>
      <td><code>XGLMForCausalLM</code></td>
      <td>XGLM</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/facebook/xglm-564M"><code>facebook/xglm-564M</code></a></li>
          <li><a href="https://huggingface.co/facebook/xglm-1.7B"><code>facebook/xglm-1.7B</code></a></li>
          <li><a href="https://huggingface.co/facebook/xglm-2.9B"><code>facebook/xglm-2.9B</code></a></li>
          <li><a href="https://huggingface.co/facebook/xglm-4.5B"><code>facebook/xglm-4.5B</code></a></li>
          <li><a href="https://huggingface.co/facebook/xglm-7.5B"><code>facebook/xglm-7.5B</code></a></li>
        </ul>
      </td>
    </tr>
    </tr>
      <tr>
      <td><code>XverseForCausalLM</code></td>
      <td>Xverse</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/xverse/XVERSE-7B"><code>xverse/XVERSE-7B</code></a></li>
          <li><a href="https://huggingface.co/xverse/XVERSE-7B-Chat"><code>xverse/XVERSE-7B-Chat</code></a></li>
          <li><a href="https://huggingface.co/xverse/XVERSE-13B"><code>xverse/XVERSE-13B</code></a></li>
          <li><a href="https://huggingface.co/xverse/XVERSE-13B-Chat"><code>xverse/XVERSE-13B-Chat</code></a></li>
          <li><a href="https://huggingface.co/xverse/XVERSE-65B"><code>xverse/XVERSE-65B</code></a></li>
          <li><a href="https://huggingface.co/xverse/XVERSE-65B-Chat"><code>xverse/XVERSE-65B-Chat</code></a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

> [!NOTE]
> LoRA adapters are supported.

The pipeline can work with other similar topologies produced by `optimum-intel` with the same model signature. The model is required to have the following inputs after the conversion:
1. `input_ids` contains the tokens.
2. `attention_mask` is filled with `1`.
3. `beam_idx` selects beams.
4. `position_ids` (optional) encodes a position of currently generating token in the sequence and a single `logits` output.

> [!NOTE]
> Models should belong to the same family and have the same tokenizers.

## Image generation models

<table>
  <tbody style="vertical-align: top;">
    <tr>
      <th>Architecture</th>
      <th>Text 2 image</th>
      <th>Image 2 image</th>
      <th>Inpainting</th>
      <th>LoRA support</th>
      <th>Example HuggingFace Models</th>
    </tr>
    <tr>
      <td><code>Latent Consistency Model</code></td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7"><code>SimianLuo/LCM_Dreamshaper_v7</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>Stable Diffusion</code></td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/CompVis/stable-diffusion-v1-1"><code>CompVis/stable-diffusion-v1-1</code></a></li>
          <li><a href="https://huggingface.co/CompVis/stable-diffusion-v1-2"><code>CompVis/stable-diffusion-v1-2</code></a></li>
          <li><a href="https://huggingface.co/CompVis/stable-diffusion-v1-3"><code>CompVis/stable-diffusion-v1-3</code></a></li>
          <li><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4"><code>CompVis/stable-diffusion-v1-4</code></a></li>
          <li><a href="https://huggingface.co/jcplus/stable-diffusion-v1-5"><code>jcplus/stable-diffusion-v1-5</code></a></li>
          <li><a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5"><code>stable-diffusion-v1-5/stable-diffusion-v1-5</code></a></li>
          <li><a href="https://huggingface.co/botp/stable-diffusion-v1-5"><code>botp/stable-diffusion-v1-5</code></a></li>
          <li><a href="https://huggingface.co/dreamlike-art/dreamlike-anime-1.0"><code>dreamlike-art/dreamlike-anime-1.0</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-2"><code>stabilityai/stable-diffusion-2</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-2-base"><code>stabilityai/stable-diffusion-2-base</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-2-1"><code>stabilityai/stable-diffusion-2-1</code></a></li>
          <li><a href="https://huggingface.co/bguisard/stable-diffusion-nano-2-1"><code>bguisard/stable-diffusion-nano-2-1</code></a></li>
          <li><a href="https://huggingface.co/justinpinkney/pokemon-stable-diffusion"><code>justinpinkney/pokemon-stable-diffusion</code></a></li>
          <li><a href="https://huggingface.co/stablediffusionapi/architecture-tuned-model"><code>stablediffusionapi/architecture-tuned-model</code></a></li>
          <li><a href="https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1"><code>IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1</code></a></li>
          <li><a href="https://huggingface.co/ZeroCool94/stable-diffusion-v1-5"><code>ZeroCool94/stable-diffusion-v1-5</code></a></li>
          <li><a href="https://huggingface.co/pcuenq/stable-diffusion-v1-4"><code>pcuenq/stable-diffusion-v1-4</code></a></li>
          <li><a href="https://huggingface.co/rinna/japanese-stable-diffusion"><code>rinna/japanese-stable-diffusion</code></a></li>
          <li><a href="https://huggingface.co/benjamin-paine/stable-diffusion-v1-5"><code>benjamin-paine/stable-diffusion-v1-5</code></a></li>
          <li><a href="https://huggingface.co/philschmid/stable-diffusion-v1-4-endpoints"><code>philschmid/stable-diffusion-v1-4-endpoints</code></a></li>
          <li><a href="https://huggingface.co/naclbit/trinart_stable_diffusion_v2"><code>naclbit/trinart_stable_diffusion_v2</code></a></li>
          <li><a href="https://huggingface.co/Fictiverse/Stable_Diffusion_PaperCut_Model"><code>Fictiverse/Stable_Diffusion_PaperCut_Model</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>Stable Diffusion Inpainting</code></td>
      <td>Not applicable</td>
      <td>Not applicable</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-2-inpainting"><code>stabilityai/stable-diffusion-2-inpainting</code></a></li>
          <li><a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting"><code>stable-diffusion-v1-5/stable-diffusion-inpainting</code></a></li>
          <li><a href="https://huggingface.co/botp/stable-diffusion-v1-5-inpainting"><code>botp/stable-diffusion-v1-5-inpainting</code></a></li>
          <li><a href="https://huggingface.co/parlance/dreamlike-diffusion-1.0-inpainting"><code>parlance/dreamlike-diffusion-1.0-inpainting</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>Stable Diffusion XL</code></td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9"><code>stabilityai/stable-diffusion-xl-base-0.9</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"><code>stabilityai/stable-diffusion-xl-base-1.0</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/sdxl-turbo"><code>stabilityai/sdxl-turbo</code></a></li>
          <li><a href="https://huggingface.co/cagliostrolab/animagine-xl-4.0"><code>cagliostrolab/animagine-xl-4.0</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>Stable Diffusion XL Inpainting</code></td>
      <td>Not applicable</td>
      <td>Not applicable</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1"><code>diffusers/stable-diffusion-xl-1.0-inpainting-0.1</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>Stable Diffusion 3</code></td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Not supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers"><code>stabilityai/stable-diffusion-3-medium-diffusers</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"><code>stabilityai/stable-diffusion-3.5-medium</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-3.5-large"><code>stabilityai/stable-diffusion-3.5-large</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo"><code>stabilityai/stable-diffusion-3.5-large-turbo</code></a></li>
        </ul>
      </td>
      <tr>
      <td><code>Flux</code></td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Partially Supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/black-forest-labs/FLUX.1-schnell"><code>black-forest-labs/FLUX.1-schnell</code></a></li>
          <li><a href="https://huggingface.co/Freepik/flux.1-lite-8B-alpha"><code>Freepik/flux.1-lite-8B-alpha</code></a></li>
          <li><a href="https://huggingface.co/black-forest-labs/FLUX.1-dev"><code>black-forest-labs/FLUX.1-dev</code></a></li>
          <li><a href="https://huggingface.co/shuttleai/shuttle-3-diffusion"><code>shuttleai/shuttle-3-diffusion</code></a></li>
          <li><a href="https://huggingface.co/shuttleai/shuttle-3.1-aesthetic"><code>shuttleai/shuttle-3.1-aesthetic</code></a></li>
          <li><a href="https://huggingface.co/shuttleai/shuttle-jaguar"><code>shuttleai/shuttle-jaguar</code></a></li>
          <li><a href="https://huggingface.co/Shakker-Labs/AWPortrait-FL"><code>Shakker-Labs/AWPortrait-FL</code></a></li>
          <li><a href="https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev"><code>black-forest-labs/FLUX.1-Fill-dev</code></a></li>
        </ul>
      </td>
    </tr>
    </tr>
  </tbody>
</table>

## Visual language models

<table>
  <tbody style="vertical-align: top;">
    <tr>
      <th>Architecture</th>
      <th>Models</th>
      <th>LoRA support</th>
      <th>Example HuggingFace Models</th>
      <th>Notes</th>
    </tr>
    <tr>
      <td><code>InternVL2</code></td>
      <td>InternVL2</td>
      <td>Not supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/OpenGVLab/InternVL2-1B"><code>OpenGVLab/InternVL2-1B</code></a></li>
          <li><a href="https://huggingface.co/OpenGVLab/InternVL2-2B"><code>OpenGVLab/InternVL2-2B</code></a></li>
          <li><a href="https://huggingface.co/OpenGVLab/InternVL2-4B"><code>OpenGVLab/InternVL2-4B</code></a></li>
          <li><a href="https://huggingface.co/OpenGVLab/InternVL2-8B"><code>OpenGVLab/InternVL2-8B</code></a></li>
          <li><a href="https://huggingface.co/OpenGVLab/InternVL2_5-1B"><code>OpenGVLab/InternVL2_5-1B</code></a></li>
          <li><a href="https://huggingface.co/OpenGVLab/InternVL2_5-2B"><code>OpenGVLab/InternVL2_5-2B</code></a></li>
          <li><a href="https://huggingface.co/OpenGVLab/InternVL2_5-4B"><code>OpenGVLab/InternVL2_5-4B</code></a></li>
          <li><a href="https://huggingface.co/OpenGVLab/InternVL2_5-8B"><code>OpenGVLab/InternVL2_5-8B</code></a></li>
        </ul>
      </td>
      <td></td>
    </tr>
    <tr>
      <td><code>LLaVA</code></td>
      <td>LLaVA-v1.5</td>
      <td>Not supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/llava-hf/llava-1.5-7b-hf"><code>llava-hf/llava-1.5-7b-hf</code></a></li>
        </ul>
      </td>
      <td></td>
    </tr>
    <tr>
      <td><code>LLaVA-NeXT</code></td>
      <td>LLaVa-v1.6</td>
      <td>Not supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf"><code>llava-hf/llava-v1.6-mistral-7b-hf</code></a></li>
          <li><a href="https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf"><code>llava-hf/llava-v1.6-vicuna-7b-hf</code></a></li>
          <li><a href="https://huggingface.co/llava-hf/llama3-llava-next-8b-hf"><code>llava-hf/llama3-llava-next-8b-hf</code></a></li>
        </ul>
      </td>
      <td></td>
    </tr>
    <tr>
      <td><code>MiniCPMV</code></td>
      <td>MiniCPM-V-2_6</td>
      <td>Not supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/openbmb/MiniCPM-V-2_6"><code>openbmb/MiniCPM-V-2_6</code></a></li>
        </ul>
      </td>
      <td></td>
    </tr>
    <tr>
      <td><code>Phi3VForCausalLM</code></td>
      <td>phi3_v</td>
      <td>Not supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct"><code>microsoft/Phi-3-vision-128k-instruct</code></a></li>
          <li><a href="https://huggingface.co/microsoft/Phi-3.5-vision-instruct"><code>microsoft/Phi-3.5-vision-instruct</code></a></li>
        </ul>
      </td>
      <td>
          <li>GPU isn't supported</li>
          <li>These models' configs aren't consistent. It's required to override the default <code>eos_token_id</code> with the one from a tokenizer: <code>generation_config.set_eos_token_id(pipe.get_tokenizer().get_eos_token_id())</code>.</li>
      </td>
    </tr>
    <tr>
      <td><code>Qwen2-VL</code></td>
      <td>Qwen2-VL</td>
      <td>Not supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct"><code>Qwen/Qwen2-VL-2B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct"><code>Qwen/Qwen2-VL-7B-Instruct</code></a></li>
        </ul>
      </td>
      <td></td>
    </tr>
  </tbody>
</table>

## Whisper models

<table>
  <tbody style="vertical-align: top;">
    <tr>
      <th>Architecture</th>
      <th>Models</th>
      <th>LoRA support</th>
      <th>Example HuggingFace Models</th>
    </tr>
    <tr>
      <td rowspan=2><code>WhisperForConditionalGeneration</code></td>
      <td>Whisper</td>
      <td>Not supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/openai/whisper-tiny"><code>openai/whisper-tiny</code></a></li>
          <li><a href="https://huggingface.co/openai/whisper-tiny.en"><code>openai/whisper-tiny.en</code></a></li>
          <li><a href="https://huggingface.co/openai/whisper-base"><code>openai/whisper-base</code></a></li>
          <li><a href="https://huggingface.co/openai/whisper-base.en"><code>openai/whisper-base.en</code></a></li>
          <li><a href="https://huggingface.co/openai/whisper-small"><code>openai/whisper-small</code></a></li>
          <li><a href="https://huggingface.co/openai/whisper-small.en"><code>openai/whisper-small.en</code></a></li>
          <li><a href="https://huggingface.co/openai/whisper-medium"><code>openai/whisper-medium</code></a></li>
          <li><a href="https://huggingface.co/openai/whisper-medium.en"><code>openai/whisper-medium.en</code></a></li>
          <li><a href="https://huggingface.co/openai/whisper-large-v3"><code>openai/whisper-large-v3</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Distil-Whisper</td>
      <td>Not supported</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/distil-whisper/distil-small.en"><code>distil-whisper/distil-small.en</code></a></li>
          <li><a href="https://huggingface.co/distil-whisper/distil-medium.en"><code>distil-whisper/distil-medium.en</code></a></li>
          <li><a href="https://huggingface.co/distil-whisper/distil-large-v3"><code>distil-whisper/distil-large-v3</code></a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>
Some models may require access request submission on the Hugging Face page to be downloaded.

If https://huggingface.co/ is down, the conversion step won't be able to download the models.

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
      <td><code>ChatGLMModel</code></td>
      <td>ChatGLM</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/THUDM/chatglm2-6b"><code>THUDM/chatglm2-6b</code></a></li>
          <li><a href="https://huggingface.co/THUDM/chatglm3-6b"><code>THUDM/chatglm3-6b</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>GemmaForCausalLM</code></td>
      <td>Gemma</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/google/gemma-2b-it"><code>google/gemma-2b-it</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td rowspan="2"><code>GPTNeoXForCausalLM</code></td>
      <td>Dolly</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/databricks/dolly-v2-3b"><code>databricks/dolly-v2-3b</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <!-- <td><code>GPTNeoXForCausalLM</code></td> -->
      <td> RedPajama</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/ikala/redpajama-3b-chat"><code>ikala/redpajama-3b-chat</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td rowspan="4" vertical-align="top"><code>LlamaForCausalLM</code></td>
      <td>Llama 3</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B"><code>meta-llama/Meta-Llama-3-8B</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"><code>meta-llama/Meta-Llama-3-8B-Instruct</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B"><code>meta-llama/Meta-Llama-3-70B</code></a></li>
          <li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct"><code>meta-llama/Meta-Llama-3-70B-Instruct</code></a></li>
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
          <li><a href="https://huggingface.co/microsoft/Llama2-7b-WhoIsHarryPotter"><code>microsoft/Llama2-7b-WhoIsHarryPotter</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <!-- <td><code>LlamaForCausalLM</code></td> -->
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
      <!-- <td><code>LlamaForCausalLM</code></td> -->
      <td>TinyLlama</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0"><code>TinyLlama/TinyLlama-1.1B-Chat-v1.0</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td rowspan="3"><code>MistralForCausalLM</code></td>
      <td>Mistral</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/mistralai/Mistral-7B-v0.1"><code>mistralai/Mistral-7B-v0.1</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <!-- <td><code>MistralForCausalLM</code></td> -->
      <td>Notus</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/argilla/notus-7b-v1"><code>argilla/notus-7b-v1</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <!-- <td><code>MistralForCausalLM</code></td> -->
      <td>Zephyr </td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-beta"><code>HuggingFaceH4/zephyr-7b-beta</code></a></li>
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
      <td><code>QWenLMHeadModel</code></td>
      <td>Qwen</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/Qwen/Qwen-7B-Chat"><code>Qwen/Qwen-7B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen-7B-Chat-Int4"><code>Qwen/Qwen-7B-Chat-Int4</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen1.5-7B-Chat"><code>Qwen/Qwen1.5-7B-Chat</code></a></li>
          <li><a href="https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GPTQ-Int4"><code>Qwen/Qwen1.5-7B-Chat-GPTQ-Int4</code></a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


The pipeline can work with other similar topologies produced by `optimum-intel` with the same model signature. The model is required to have the following inputs after the conversion:
1. `input_ids` contains the tokens.
2. `attention_mask` is filled with `1`.
3. `beam_idx` selects beams.
4. `position_ids` (optional) encodes a position of currently generating token in the sequence and a single `logits` output.

> [!NOTE]
> Models should belong to the same family and have the same tokenizers.

## Visual language models

<table>
  <tbody style="vertical-align: top;">
    <tr>
      <th>Architecture</th>
      <th>Models</th>
      <th>Example HuggingFace Models</th>
    </tr>
    <tr>
      <td><code>MiniCPM-V-2_6</code></td>
      <td>MiniCPMV</td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/openbmb/MiniCPM-V-2_6"><code>openbmb/MiniCPM-V-2_6</code></a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## Text 2 image models

<table>
  <tbody style="vertical-align: top;">
    <tr>
      <th>Architecture</th>
      <th>Example HuggingFace Models</th>
    </tr>
    <tr>
      <td><code>Stable Diffusion</code></td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/botp/stable-diffusion-v1-5"><code>botp/stable-diffusion-v1-5</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-2"><code>stabilityai/stable-diffusion-2</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-2-1"><code>stabilityai/stable-diffusion-2-1</code></a></li>
          <li><a href="https://huggingface.co/dreamlike-art/dreamlike-anime-1.0"><code>dreamlike-art/dreamlike-anime-1.0</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>Latent Consistency Model</code></td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7"><code>SimianLuo/LCM_Dreamshaper_v7</code></a></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>Stable Diffusion XL</code></td>
      <td>
        <ul>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9"><code>stabilityai/stable-diffusion-xl-base-0.9</code></a></li>
          <li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"><code>stabilityai/stable-diffusion-xl-base-1.0</code></a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

Some models may require access request submission on the Hugging Face page to be downloaded.

If https://huggingface.co/ is down, the conversion step won't be able to download the models.

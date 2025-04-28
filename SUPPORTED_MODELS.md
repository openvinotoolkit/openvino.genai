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
          <li><a href="https://huggingface.co/google/gemma-7b-it"><code>google/gemma-7b-it</code></a></li>
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

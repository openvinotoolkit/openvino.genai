# **OpenVINO GenAI Agentic Toolkit Python Samples**

These samples showcase the integration of OpenVINO™ GenAI with the LangChain ecosystem. The provided wrapper and examples demonstrate how to leverage OpenVINO-accelerated Large Language Models (LLMs) within autonomous agent loops, RAG pipelines, and tool-use workflows.

The applications don't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU or NPU, or modify the toolset provided to the agent.

## **Table of Contents**

1. [Download and Convert the Model and Tokenizers](https://www.google.com/search?q=%23download-and-convert-the-model-and-tokenizers)  
2. [Sample Descriptions](https://www.google.com/search?q=%23sample-descriptions)  
3. [Key Architectural Features](https://www.google.com/search?q=%23key-architectural-features)  
4. [Troubleshooting](https://www.google.com/search?q=%23troubleshooting)  
5. [Support and Contribution](https://www.google.com/search?q=%23support-and-contribution)

## **Download and convert the model and tokenizers**

The \--upgrade-strategy eager option is needed to ensure optimum-intel is upgraded to the latest version.

Install [../../export-requirements.txt](https://www.google.com/search?q=../../export-requirements.txt) if model conversion is required.

pip install \--upgrade-strategy eager \-r ../../export-requirements.txt

Then, run the export with Optimum CLI. For agentic workflows, instruct-tuned models (e.g., Llama-3.2) are highly recommended:

optimum-cli export openvino \--model meta-llama/Llama-3.2-3B-Instruct \--weight-format fp16 models/llama-3.2-3b/FP16

If a converted model in OpenVINO IR format is already available in the collection of [OpenVINO optimized LLMs](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd) on Hugging Face, it can be downloaded directly via huggingface-cli.

pip install huggingface-hub  
huggingface-cli download OpenVINO/Llama-3.2-3B-Instruct-int4-ov \--local-dir models/llama-3.2-3b/INT4

## **Sample Descriptions**

### **Common information**

Follow [Get Started with Samples](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/get-started-demos.html) to get common information about OpenVINO samples.

GPUs usually provide better performance compared to CPUs. Modify the source code in the scripts to change the device parameter for inference to the GPU.

Install [../../deployment-requirements.txt](https://www.google.com/search?q=../../deployment-requirements.txt) and the sample-specific requirements to run these files:

pip install \--upgrade-strategy eager \-r ../../deployment-requirements.txt  
pip install langchain langchain-community pydantic

### **1\. LangChain LLM Wrapper (llm\_wrapper.py)**

* **Description:** Provides the foundational OpenVINOChatModel, a custom class implementing LangChain's BaseChatModel interface. It handles model loading via ov\_genai.LLMPipeline, chat history conversions, and exposes generation configurations.  
* **Main Feature:** Native C++ stop-strings mapping and thread-safe streaming integration.  
* **Usage:** This file acts as a module to be imported into your LangChain applications.

### **2\. ReAct Agent Sample (example\_agent.py)**

* **Description:** A complete ReAct (Reasoning and Acting) agent pipeline built from scratch using langchain\_core messages. It bypasses deprecated legacy components like AgentExecutor to guarantee forward compatibility. It binds OpenVINO hardware-accelerated local inference directly to a Python calculator tool.  
* **Main Feature:** Real-time reasoning-action loops utilizing tool calls and regex parsing.  
* **Run Command:**  
  python example\_agent.py \[model\_dir\]

* **Example Execution:**  
  Thought: I need to calculate 25 \* 4 first, and then add 10 to the result.  
  Action: calculator  
  Action Input: 25 \* 4 \+ 10  
  Observation: 110  
  Thought: I now know the final answer.  
  Final Answer: The result of 25 \* 4 \+ 10 is 110\.

## **Key Architectural Features**

\[\!NOTE\]

This implementation is designed for production stability. It utilizes **Pydantic V2 Private Attributes** to shield C++ memory pointers and a **Thread-Safe Queue Bridge** to manage the GIL during high-performance inference.

| Feature | Implementation Detail | Benefit |
| :---- | :---- | :---- |
| **Pydantic V2 Safety** | PrivateAttr & ConfigDict isolation | Prevents Pydantic from attempting to validate C++ memory pointers, avoiding crashes. |
| **Native Stop-Strings** | Direct mapping to GenerationConfig.stop\_strings | Zero-latency halting at the C++ level; critical for ReAct agents to stop at "Observation:". |
| **Thread-Safe Streaming** | Decoupled Push-Pull Queue (Daemon Thread) | Prevents GIL deadlocks while bridging OpenVINO's C++ callback to Python's yield generator. |
| **Exception Propagation** | Cross-thread try...except capture | Ensures C++ runtime errors (OOM/Runtime) are re-raised in Python for graceful handling. |

## **Running Unit Tests**

The test suite validates the architecture without requiring a GPU or large model binaries by dynamically injecting the module and mocking the C++ pipeline.

python \-m unittest ../../tests/python\_tests/agentic/test\_llm\_wrapper.py

## **Troubleshooting**

### **Unicode characters encoding error on Windows**

Example error:

UnicodeEncodeError: 'charmap' codec can't encode character '\\u25aa' in position 0: character maps to \<undefined\>

If you encounter this error when the agent is printing text to the Windows console, it is likely due to the default Windows encoding not supporting certain Unicode characters generated by the LLM. To resolve this:

1. Enable Unicode characters for Windows cmd \- open Region settings from Control panel. Administrative-\>Change system locale-\>Beta: Use Unicode UTF-8 for worldwide language support-\>OK. Reboot.  
2. Enable UTF-8 mode by setting environment variable PYTHONIOENCODING="utf8".

## **Support and Contribution**

* For troubleshooting, consult the [OpenVINO documentation](https://docs.openvino.ai).  
* To report issues or contribute, visit the [GitHub repository](https://github.com/openvinotoolkit/openvino.genai).
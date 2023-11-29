# Notes
## chatglm2-6b - AttributeError: can't set attribute
Download chatglm2-6b from hugginface, convert to OpenVINO IR files and run with benchmark.py, the following error may occur：
```bash
AttributeError: can't set attribute
```
Reproduced with https://huggingface.co/THUDM/chatglm2-6b 7fabe56db91e085c9c027f56f1c654d137bdba40 <br />
As on https://huggingface.co/THUDM/chatglm2-6b/discussions/99 <br />
Solution: update `tokenization_chatglm.py` as following: <br />
```Python
          self.vocab_file = vocab_file
          self.tokenizer = SPTokenizer(vocab_file)
 +        kwargs.pop("eos_token", None)
 +        kwargs.pop("pad_token", None)
 +        kwargs.pop("unk_token", None)
          self.special_tokens = {
              "<bos>": self.tokenizer.bos_id,
              "<eos>": self.tokenizer.eos_id,
```              

> The solution works for chatglm3-6b as well.

## Qwen-7B-Chat-Int4 - Torch not compiled with CUDA enabled
Convert Qwen-7B-Chat-Int4 to OpenVINO IR files run with convert.py, the following error may occur：
```bash
raise AssertionError("Torch not compiled with CUDA enabled")
```
Solution: update `modeling_qwen.py` as following: <br />
```Python
-SUPPORT_CUDA = torch.cuda.is_available()
+SUPPORT_CUDA = False
 SUPPORT_BF16 = SUPPORT_CUDA and torch.cuda.is_bf16_supported()
 ```
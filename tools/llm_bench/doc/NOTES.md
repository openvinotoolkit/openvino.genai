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
Reproduced with https://huggingface.co/Qwen/Qwen-7B-Chat-Int4 8750247cc50f2a7bb84bef322f7707159b700723 <br />
Solution: update `modeling_qwen.py` as following: <br />
```Python
-SUPPORT_CUDA = torch.cuda.is_available()
+SUPPORT_CUDA = False
 SUPPORT_BF16 = SUPPORT_CUDA and torch.cuda.is_bf16_supported()
 ```

## Baichuan2-7B-Chat - AttributeError: 'BaichuanTokenizer' object has no attribute 'sp_model'
Convert Baichuan2-7B-Chat to OpenVINO IR files run with convert.py, the following error may occur：
```bash
AttributeError: 'BaichuanTokenizer' object has no attribute 'sp_model'
```
Reproduced with https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat 84603cde5ebffb6084e476cfaeceaf0b8b91fe54 <br />
Reference to https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/discussions/2 <br />
Solution: update `tokenization_baichuan.py` as following: <br />
```Python
         eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
         unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
         pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
+        self.vocab_file = vocab_file
+        self.add_bos_token = add_bos_token
+        self.add_eos_token = add_eos_token
+        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
+        self.sp_model.Load(vocab_file)
         super().__init__(
             bos_token=bos_token,
             eos_token=eos_token,
             clean_up_tokenization_spaces=clean_up_tokenization_spaces,
             **kwargs,
         )
-        self.vocab_file = vocab_file
-        self.add_bos_token = add_bos_token
-        self.add_eos_token = add_eos_token
-        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
-        self.sp_model.Load(vocab_file)
```

## CompressWeights Mode INT4 - ConnectionError: Couldn't reach 'wikitext' on the Hub (SSLError)
Download LLM from hugginface, convert to OpenVINO IR files and run with convert.py and CompressWeights Mode to INT4, the following error may occur：
```bash
raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({type(e)._name_})")
ConnectionError: Couldn't reach 'wikitext' on the Hub (SSLError)
```
root cause: The wikitext data set was not downloaded correctly, or the Hugging Face Hub network could not be connected normally. <br />
Solution: <br />
Refer to https://huggingface.co/docs/datasets/loading#arrow , copy wikitext data set to ~/.cache/huggingface/datasets/ folder, set the environment variable HF_DATASETS_OFFLINE to 1.
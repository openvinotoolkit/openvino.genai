from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from openvino_tokenizers import convert_tokenizer
from openvino_tokenizers import pack_strings, unpack_strings

from transformers import AutoTokenizer
from openvino import serialize, Core, Type

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, default="stable-diffusion-v1-5/tokenizer",
                        help="Model id of a pretrained Hugging Face tokenizer on the Hub or local directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True, default="models/tokenizer",
                        help="Save directory of converted OpenVINO Model and configurations")
    parser.add_argument("-p", "--prompt", type=str, required=False, 
                        default="cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting",
                        help="A test prompt to test converted OpenVINO tokenizers against native HuggingFace")
    parser.add_argument("--with_detokenizer", type=bool, required=False, default=True,
                        help="Whether save tokenzier decoder")
    args = parser.parse_args()

    print("Load HF Tokenizer ...")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print("Convert HF Tokenizer to OV Tokenizer ...")
    ov_tokenizer_encoder, ov_tokenizer_decoder = convert_tokenizer(hf_tokenizer, with_detokenizer=args.with_detokenizer, tokenizer_output_type=Type.i32)

    print(f"Serialize OV Tokenizer to {args.output_dir} ...")

    serialize(ov_tokenizer_encoder, Path(args.output_dir) / "tokenizer_encoder.xml", Path(args.output_dir) / "tokenizer_encoder.bin")
    serialize(ov_tokenizer_decoder, Path(args.output_dir) / "tokenizer_decoder.xml", Path(args.output_dir) / "tokenizer_decoder.bin")

    print(f"Test tokenizer with prompt: {args.prompt}")
    hf_tokens = hf_tokenizer.encode(args.prompt)
    print(f"HF Tokenizer encode results {hf_tokens}")
    hf_decode_str = hf_tokenizer.decode(hf_tokens, skip_special_tokens=True)
    print(f"HF Tokenizer decode results {hf_decode_str}")

    core = Core()
    compiled_ov_tokenizer_encoder = core.compile_model(ov_tokenizer_encoder)
    compiled_ov_tokenizer_decoder = core.compile_model(ov_tokenizer_decoder)

    prompt_uint8 = pack_strings([args.prompt])
    input_ids = compiled_ov_tokenizer_encoder(prompt_uint8)["input_ids"]
    print(f"OV Tokenizer results input ids: {input_ids} with shape {input_ids.shape}")

    decoder_outputs = compiled_ov_tokenizer_decoder([input_ids])
    output_text = unpack_strings(decoder_outputs["string_output"])
    print("OV deTokenizer output_text: ", output_text)


if __name__ == "__main__":
    main()

from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from tokenizer.convert_tokenizer import convert_tokenizer, connect_models
from tokenizer.str_pack import pack_strings, unpack_strings

from transformers import AutoTokenizer
from openvino.runtime import serialize, Core

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, default="stable-diffusion-v1-5/tokenizer",
                        help="Model id of a pretrained Hugging Face tokenizer on the Hub or local directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True, default="models/tokenizer",
                        help="Save directory of converted OpenVINO Model and configurations")
    parser.add_argument("-p", "--prompt", type=str, required=False, 
                        default="cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting",
                        help="Save directory of converted OpenVINO Model and configurations")
    parser.add_argument("--with_decoder", type=bool, required=False, default=True,
                        help="Whether save tokenzier decoder")
    args = parser.parse_args()

    print("Load HF Tokenizer...")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print("Convert HF Tokenizer to OV Tokenizer...")
    ov_tokenizer_encoder, ov_tokenizer_decoder = convert_tokenizer(hf_tokenizer, with_decoder=args.with_decoder)
    #print("ov_tokenizer_encoder: ", ov_tokenizer_encoder)
    #print("ov_tokenizer_decoder: ", ov_tokenizer_decoder)

    print(f"Serialize OV Tokenizer to {args.output_dir}")
    
    serialize(ov_tokenizer_encoder, Path(args.output_dir) / "tokenizer_encoder.xml", Path(args.output_dir) / "tokenizer_encoder.bin")
    serialize(ov_tokenizer_decoder, Path(args.output_dir) / "tokenizer_decoder.xml", Path(args.output_dir) / "tokenizer_decoder.bin")

    print(f"Test tokenzier with prompt: {args.prompt}")
    hf_tokens = hf_tokenizer.encode(args.prompt)
    print(f"HF Tokenizer encode results {hf_tokens}")
    hf_decode_str = hf_tokenizer.decode(hf_tokens)
    print(f"HF Tokenizer decode results {hf_decode_str}")

    core = Core()
    compiled_ov_tokenizer_encoder = core.compile_model(ov_tokenizer_encoder)
    compiled_ov_tokenizer_decoder = core.compile_model(ov_tokenizer_decoder)

    prompt_uint8 = pack_strings([args.prompt])
    #print(f"Pack string as byte array: {prompt_uint8}")
    encoder_outputs = compiled_ov_tokenizer_encoder(prompt_uint8)
    #print("outputs: ", encoder_outputs)
    input_ids = encoder_outputs["input_ids"]
    attention_mask = encoder_outputs["attention_mask"]
    print(f"OV Tokenizer results input ids: {input_ids} with shape {input_ids.shape}")
    print(f"OV Tokenizer results attention_mask: {attention_mask} with shape {attention_mask.shape}")
    """
    logits = np.array([input_ids])
    print(f"logits for tokenizer decoder: {logits}")
    decoder_outputs = compiled_ov_tokenizer_decoder(logits)
    output_bytes = decoder_outputs["string_output"]
    print("output_bytes: ", output_bytes)
    output_text = unpack_strings(output_bytes)
    print("output_text: ", output_text)

    expected_output = unpack_strings(prompt_uint8)
    print(f"expected_output: {expected_output}")
    """


if __name__ == "__main__":
    main()

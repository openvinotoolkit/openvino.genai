from argparse import ArgumentParser
from pathlib import Path
from openvino_tokenizers import convert_tokenizer

from transformers import AutoTokenizer
from openvino import serialize, Type

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, default="stable-diffusion-v1-5/tokenizer",
                        help="Model id of a pretrained Hugging Face tokenizer on the Hub or local directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True, default="models/tokenizer",
                        help="Save directory of converted OpenVINO Model and configurations")
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


if __name__ == "__main__":
    main()

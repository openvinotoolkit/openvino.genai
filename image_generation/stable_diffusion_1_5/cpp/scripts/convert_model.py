from pathlib import Path
import argparse
from optimum.intel.openvino import OVStableDiffusionPipeline
from openvino import Type, save_model
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-b', '--batch', type = int, default = 1, required = True,
                      help='Required. batch_size for solving single/multiple prompt->image generation.')
    args.add_argument('-t', '--type', type = str, default = "FP32", required = True,
                      help='Required. data type, FP32, FP16, and compressed type INT8.')
    args.add_argument('-dyn', '--dynamic', type = bool, default = False, required = False,
                      help='Specify the model input shape to use dynamic shape.')
    args.add_argument('-sd','--sd_weights', type = str, default="", required = True,
                      help='Specify the path of stable diffusion model')
    return parser.parse_args()

args = parse_args()

load_in_8bit = True if args.type == "INT8" else False
output_path = Path(args.sd_weights) / (args.type + ("_dyn" if args.dynamic else "_static"))

# convert SD models to IR

model = OVStableDiffusionPipeline.from_pretrained(args.sd_weights, trust_remote_code=True, export=True, compile=False, load_in_8bit=load_in_8bit)
if args.type == "FP16":
    model.half()
if not args.dynamic:
    model.reshape(args.batch, 512, 512, 1)

model.save_pretrained(output_path)

# convert tokenizer

tokenizer_path = output_path / "tokenizer"
hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
ov_tokenizer_encoder = convert_tokenizer(hf_tokenizer, tokenizer_output_type=Type.i32)

save_model(ov_tokenizer_encoder, tokenizer_path / "openvino_tokenizer.xml", compress_to_fp16=False)

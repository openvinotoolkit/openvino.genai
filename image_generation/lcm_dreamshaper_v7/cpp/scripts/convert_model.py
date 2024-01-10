from pathlib import Path
import argparse
from optimum.intel.openvino import OVLatentConsistencyModelPipeline
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer
from openvino import Type, save_model


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-t', '--type', type = str, default = "FP32", required = True,
                      help='Required. data type, FP32, FP16.')
    args.add_argument('-lcm','--lcm_weights', type = str, default="SimianLuo/LCM_Dreamshaper_v7", required = True,
                      help='Specify the path of lcm model')
    return parser.parse_args()

args = parse_args()
output_path = Path(args.lcm_weights) / (args.type + "_static")

###convert LCM model to IR

model = OVLatentConsistencyModelPipeline.from_pretrained(args.lcm_weights, trust_remote_code=True, export=True, compile=False)
if args.type == "FP16":
    model.half()

model.reshape(1, 512, 512, 1)

model.compile()
model.save_pretrained(output_path)

# convert tokenizer

tokenizer_path = output_path / "tokenizer"
hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
ov_tokenizer_encoder = convert_tokenizer(hf_tokenizer, tokenizer_output_type=Type.i32)

save_model(ov_tokenizer_encoder, tokenizer_path / "openvino_tokenizer.xml", compress_to_fp16=False)

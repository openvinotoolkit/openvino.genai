from pathlib import Path
import argparse
from optimum.intel.openvino import OVLatentConsistencyModelPipeline


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-t', '--type', type = str, default = "FP32", required = True,
                      help='Required. data type, FP32, FP16.')
    args.add_argument('-lcm','--lcm_weights', type = str, default="SimianLuo/LCM_Dreamshaper_v7", required = True,
                      help='Specify the path of lcm model')
    # fmt: on
    return parser.parse_args()

args = parse_args()

###convert LCM model to IR

model = OVLatentConsistencyModelPipeline.from_pretrained(args.lcm_weights,trust_remote_code=True, export=True, compile=False)
if args.type == "FP16":
    model.half()
    model.reshape(1,512,512,1)
model.compile()
model.save_pretrained(str(Path(args.lcm_weights) / args.type) + "_static")

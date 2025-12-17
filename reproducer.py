import datasets
import logging
from transformers import set_seed

logging.basicConfig(level=logging.DEBUG)

DATASET_NAME = "paint-by-inpaint/PIPE"
NUM_SAMPLES = 10
set_seed(42)
default_dataset = datasets.load_dataset(
    DATASET_NAME, split="test", streaming=True
).filter(lambda example: example["Instruction_VLM-LLM"] != "").take(NUM_SAMPLES)

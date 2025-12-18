import datasets
import logging
import pprint
import ssl
import socket
from transformers import set_seed

logging.basicConfig(level=logging.DEBUG)

context = ssl.create_default_context()

def get_host_cert_info(fqdn):
    conn = context.wrap_socket(socket.socket(socket.AF_INET),
                            server_hostname=fqdn)
    conn.connect((fqdn, 443))
    cert = conn.getpeercert()
    pprint.pprint(cert)

# Get more information about certs various Hugging Face endpoints use
get_host_cert_info("huggingface.co")
get_host_cert_info("cdn-lfs.huggingface.co")
get_host_cert_info("datasets-server.huggingface.co")

# Code from the original test
DATASET_NAME = "paint-by-inpaint/PIPE"
NUM_SAMPLES = 10
set_seed(42)
default_dataset = datasets.load_dataset(
    DATASET_NAME, split="test", streaming=True
).filter(lambda example: example["Instruction_VLM-LLM"] != "").take(NUM_SAMPLES)

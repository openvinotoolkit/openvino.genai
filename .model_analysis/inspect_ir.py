from openvino import Core
from pathlib import Path
import sys

core = Core()
for xml in sorted(Path(sys.argv[1]).glob("openvino_*.xml")):
    if "tokenizer" in xml.name or "detokenizer" in xml.name:
        print(f"\n=== {xml.name} === SKIPPED (tokenizer)")
        continue
    m = core.read_model(xml)
    print(f"\n=== {xml.name} ===")
    for i in m.inputs:
        print(f"  IN  {i.any_name}: {i.partial_shape} {i.element_type}")
    for o in m.outputs:
        print(f"  OUT {o.any_name}: {o.partial_shape} {o.element_type}")

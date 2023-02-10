
from paddle_annotation import get_pinyins

label_input_path = ""
label_output_path = ""

with open(label_output_path, "w") as f:
    for line in open(label_input_path).readlines():
        id_, text = line.strip().split("|")
        assert text.startswith("[ZH]") and text.endswith("[ZH]"), text
        text = text.replace("[ZH]", "")
        pinyins =  " ".join(get_pinyins([text]))
        f.write(f"{id_}|[P]{pinyins}[P]\n")

import os
from tqdm import tqdm
punc = ['！', '？', "…", "，", "。", '!', '?', "…", ",", ".", " "]
from paddle_annotation import annotation_dataset
# xmzhu/whisper-small-zh
if not os.path.exists("labels"):
    os.mkdir("labels")

for spk in os.listdir("output"):
    if os.path.isdir(f"output/{spk}") and os.path.exists(f"output/{spk}/zh"):
        name = spk
        os.makedirs("labels/zh", exist_ok=True)
        annotation_dataset(f"output/{spk}", f"labels/zh/{spk}_label.txt", use_ffmpeg=False)



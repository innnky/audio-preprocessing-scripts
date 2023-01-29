import zhconv
import os
import whisper
from tqdm import tqdm
punc = ['！', '？', "…", "，", "。", '!', '?', "…", ",", ".", " "]

def is_chinese(char):
    if '\u4e00' <= char <= '\u9fa5':
        return True
    else:
        return False
def is_all_chinese(text):
    if len(text) == 0:
        return False
    for ch in text:
        if ch in punc or is_chinese(ch):
            continue
        else:
            return False
    return True

model = whisper.load_model("large-v2")

if not os.path.exists("labels"):
    os.mkdir("labels")

for spk in os.listdir("output"):
    if os.path.isdir(f"output/{spk}") and os.path.exists(f"output/{spk}/zh"):
        name = spk
        wav_paths = [f"output/{name}/{i}" for i in sorted(os.listdir(f"output/{name}")) if i.endswith("wav")]
        os.system(f"touch labels/{spk}_label.txt")
        fo = open(f"labels/{spk}_label.txt", "w")
        for path in tqdm(wav_paths):
            result = model.transcribe(path)
            txt = zhconv.convert(result["text"], "zh-cn")
            if not is_all_chinese(txt):
                print("not chinese", txt)
                continue
            fo.write("{}|{}\n".format(path, txt))
            fo.flush()
            print("{}|{}".format(path, txt))
        fo.close()



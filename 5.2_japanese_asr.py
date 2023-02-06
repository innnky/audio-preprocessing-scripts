import langid
import os
import whisper
from tqdm import tqdm

chinaTab = ['：', '；', '，', '。', '！', '？', '【', '】', '“', '（', '）', '%', '#', '@', '&', "‘", ' ', '\n', '”', "—", "·",
            '、', '...']
punc = ['！', '？', "…", "，", "。", '!', '?', "…", ",", ".", " "]+chinaTab

import re

def is_japanese(char):
    return re.search(r'[\u3005\u3040-\u30ff\u4e00-\u9fff\uff41-\uff5a\uff66-\uff9d]', char) is not None

def is_all_japanese(text):
    if len(text) == 0:
        return False
    for ch in text:
        if ch in punc or is_japanese(ch):
            continue
        else:
            return False
    return True
# List of (symbol, Japanese) pairs for marks:
_symbols_to_japanese = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('％', 'パーセント')
]]
def str_replace( data):
    chinaTab = ['：', '；', '，', '。', '！', '？', '【', '】', '“', '（', '）', '%', '#', '@', '&', "‘", ' ', '\n', '”',"—", "·",'、','...']
    englishTab = [':', ';', ',', '.', '!', '?', '[', ']', '"', '(', ')', '%', '#', '@', '&', "'", ' ', '', '"', "-", "-", ",","…"]
    for index in range(len(chinaTab)):
        if chinaTab[index] in data:
            data = data.replace(chinaTab[index], englishTab[index])
    return data
def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text

model = whisper.load_model("medium")

if not os.path.exists("labels"):
    os.mkdir("labels")

for spk in os.listdir("output"):
    if os.path.isdir(f"output/{spk}") and os.path.exists(f"output/{spk}/ja"):
        name = spk
        wav_paths = [f"output/{name}/{i}" for i in sorted(os.listdir(f"output/{name}")) if i.endswith("wav")]
        os.system(f"touch labels/{spk}_label.txt")
        fo = open(f"labels/{spk}_label.txt", "w")
        for path in tqdm(wav_paths):
            result = model.transcribe(path)
            text = ''
            for seg in result["segments"]:
                text += seg["text"]
                if text!="" and text[-1] not in punc:
                    text += ","
            if text == '':
                print("null")
                continue
            if text[-1] == ",":
                text = text[:-1] + "."
            txt = str_replace(text)
            txt = symbols_to_japanese(txt)
            if not is_all_japanese(txt):
                print("not japanese", txt)
                continue
            if langid.classify(text)[0]!="ja":
                print("langid not japanese", txt)
                continue
            fo.write("{}|{}\n".format(path, txt))
            fo.flush()
            print("{}|{}".format(path, txt))
        fo.close()



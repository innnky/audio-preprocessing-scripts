import langid
import os

from tqdm import tqdm
import librosa
import re
timeout_threshold = 3

import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
  "automatic-speech-recognition",
  model="vumichien/whisper-small-ja",
  chunk_length_s=30,
  device=device,
)


chinaTab = ['：', '；', '，', '。', '！', '？', '【', '】', '“', '（', '）', '%', '#', '@', '&', "‘", ' ', '\n', '”', "—", "·",
            '、', '...']
punc = ['！', '？', "…", "，", "。", '!', '?', "…", ",", ".", " ",'-']+chinaTab
# List of (symbol, Japanese) pairs for marks:
_symbols_to_japanese = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('％', 'パーセント')
]]
import signal
import re

def check_repeated_substring(string):
    pattern = re.compile(r"(.+?)\1{5,}")
    match = pattern.search(string)
    return match is not None

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("The function call timed out.")

def generate_with_timeout(pipe, sample, timeout):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        return pipe(sample.copy())["text"]
    except TimeoutException as e:
        return None
    finally:
        signal.alarm(0)

def transcribe(path):
    # path = "/Volumes/Extend/AI/JETS/dataset/nyaru/84.wav"

    # path = "/content/audio-preprocessing-scripts/output/koni/000001.wav"
    wav, sr = librosa.load(path, 16000)

    sample = {
        "array": wav,
        "sampling_rate": sr

    }
    text = generate_with_timeout(pipe, sample, timeout_threshold)
    if text is None:
        return "timeout!"
    return text.replace("\n","")

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

def str_replace( data):
    chinaTab = ['：', '；', '，', '。', '！', '？', '【', '】', '“', '（', '）', '%', '#', '@', '&', "‘", ' ', '\n', '”',"—", "·",'、','...']
    englishTab =[':', ';',  ',', '.',  '!',  '?',  '[', ']',  '"',  '(', ')', '%', '#', '@', '&', "'",  ' ', '', '"', "-", "-", ",",  "…"]
    for index in range(len(chinaTab)):
        if chinaTab[index] in data:
            data = data.replace(chinaTab[index], englishTab[index])
    return data
def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text
#
if not os.path.exists("labels"):
    os.mkdir("labels")

for spk in os.listdir("output"):
    if os.path.isdir(f"output/{spk}") and os.path.exists(f"output/{spk}/ja"):
        name = spk
        os.makedirs("labels/ja", exist_ok=True)
        wav_paths = [f"output/{name}/{i}" for i in sorted(os.listdir(f"output/{name}")) if i.endswith("wav")]
        fo = open(f"labels/ja/{spk}_label.txt", "w")
        for path in tqdm(wav_paths):
            text = transcribe(path)
            print("raw:\n", text)
            txt = str_replace(text)
            txt = symbols_to_japanese(txt)
            if txt == "":
                print("null", txt)
                continue
            if check_repeated_substring(txt):
                print("repeate", txt)
                continue
            if len(txt) >150:
                print("too long", txt)
                continue
            if not is_all_japanese(txt):
                print("not japanese", txt)
                continue
            if langid.classify(text)[0]!="ja":
                print("langid not japanese", txt)
                continue
            fo.write("{}|[JA]{}[JA]\n".format(path, txt))
            fo.flush()
            print(" {}|{}".format(txt,path))
        fo.close()



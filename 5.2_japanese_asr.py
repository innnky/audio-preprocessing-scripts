import langid
import os
from tqdm import tqdm
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import re
timeout_threshold = 15
processor = WhisperProcessor.from_pretrained("vumichien/whisper-small-ja")
model = WhisperForConditionalGeneration.from_pretrained("vumichien/whisper-small-ja")
model.config.forced_decoder_ids = None
chinaTab = ['：', '；', '，', '。', '！', '？', '【', '】', '“', '（', '）', '%', '#', '@', '&', "‘", ' ', '\n', '”', "—", "·",
            '、', '...']
punc = ['！', '？', "…", "，", "。", '!', '?', "…", ",", ".", " ",'-']+chinaTab
# List of (symbol, Japanese) pairs for marks:
_symbols_to_japanese = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('％', 'パーセント')
]]
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("The function call timed out.")

def generate_with_timeout(model, input_features, timeout):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        return model.generate(input_features)
    except TimeoutException as e:
        return None
    finally:
        signal.alarm(0)

def transcribe(path):
    # path = "/Volumes/Extend/AI/JETS/dataset/nyaru/84.wav"

    sample, sr = librosa.load(path, 16000)
    input_features = processor(sample, sampling_rate=sr, return_tensors="pt").input_features
    predicted_ids = generate_with_timeout(model, input_features, timeout_threshold)
    if predicted_ids is None:
        return "timeout!"
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    assert len(transcription) == 1
    return transcription[0].replace("\n","")

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
        wav_paths = [f"output/{name}/{i}" for i in sorted(os.listdir(f"output/{name}")) if i.endswith("wav")]
        os.system(f"touch labels/{spk}_label.txt")
        fo = open(f"labels/{spk}_label.txt", "w")
        for path in tqdm(wav_paths):
            text = transcribe(path)
            print("raw:\n", text)
            txt = str_replace(text)
            txt = symbols_to_japanese(txt)
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



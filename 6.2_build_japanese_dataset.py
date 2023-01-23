import os.path
import shutil

import librosa
import soundfile
import tqdm
from pypinyin import lazy_pinyin
all_pinyins = [i.split("\t")[0] for i in open("assets/opencpop-strict.dict").readlines()]
pinyin2phoneme = {i.split("\t")[0]: i.split("\t")[1].strip() for i in open("assets/opencpop-strict.dict").readlines()}
punc = ['！', '？', "…", "，", "。", '!', '?', "…", ",", ".", " ", "、", "~"]
from janome.tokenizer import Tokenizer

def to_romaji(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    romaji_text = ""
    for token in tokens:
        romaji_text += token.reading + " " if token.reading else ""
    return romaji_text.strip()

print(to_romaji("あかいくつ"))
#
# def to_pinyin(s):
#     # Create a translation table
#     table = str.maketrans('', '', "".join(punc))
#     # Remove punctuation marks
#     s = s.translate(table)
#
#     pinyin_list = lazy_pinyin(s)
#     for i, pinyin in enumerate(pinyin_list):
#         if pinyin == 'n':
#             pinyin_list[i] = 'en'
#             pinyin = "en"
#         assert pinyin in all_pinyins, (pinyin, s,pinyin_list)
#
#     return ' '.join(pinyin_list)
#
# def to_phones(pinyin):
#     pinyin = [i for i in pinyin.split(" ")]
#     phones = [pinyin2phoneme[i] for i in pinyin]
#     return " ".join(phones)
#
# for spk in os.listdir("output"):
#     if os.path.isdir(f"output/{spk}"):
#         label_path = f"labels/{spk}_label.txt"
#         os.makedirs(f"singer_data/{spk}/raw/wavs", exist_ok=True)
#         transcriptions = open(f"singer_data/{spk}/raw/transcriptions.txt", "w")
#         for line in tqdm.tqdm(open(label_path).readlines()):
#             wavpath, text = line.strip().split("|")
#             name = os.path.splitext(os.path.basename(wavpath))[0]
#             if not os.path.exists(wavpath):
#                 print(wavpath, "not exist, skip")
#                 continue
#             pinyin = to_pinyin(text)
#             if pinyin == "":
#                 print(wavpath, "null")
#                 continue
#             # print(pinyin)
#             phones = to_phones(pinyin)
#             # print(phones)
#             transcriptions.write(f"{name}|啊|{phones} SP|rest|0|0|0\n")
#             if not os.path.exists(f"singer_data/{spk}/raw/wavs/{name}.wav"):
#                 wav, sr = librosa.load(wavpath, sr=44100)
#                 soundfile.write(f"singer_data/{spk}/raw/wavs/{name}.wav", wav, sr)
#         transcriptions.close()

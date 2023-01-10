import os.path
import shutil
from pypinyin import lazy_pinyin
all_pinyins = [i.split("\t")[0] for i in open("assets/opencpop-strict.txt").readlines()]
punc = ['！', '？', "…", "，", "。", '!', '?', "…", ",", ".", " ", "、", "~"]

def to_pinyin(s):
    # Create a translation table
    table = str.maketrans('', '', "".join(punc))
    # Remove punctuation marks
    s = s.translate(table)

    pinyin_list = lazy_pinyin(s)
    for i, pinyin in enumerate(pinyin_list):
        if pinyin == 'n':
            pinyin_list[i] = 'en'
            pinyin = "en"
        assert pinyin in all_pinyins, (pinyin, s,pinyin_list)

    return ' '.join(pinyin_list)
spk = "otto"
label_path = f"labels/{spk}_label.txt"
for line in open(label_path).readlines():
    wavpath, text = line.strip().split("|")
    assert os.path.exists(wavpath), wavpath
    pinyin = to_pinyin(text)
    print(pinyin)
    with open(wavpath.replace(".wav", ".lab"), "w") as f:
        f.write(pinyin+"\n")

# 删除没有标注的音频
print("正在删除没有标注的音频...")
for wavname in os.listdir(f"output/{spk}"):
    if wavname.endswith("wav"):
        labname = wavname.replace("wav", "lab")
        if not os.path.exists(f"output/{spk}/{labname}"):
            print(wavname)
            os.system(f"rm output/{spk}/{wavname}")

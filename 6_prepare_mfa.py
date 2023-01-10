import os.path

from pypinyin import lazy_pinyin
all_pinyins = [i.split("\t")[0] for i in open("opencpop-strict.txt").readlines()]
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


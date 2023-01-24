import os

import librosa
import soundfile
from tqdm import tqdm

import utils

model = utils.load_phoneme_asr_model()


for spk in os.listdir("output"):
    if os.path.isdir(f"output/{spk}"):
        name = spk
        wav_paths = [f"output/{name}/{i}" for i in sorted(os.listdir(f"output/{name}")) if i.endswith("wav")]
        os.makedirs(f"singer_data/{spk}/raw/wavs", exist_ok=True)
        fo = open(f"singer_data/{spk}/raw/transcriptions.txt", "w")
        for path in tqdm(wav_paths):
            wav, sr = librosa.load(path, 44100)
            phones, durs = utils.get_asr_result(model, wavpath=path)
            phones = " ".join(phones)
            id_ = os.path.splitext(os.path.basename(path))[0]
            durs = " ".join([str(i) for i in durs])
            fo.write(f"{id_}|啊|{phones}|rest|0|{durs}|0\n")
            fo.flush()
            soundfile.write(f"singer_data/{spk}/raw/wavs/{id_}.wav", wav, sr)
            print("{}|{}".format(path, phones))
        fo.close()



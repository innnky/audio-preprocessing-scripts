import os
from tqdm import tqdm
import soundfile
import librosa

# 默认：保持原采样率，仅将音频修改为单声道
target_sr = None

for spk in os.listdir("output"):
    if os.path.isdir(f"output/{spk}"):
        name = spk
        wav_paths = [f"output/{name}/{i}" for i in sorted(os.listdir(f"output/{name}")) if i.endswith("wav")]
        for path in tqdm(wav_paths):
            wav, sr = librosa.load(path, target_sr)
            soundfile.write(path, wav, sr)

import os
from tqdm import tqdm
import soundfile
import librosa
import shutil
os.makedirs("output_deepfilternet", exist_ok=True)
for spk in tqdm(os.listdir("output")):
    if os.path.isdir(f"output/{spk}"):
        os.system(f"deepFilter output/{spk}/*.wav -o output_deepfilternet/{spk} -m DeepFilterNet2")
        for wavname in os.listdir(f"output_deepfilternet/{spk}"):
            if wavname.endswith("_DeepFilterNet2.wav"):
                ori_name = wavname.replace("_DeepFilterNet2.wav", ".wav")
                shutil.move(f"output_deepfilternet/{spk}/{wavname}", f"output_deepfilternet/{spk}/{ori_name}")
shutil.move("output", "output_buckup")
shutil.move("output_deepfilternet", "output")

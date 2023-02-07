import os.path
import librosa
import soundfile
import tqdm

for lang in os.listdir("labels"):
    if os.path.isdir(f"labels/{lang}"):
        for label_name in os.listdir(f"labels/{lang}"):
            if label_name.endswith("_label.txt"):
                label_path = f"labels/{lang}/{label_name}"
                spk = label_name.split("_")[0]
                os.makedirs(f"tts_data/{lang}/{spk}/wavs", exist_ok=True)
                transcriptions = open(f"tts_data/{lang}/{spk}/transcription_raw.txt", "w")
                for line in tqdm.tqdm(open(label_path).readlines()):
                    wavpath, text = line.strip().split("|")
                    name = os.path.splitext(os.path.basename(wavpath))[0]
                    if not os.path.exists(wavpath):
                        print(wavpath, "not exist, skip")
                        continue
                    transcriptions.write(f"{name}|{text}\n")
                    wav, sr = librosa.load(wavpath, sr=44100)
                    soundfile.write(f"tts_data/{lang}/{spk}/wavs/{name}.wav", wav, sr)
                transcriptions.close()

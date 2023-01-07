import os
import soundfile
import time
slice_time = 60*20

if not os.path.exists("temp/raw_with_music/"):
    os.mkdir("temp/raw_with_music/")
for spk in os.listdir("dataset"):
    if os.path.isdir(f"dataset/{spk}"):
        for long_item_idx, long_mp3_item_name in enumerate([i for i in os.listdir(f"dataset/{spk}") if not i.startswith(".")]):

            print("start loading: ", long_mp3_item_name)
            name = long_mp3_item_name.split(".")[0]
            # 转成wav
            if not long_mp3_item_name.endswith(".wav"):
                os.system(f"""ffmpeg -i 'dataset/{spk}/{long_mp3_item_name}' 'dataset/{spk}/{name}.wav'""")
                os.system(f"""rm 'dataset/{spk}/{long_mp3_item_name}'""")
            t = time.time()
            wav, sr = soundfile.read(f"dataset/{spk}/{name}.wav")
            print("load complete.", time.time()-t)
            # 切片
            for slice_idx in range((wav.shape[0]//sr )// slice_time):
                if not os.path.exists(f"temp/raw_with_music/{spk}"):
                    os.mkdir(f"temp/raw_with_music/{spk}")
                soundfile.write(f"temp/raw_with_music/{spk}/spk_{spk}_item_{long_item_idx}_slice_{slice_idx}.wav", wav[slice_idx*sr*slice_time:(slice_idx+1)*sr*slice_time, :], sr)
            # os.system(f"""rm 'dataset/{spk}/{name}.wav'""")

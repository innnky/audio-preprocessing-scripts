import os
from tqdm import tqdm
if not os.path.exists("temp/raw_without_music/"):
    os.mkdir("temp/raw_without_music/")
print("开始分离伴奏。。。。。。")
# 使用demucs去除BGM
for spk in tqdm(os.listdir("temp/raw_with_music")):
    if os.path.isdir(f"temp/raw_with_music/{spk}"):
        print("说话人", spk)
        iii = 0
        for slicepath in tqdm([i for i in os.listdir(f"temp/raw_with_music/{spk}") if i.endswith("wav")]):
            print(slicepath, iii)
            iii +=1
            os.system(f"""demucs --two-stems=vocals -d cuda temp/raw_with_music/{spk}/{slicepath}""")
            name = slicepath.split(".")[0]
            if not os.path.exists(f"temp/raw_without_music/{spk}"):
                os.mkdir(f"temp/raw_without_music/{spk}")
            os.system(f"""mv separated/htdemucs/{name}/vocals.wav temp/raw_without_music/{spk}/{name}.wav""")
            os.system("rm -rf separated/htdemucs/*")

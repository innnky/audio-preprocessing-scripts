import os

# 转成mp3压缩体积方便上传服务器
for spk in os.listdir("dataset"):
    if os.path.isdir(f"dataset/{spk}"):
        for long_item_idx, long_mp3_item_name in enumerate(
                [i for i in os.listdir(f"dataset/{spk}") if not i.startswith(".")]):

            print("start loading: ", long_mp3_item_name)
            name = long_mp3_item_name.split(".")[0]
            # 转成mp3
            if not long_mp3_item_name.endswith("mp3"):
                os.system(f"""ffmpeg -i 'dataset/{spk}/{long_mp3_item_name}' 'dataset/{spk}/{name}.mp3'""")
                os.system(f"""rm 'dataset/{spk}/{long_mp3_item_name}'""")

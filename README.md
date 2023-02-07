# audio-preprocessing-scripts
主要功能：diffsinger、visinger数据集一键制作
+ 长录播音频切片
+ 基于demucs伴奏分离
+ 基于whisper语音识别
+ mfa对齐
+ 完成后处理，构建diffsinger nomidi格式数据集
# 注意！！！！
开发中，，仅在mac和linux下通过部分测试，可能有各种bug！若要尝试请备份好自己的数据！！！！

需要安装的依赖：
+ ffmpeg
+ demucs==4.0.0 auditok zhconv pypinyin librosa matplotlib praat-parselmouth pyyaml soundfile sox textgrid
+ git+https://github.com/openai/whisper.git

### 数据集准备
```shell
dataset
├───speaker0
│   ├───录播1.mp4
│   ├───...
│   └───录播2.mp4
└───speaker1
    ├───录播1.mp4
    ├───...
    └───录播2.mp4
```


[//]: # (### requirements)

[//]: # (+ demucs==4.0.0 auditok==0.2.0  librosa=0.8.1 soundfile tqdm)

[//]: # (+ ffmpeg)

[//]: # (+ linux or macos)

### colab

[colab notebook link](https://colab.research.google.com/drive/1VZ7aD8Iql0sJEKztQet7UHkcfVwruuvS?usp=sharing) 

## Reference
+ [DiffSinger数据集制作](https://github.com/openvpi/DiffSinger/tree/refactor/pipelines)

[//]: # (### note)

[//]: # (如果不希望切出超长音频可以调整3_final_slice.py中mmax_dur的值，但调小这个值的代价是可能会将一句完整的长句中途截断，如果后续做asr会不太好)
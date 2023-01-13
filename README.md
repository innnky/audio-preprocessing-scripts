# audio-preprocessing-scripts
数据集制作-从录播到伴奏分离到singer多说话人数据集制作脚本
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
### requirements
+ demucs==4.0.0 auditok==0.2.0  librosa=0.8.1 soundfile tqdm
+ ffmpeg
+ linux or macos

### colab
[colab notebook link](https://colab.research.google.com/drive/1Z-a4HQ4CxyY1cSpVcaEZxta4GVRReens?usp=sharing) 
### note
如果不希望切出超长音频可以调整3_final_slice.py中mmax_dur的值，但调小这个值的代价是可能会将一句完整的长句中途截断，如果后续做asr会不太好
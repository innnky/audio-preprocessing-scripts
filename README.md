# audio-preprocessing-scripts
数据集制作-从录播到伴奏分离到切片脚本
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
+ demucs==4.0.0 auditok==0.2.0 soundfile librosa
+ ffmpeg
+ linux or macos

### colab
[colab notebook link](https://colab.research.google.com/drive/1Z-a4HQ4CxyY1cSpVcaEZxta4GVRReens?usp=sharing) 
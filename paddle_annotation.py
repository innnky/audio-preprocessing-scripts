# 自动标注文件
import os
import re
import tqdm
import jieba.posseg as psg
import json
import shutil
import uuid
from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.t2s.exps.syn_utils import get_frontend
import random
import string
import librosa
 

def random_string_generator(str_size):
    allowed_chars = "abcdedghijklmnopqrstuvwxyz"
    return ''.join(random.choice(allowed_chars) for x in range(str_size))

def status_change(status_path, status):
    with open(status_path, "w", encoding="utf8") as f:
        status = {
            'on_status': status
        }
        json.dump(status, f, indent=4)

# 初始化 TTS，自动下载预训练模型
tts = TTSExecutor()
tts(text="今天天气十分不错。", output="output.wav")

# 初始化 ASR，自动下载预训练模型
asr = ASRExecutor()
result = asr(audio_file="output.wav", force_yes=True)


def get_pinyins(sentences):
    # frontend = get_frontend(
    #     lang="mix",
    #     phones_dict="/home/aistudio/PaddleSpeech/examples/other/tts_finetune/tts3/models/fastspeech2_mix_ckpt_1.2.0/phone_id_map.txt",
    #     tones_dict=None)
    
    segments = sentences
    for seg in segments:
        # Replace all English words in the sentence
        seg = re.sub('[a-zA-Z]+', '', seg)
        seg_cut = psg.lcut(seg)

        seg_cut = tts.frontend.tone_modifier.pre_merge_for_modify(seg_cut)
        all_pinyins = []
        # 为了多音词获得更好的效果，这里采用整句预测
        if tts.frontend.g2p_model == "g2pW":
            try:
                pinyins = tts.frontend.g2pW_model(seg)[0]
            except Exception:
                # g2pW采用模型采用繁体输入，如果有cover不了的简体词，采用g2pM预测
                print("[%s] not in g2pW dict,use g2pM" % seg)
                pinyins = tts.frontend.g2pM_model(seg, tone=True, char_split=False)
            pre_word_length = 0
            for word, pos in seg_cut:
                now_word_length = pre_word_length + len(word)
                if pos == 'eng':
                    pre_word_length = now_word_length
                    continue
                word_pinyins = pinyins[pre_word_length:now_word_length]
                # 矫正发音
                word_pinyins = tts.frontend.corrector.correct_pronunciation(
                    word, word_pinyins)
                all_pinyins.extend(word_pinyins)
                pre_word_length = now_word_length
    return all_pinyins 


def trans_wav_format(input_path, output_path):
    # 统一输入音频格式
    cmd = f"ffmpeg -i {input_path} -ac 1 -ar 24000 -acodec pcm_s16le {output_path}"
    print(cmd)
    os.system(cmd)
    if not os.path.exists(output_path):
        print(f"文件转换失败，请检查报错: {input_path}")
        return None
    else:
        return output_path

def annotation_dataset(data_dir, label_path, temp_dir="temp/paddle/temp", use_ffmpeg=True):
    # 先清空 temp 目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    ann_json = "temp/paddle/ann_status.json"
    status_change(ann_json, True)

    ann_result = []

    wavs = [filename for filename in os.listdir(data_dir) if filename[-4:] in ['.wav', '.mp3', '.ogg']]
    if len(wavs) < 5:
        print("数据小于5句，不建议微调，请添加数据")
        return
    with open(label_path, "w", encoding="utf8") as f:

        for idx, filename in tqdm.tqdm(enumerate(wavs)):
            # 检查文件名中是否有非法字符，存在非法字符则重新命名
            input_path = os.path.join(data_dir, filename)
            if " " in filename:
                new_filename = str(idx) + "_" + random_string_generator(4) + ".wav"
                filename = new_filename
                new_file_path = os.path.join(data_dir, new_filename)
                os.rename(input_path, new_file_path)
                print(f"文件名不合法：{input_path}")
                print(f"重命名结果：{new_file_path}")
                input_path = new_file_path

            if filename[-4:] != ".wav":
                filename = filename[:-4] + ".wav"

            if use_ffmpeg:
                # 使用 ffmpeg 统一音频格式
                output_path = os.path.join(temp_dir, filename)
                output_path = trans_wav_format(input_path, output_path)
                filepath = output_path
            else:
                filepath = input_path

            if filepath:
                asr_result = asr(audio_file=filepath, force_yes=True)
                pinyin = " ".join(get_pinyins([asr_result]))
                ann_result.append(
                    {
                        "filename": filename,
                        "filepath": filepath,
                        "asr_result": asr_result,
                        "pinyin": pinyin
                    }
                )
                f.write("{}|[P]{}[P]\n".format(filepath, pinyin))
                print(filepath, pinyin)

        status_change(ann_json, False)
    
    return ann_result

def get_wav_duration(filepath):
    wav, sr = librosa.load(filepath, sr=24000)
    duration = wav.shape[0] / sr
    return duration


# 按照 streamlit 的格式单条检查音频格式
def annotation_dataset_streamlit_step(data_path, label_path, index, temp_dir="/home/aistudio/work/temp", use_ffmpeg=True):
    
    status_code = 0
    message = ""
    data_dir = os.path.dirname(data_path)
    filename = data_path.split(os.path.sep)[-1]
    fileend = os.path.splitext(data_path)[1]

    # 检查音频文件是否存在
    if not os.path.exists(data_path):
        # 文件不存在则跳过
        message += f"{data_path} 不存在，跳过该条！\n"
        status_code = 1
        return message, status_code
    
    # 检查标注文件是否存在
    if not os.path.exists(label_path):
        # 标注文件不存在
        label_dict_result = []
    else:
        # 标注文件存在
        with open(label_path, "r", encoding="utf8") as f:
            label_dict_result = json.load(f)

    # 检查文件是否符合格式要求
    if fileend not in ['.wav', '.mp3', '.ogg']:
        # 文件后缀不符合要求
        message += f"{data_path} 文件不符合 wav, mp3, ogg 三种音频格式\n"
        status_code = 2
        return message, status_code
    
    # 检查文件名是否符合要求
    if " " in filename:
        new_filename = filename.replace(" ", "")
        new_file_path = os.path.join(data_dir, new_filename)
        os.rename(data_path, new_file_path)
        message += f"{filename} 不符合文件命名要求，重命名为 {new_filename}"
        data_path = new_file_path
        filename = data_path.split(os.path.sep)[-1]

    # 使用 ffmpeg 做转换
    if use_ffmpeg:
        output_path = os.path.join(temp_dir, filename[:-4] + ".wav")
        output_path = trans_wav_format(data_path, output_path)
        if not output_path:
            # 转化失败
            message += f"{data_path} 使用 ffmpeg 转化失败 \n"
            status_code = 3
            return message, status_code
        else:
            filepath = output_path
    else:
        filepath = data_path
    
    # 使用ASR和TTS进行预标注
    if os.path.exists(filepath):
        duration = get_wav_duration(filepath)
        # 先对文件的音频长度做检测
        if duration > 10:
            message += f"{filename} 音频长度大于10s, 跳过该条音频！"
            status_code = 4
            return message, status_code
        # 再识别音频
        asr_result = asr(audio_file=filepath, force_yes=True)
        if asr_result.strip() == "":
            # 识别结果为空
            message += f"{filename} 语音识别结果为空，跳过该条音频"
            status_code = 4
            return message, status_code
        
        # 根据结果生成拼音
        pinyin = " ".join(get_pinyins([asr_result]))

        label_dict_result.append(
            {
                "filename": filename,
                "filepath": filepath,
                "asr_result": asr_result,
                "pinyin": pinyin
            }
        )
    
    # 保存标注结果
    if len(label_dict_result) > 0:
        label_dict_result = sorted(label_dict_result, key=lambda x:x['filename'])
        with open(label_path, "w", encoding="utf8") as f:
            json.dump(label_dict_result, f, indent=2, ensure_ascii=False)
    
    return message, status_code


        







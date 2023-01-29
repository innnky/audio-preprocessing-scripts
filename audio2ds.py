# import json
#
# import librosa
# import numpy as np
# import parselmouth
#
# import utils
#
# def resize2d(x, target_len):
#     source = np.array(x)
#     source[source<0.001] = np.nan
#     target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
#     res = np.nan_to_num(target)
#     return res
#
# def get_f0(path, f0_up_key=0):
#     x, sr = librosa.load(path, 16000)
#     t = x.shape[0] / sr
#     frames = t//0.05
#     time_step = 160 / 16000 * 1000
#     f0_min = 50
#     f0_max = 1100
#
#     f0 = parselmouth.Sound(x, 16000).to_pitch_ac(
#         time_step=time_step / 1000, voicing_threshold=0.6,
#         pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
#     f0 *= pow(2, f0_up_key / 12)
#     return resize2d(f0, frames)
#
#
# wav_path = "/Users/xingyijin/Desktop/中文test.wav"
# wav_path = "/Volumes/Extend/下载/VISinger2/olddata/yunhao/wavs/000025.wav"
# model = utils.load_phoneme_asr_model()
#
# phones, durations = utils.get_asr_result(model, wavpath=wav_path)
#
# phones = " ".join(phones)
# durations = " ".join([str(i) for i in durations])
#
# f0 = get_f0(wav_path, 0)
# ds_template = {
#     "text": "啊",
#     "ph_seq": None,
#     "note_seq": None,
#     "note_dur_seq": None,
#     "is_slur_seq": None,
#     "ph_dur": None,
#     "f0_timestep": "0.05",
#     "f0_seq": None,
#     "input_type": "phoneme",
#     "offset": 0
#   }
#
# f0 = " ".join([str(round(i, 3)) for i in f0])
# print(phones)
# ds_template["ph_seq"] = phones
# ds_template["ph_dur"] = durations
# ds_template["f0_seq"] = f0
# join = " ".join(['0' for i in phones.split(" ")])
# ds_template["is_slur_seq"] = join
# ds_template["note_dur_seq"] = join
# ds_template["note_seq"] = " ".join(['rest' for i in phones.split(" ")])
#
# ds = json.dumps([ds_template],indent=2)
# print(ds)
#

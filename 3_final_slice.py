# 引入auditok库
import auditok
import soundfile
import librosa
import os

mmin_dur = 1
mmax_dur = 100000
mmax_silence = 1
menergy_threshold = 55

if not os.path.exists("output"):
    os.mkdir("output")

# 输入类别为audio
def qiefen(path, file_pre,start_i=0, ty='audio', mmin_dur=1, mmax_dur=100000, mmax_silence=1, menergy_threshold=55):

    audio_file = path
    audio, audio_sample_rate = soundfile.read(
        audio_file, dtype="int16", always_2d=True)
    # print(path)
    try:
        audio_regions = auditok.split(
            audio_file,
            min_dur=mmin_dur,  # minimum duration of a valid audio event in seconds
            max_dur=mmax_dur,  # maximum duration of an event
            # maximum duration of tolerated continuous silence within an event
            max_silence=mmax_silence,
            energy_threshold=menergy_threshold  # threshold of detection
        )
    except:
        print("err")
        wav, sr = librosa.load(audio_file, None)
        soundfile.write(audio_file, wav, sr)
        audio_regions = auditok.split(
            audio_file,
            min_dur=mmin_dur,  # minimum duration of a valid audio event in seconds
            max_dur=mmax_dur,  # maximum duration of an event
            # maximum duration of tolerated continuous silence within an event
            max_silence=mmax_silence,
            energy_threshold=menergy_threshold  # threshold of detection
        )

    for i, r in enumerate(audio_regions):
        # Regions returned by `split` have 'start' and 'end' metadata fields
        print(
            "Region {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s".format(i=i, r=r))

        epath = ''
        if r.meta.end-r.meta.start <mmin_dur:
            continue
        if (os.path.exists(file_pre) == False):
            os.mkdir(file_pre)
        num = i
        # 为了取前三位数字排序
        s = '000000' + str(num)

        file_save =  file_pre + '/' + \
                    str(start_i+i).zfill(6) +  '.wav'
        filename = r.save(file_save)
        print("region saved as: {}".format(filename))
    return start_i+i

for spk in os.listdir("temp/raw_without_music"):
    if os.path.isdir(f"temp/raw_without_music/{spk}"):
        last_i = 0
        for slicepath in [i for i in os.listdir(f"temp/raw_without_music/{spk}/") if i.endswith("wav")]:
            last_i = qiefen(f"temp/raw_without_music/{spk}/{slicepath}",f"output/{spk}",start_i=last_i+1, ty='audio',
                            mmin_dur=mmin_dur, mmax_dur=mmax_dur, mmax_silence=mmax_silence, menergy_threshold=menergy_threshold)

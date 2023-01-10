# modified from https://github.com/openvpi/DiffSinger/tree/refactor/pipelines


spk = "otto"
sliced_path = f'output/{spk}'  # Path to your sliced segments of recordings
textgrids_dir = f'textgrids/{spk}'
textgrids_revised_dir = f'textgrids/{spk}/revised'



import glob
import os
import shutil

import librosa
import matplotlib.pyplot as plt
import numpy as np
import parselmouth as pm
import soundfile
import textgrid as tg
import tqdm


def length(src: str):
    if os.path.isfile(src) and src.endswith('.wav'):
        return librosa.get_duration(filename=src) / 3600.
    elif os.path.isdir(src):
        total = 0
        for ch in [os.path.join(src, c) for c in os.listdir(src)]:
            total += length(ch)
        return total
    return 0


print('Environment initialized successfully.')




########################################

# Configuration for data paths

########################################

assert os.path.exists(sliced_path) and os.path.isdir(sliced_path), 'The chosen path does not exist or is not a directory.'

print('Sliced recording path:', sliced_path)
print()
print('===== Segment List =====')
sliced_filelist = glob.glob(f'{sliced_path}/*.wav', recursive=True)
sliced_length = length(sliced_path)
if len(sliced_filelist) > 5:
    print('\n'.join(sliced_filelist[:5] + [f'... ({len(sliced_filelist) - 5} more)']))
else:
    print('\n'.join(sliced_filelist))
print()
print(f'Found {len(sliced_filelist)} valid segments with total length of {round(sliced_length, 2)} hours.')



reported = False
for file in tqdm.tqdm(sliced_filelist):
    wave_seconds = librosa.get_duration(filename=file)
    if wave_seconds < 2.:
        reported = True
        print(f'Too short! \'{file}\' has a length of {round(wave_seconds, 1)} seconds!')
    if wave_seconds > 20.:
        reported = True
        print(f'Too long! \'{file}\' has a length of {round(wave_seconds, 1)} seconds!')
if not reported:
    print('Congratulations! All segments have proper length.')




import utils.distribution as dist

# Load dictionary
dict_path = './opencpop-strict.txt'
with open(dict_path, 'r', encoding='utf8') as f:
    rules = [ln.strip().split('\t') for ln in f.readlines()]
dictionary = {}
phoneme_set = set()
for r in rules:
    phonemes = r[1].split()
    dictionary[r[0]] = phonemes
    phoneme_set.update(phonemes)

# Run checks
check_failed = False
covered = set()
phoneme_map = {}
for ph in sorted(phoneme_set):
    phoneme_map[ph] = 0

segment_pairs = []

for file in tqdm.tqdm(sliced_filelist):
    filename = os.path.basename(file)
    name_without_ext = filename.rsplit('.', maxsplit=1)[0]
    annotation = os.path.join(sliced_path, f'{name_without_ext}.lab')
    if not os.path.exists(annotation):
        print(f'No annotation found for \'{filename}\'!')
        check_failed = True
        continue
    with open(annotation, 'r', encoding='utf8') as f:
        syllables = f.read().strip().split()
    if not syllables:
        print(f'Annotation file \'{annotation}\' is empty!')
        check_failed = True
    else:
        oov = []
        for s in syllables:
            if s not in dictionary:
                oov.append(s)
            else:
                for ph in dictionary[s]:
                    phoneme_map[ph] += 1
                covered.update(dictionary[s])
        if oov:
            print(f'Syllable(s) {oov} not allowed in annotation file \'{annotation}\'')
            check_failed = True

# Phoneme coverage
uncovered = phoneme_set - covered
if uncovered:
    print(f'The following phonemes are not covered!')
    print(sorted(uncovered))
    print('Please add more recordings to cover these phonemes.')
    check_failed = True

if not check_failed:
    print('Congratulations! All annotations are well prepared.')
    print('Here is a summary of your phoneme coverage.')

phoneme_list = sorted(phoneme_set)
phoneme_counts = [phoneme_map[ph] for ph in phoneme_list]
dist.draw_distribution(
    title='Phoneme Distribution Summary',
    x_label='Phoneme',
    y_label='Number of occurrences',
    items=phoneme_list,
    values=phoneme_counts
)
phoneme_summary = os.path.join(sliced_path, 'phoneme_distribution.jpg')
plt.savefig(fname=phoneme_summary,
            bbox_inches='tight',
            pad_inches=0.25)
plt.show()
print(f'Summary saved to \'{phoneme_summary}\'.')

segments_dir = sliced_path

########################################

# Configuration for voice arguments based on your dataset
f0_min = 40.  # Minimum value of pitch
f0_max = 1100.  # Maximum value of pitch
br_len = 0.1  # Minimum length of aspiration in seconds
br_db = -60.  # Threshold of RMS in dB for detecting aspiration
br_centroid = 2000.  # Threshold of spectral centroid in Hz for detecting aspiration

# Other arguments, do not edit unless you understand them
time_step = 0.005  # Time step for feature extraction
min_space = 0.04  # Minimum length of space in seconds
voicing_thresh_vowel = 0.45  # Threshold of voicing for fixing long utterances
voicing_thresh_breath = 0.6  # Threshold of voicing for detecting aspiration
br_win_sz = 0.05  # Size of sliding window in seconds for detecting aspiration

########################################

# import utils.tg_optimizer as optimizer

os.makedirs(textgrids_revised_dir, exist_ok=True)
for wavfile in tqdm.tqdm(sliced_filelist):
    name = os.path.basename(wavfile).rsplit('.', maxsplit=1)[0]
    textgrid = tg.TextGrid()
    if not os.path.exists(os.path.join(textgrids_dir, f'{name}.TextGrid')):
        print(f"skip {name}!no TextGrid found.")
        continue
    textgrid.read(os.path.join(textgrids_dir, f'{name}.TextGrid'))
    words = textgrid[0]
    phones = textgrid[1]
    sound = pm.Sound(wavfile)
    f0_voicing_breath = sound.to_pitch_ac(
        time_step=time_step,
        voicing_threshold=voicing_thresh_breath,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
    ).selected_array['frequency']
    f0_voicing_vowel = sound.to_pitch_ac(
        time_step=time_step,
        voicing_threshold=voicing_thresh_vowel,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
    ).selected_array['frequency']
    y, sr = librosa.load(wavfile, sr=24000, mono=True)
    hop_size = int(time_step * sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=hop_size).squeeze(0)

    # Fix long utterances
    i = j = 0
    while i < len(words):
        word = words[i]
        phone = phones[j]
        if word.mark is not None and word.mark != '':
            i += 1
            j += len(dictionary[word.mark])
            continue
        if i == 0:
            i += 1
            j += 1
            continue
        prev_word = words[i - 1]
        prev_phone = phones[j - 1]
        # Extend length of long utterances
        while word.minTime < word.maxTime - time_step:
            pos = min(f0_voicing_vowel.shape[0] - 1, int(word.minTime / time_step))
            if f0_voicing_vowel[pos] < f0_min:
                break
            prev_word.maxTime += time_step
            prev_phone.maxTime += time_step
            word.minTime += time_step
            phone.minTime += time_step
        i += 1
        j += 1

    # Detect aspiration
    i = j = 0
    while i < len(words):
        word = words[i]
        phone = phones[j]
        if word.mark is not None and word.mark != '':
            i += 1
            j += len(dictionary[word.mark])
            continue
        if word.maxTime - word.minTime < br_len:
            i += 1
            j += 1
            continue
        ap_ranges = []
        br_start = None
        win_pos = word.minTime
        while win_pos + br_win_sz <= word.maxTime:
            all_noisy = (f0_voicing_breath[int(win_pos / time_step) : int((win_pos + br_win_sz) / time_step)] < f0_min).all()
            rms_db = 20 * np.log10(np.clip(sound.get_rms(from_time=win_pos, to_time=win_pos + br_win_sz), a_min=1e-12, a_max=1))
            # print(win_pos, win_pos + br_win_sz, all_noisy, rms_db)
            if all_noisy and rms_db >= br_db:
                if br_start is None:
                    br_start = win_pos
            else:
                if br_start is not None:
                    br_end = win_pos + br_win_sz - time_step
                    if br_end - br_start >= br_len:
                        centroid = spectral_centroid[int(br_start / time_step) : int(br_end / time_step)].mean()
                        if centroid >= br_centroid:
                            ap_ranges.append((br_start, br_end))
                    br_start = None
                    win_pos = br_end
            win_pos += time_step
        if br_start is not None:
            br_end = win_pos + br_win_sz - time_step
            if br_end - br_start >= br_len:
                centroid = spectral_centroid[int(br_start / time_step) : int(br_end / time_step)].mean()
                if centroid >= br_centroid:
                    ap_ranges.append((br_start, br_end))
        # print(ap_ranges)
        if len(ap_ranges) == 0:
            i += 1
            j += 1
            continue
        words.removeInterval(word)
        phones.removeInterval(phone)
        if word.minTime < ap_ranges[0][0]:
            words.add(minTime=word.minTime, maxTime=ap_ranges[0][0], mark=None)
            phones.add(minTime=phone.minTime, maxTime=ap_ranges[0][0], mark=None)
            i += 1
            j += 1
        for k, ap in enumerate(ap_ranges):
            if k > 0:
                words.add(minTime=ap_ranges[k - 1][1], maxTime=ap[0], mark=None)
                phones.add(minTime=ap_ranges[k - 1][1], maxTime=ap[0], mark=None)
                i += 1
                j += 1
            words.add(minTime=ap[0], maxTime=min(word.maxTime, ap[1]), mark='AP')
            phones.add(minTime=ap[0], maxTime=min(word.maxTime, ap[1]), mark='AP')
            i += 1
            j += 1
        if ap_ranges[-1][1] < word.maxTime:
            words.add(minTime=ap_ranges[-1][1], maxTime=word.maxTime, mark=None)
            phones.add(minTime=ap_ranges[-1][1], maxTime=phone.maxTime, mark=None)
            i += 1
            j += 1

    # Remove short spaces
    i = j = 0
    while i < len(words):
        word = words[i]
        phone = phones[j]
        if word.mark is not None and word.mark != '':
            i += 1
            j += (1 if word.mark == 'AP' else len(dictionary[word.mark]))
            continue
        if word.maxTime - word.minTime >= min_space:
            word.mark = 'SP'
            phone.mark = 'SP'
            i += 1
            j += 1
            continue
        if i == 0:
            if len(words) >= 2:
                words[i + 1].minTime = word.minTime
                phones[j + 1].minTime = phone.minTime
                words.removeInterval(word)
                phones.removeInterval(phone)
            else:
                break
        elif i == len(words) - 1:
            if len(words) >= 2:
                words[i - 1].maxTime = word.maxTime
                phones[j - 1].maxTime = phone.maxTime
                words.removeInterval(word)
                phones.removeInterval(phone)
            else:
                break
        else:
            words[i - 1].maxTime = words[i + 1].minTime = (word.minTime + word.maxTime) / 2
            phones[j - 1].maxTime = phones[j + 1].minTime = (phone.minTime + phone.maxTime) / 2
            words.removeInterval(word)
            phones.removeInterval(phone)
    textgrid.write(os.path.join(textgrids_revised_dir, f'{name}.TextGrid'))



import utils.distribution as dist


def key_to_name(midi_key):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return note_names[midi_key % 12] + str(midi_key // 12 - 1)


pit_map = {}
if not f0_min in locals():
    f0_min = 40.
if not f0_max in locals():
    f0_max = 1100.
if not voicing_thresh_vowel in locals():
    voicing_thresh_vowel = 0.45
for wavfile in tqdm.tqdm(sliced_filelist):
    name = os.path.basename(wavfile).rsplit('.', maxsplit=1)[0]
    textgrid = tg.TextGrid()
    if not os.path.exists(os.path.join(textgrids_dir, f'{name}.TextGrid')):
        print(f"skip {name}!no TextGrid found.")
        continue
    textgrid.read(os.path.join(textgrids_revised_dir, f'{name}.TextGrid'))
    timestep = 0.01
    f0 = pm.Sound(wavfile).to_pitch_ac(
        time_step=timestep,
        voicing_threshold=voicing_thresh_vowel,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
    ).selected_array['frequency']
    pitch = 12. * np.log2(f0 / 440.) + 69.
    for word in textgrid[0]:
        if word.mark in ['AP', 'SP']:
            continue
        if word.maxTime - word.minTime < timestep:
            continue
        word_pit = pitch[int(word.minTime / timestep) : int(word.maxTime / timestep)]
        word_pit = np.extract(word_pit >= 0, word_pit)
        if word_pit.shape[0] == 0:
            continue
        counts = np.bincount(word_pit.astype(np.int64))
        midi = counts.argmax()
        if midi in pit_map:
            pit_map[midi] += 1
        else:
            pit_map[midi] = 1
midi_keys = sorted(pit_map.keys())
midi_keys = list(range(midi_keys[0], midi_keys[-1] + 1))
dist.draw_distribution(
    title='Pitch Distribution Summary',
    x_label='Pitch',
    y_label='Number of occurrences',
    items=[key_to_name(k) for k in midi_keys],
    values=[pit_map.get(k, 0) for k in midi_keys]
)
pitch_summary = os.path.join(sliced_path, 'pitch_distribution.jpg')
plt.savefig(fname=pitch_summary,
            bbox_inches='tight',
            pad_inches=0.25)
plt.show()
print(f'Summary saved to \'{pitch_summary}\'.')




########################################

# Name and tags of your dataset
dataset_name = spk
dataset_tags = ''  # Optional

########################################

import random
import re

from textgrid import TextGrid

assert dataset_name != '', 'Dataset name cannot be empty.'
assert re.search(r'[^0-9A-Za-z_]', dataset_name) is None, 'Dataset name contains invalid characters.'
full_name = dataset_name
if dataset_tags != '':
    assert re.fullmatch(r'[^0-9A-Za-z_]', dataset_name) is None, 'Dataset tags contain invalid characters.'
    full_name += f'_{dataset_tags}'
assert not os.path.exists(f'singer_data/{full_name}'), f'The name \'{full_name}\' already exists in your \'data\' folder!'

print('Dataset name:', dataset_name)
if dataset_tags != '':
    print('Tags:', dataset_tags)

formatted_path = f'singer_data/{full_name}/raw/wavs'
os.makedirs(formatted_path)
transcriptions = []
samplerate = 44100
min_sil = int(0.1 * samplerate)
max_sil = int(2. * samplerate)
for wavfile in tqdm.tqdm(sliced_filelist):
    name = os.path.basename(wavfile).rsplit('.', maxsplit=1)[0]
    y, _ = librosa.load(wavfile, sr=samplerate, mono=True)
    tg = TextGrid()
    if not os.path.exists(os.path.join(textgrids_dir, f'{name}.TextGrid')):
        continue
    tg.read(os.path.join(textgrids_revised_dir, f'{name}.TextGrid'))
    ph_seq = [ph.mark for ph in tg[1]]
    ph_dur = [ph.maxTime - ph.minTime for ph in tg[1]]
    if random.random() < 0.5:
        len_sil = random.randrange(min_sil, max_sil)
        y = np.concatenate((np.zeros((len_sil,), dtype=np.float32), y))
        if ph_seq[0] == 'SP':
            ph_dur[0] += len_sil / samplerate
        else:
            ph_seq.insert(0, 'SP')
            ph_dur.insert(0, len_sil / samplerate)
    if random.random() < 0.5:
        len_sil = random.randrange(min_sil, max_sil)
        y = np.concatenate((y, np.zeros((len_sil,), dtype=np.float32)))
        if ph_seq[-1] == 'SP':
            ph_dur[-1] += len_sil / samplerate
        else:
            ph_seq.append('SP')
            ph_dur.append(len_sil / samplerate)
    ph_seq = ' '.join(ph_seq)
    ph_dur = ' '.join([str(round(d, 6)) for d in ph_dur])
    soundfile.write(os.path.join(formatted_path, f'{name}.wav'), y, samplerate)
    transcriptions.append(f'{name}|啊|{ph_seq}|rest|0|{ph_dur}|0')
with open(f'singer_data/{full_name}/raw/transcriptions.txt', 'w', encoding='utf8') as f:
    print('\n'.join(transcriptions), file=f)
print(f'All wavs and transcriptions saved at \'data/{full_name}/raw/\'.')

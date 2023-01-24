import sys
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

sys.path.append('../..')
import utils.commons  as commons
import  utils.attentions as attentions
import  utils.utils as ut


ttsing_phone_set = ['_'] + [
    "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r",
    "s", "sh", "t", "x", "z", "zh", "a", "ai", "an", "ang", "ao", "e", "ei",
    "en", "eng", "er", "iii", "ii", "i", "ia", "ian", "iang", "iao", "ie", "in",
    "ing", "iong", "iou", "o", "ong", "ou", "u", "ua", "uai", "uan", "uang",
    "uei", "uen", "ueng", "uo", "v", "van", "ve", "vn", "AH", "AA", "AO", "ER",
    "IH", "IY", "UH", "UW", "EH", "AE", "AY", "EY", "OY", "AW", "OW", "P", "B",
    "T", "D", "K", "G", "M", "N", "NG", "L", "S", "Z", "Y", "TH", "DH", "SH",
    "ZH", "CH", "JH", "V", "W", "F", "R", "HH", "AH0", "AA0", "AO0", "ER0",
    "IH0", "IY0", "UH0", "UW0", "EH0", "AE0", "AY0", "EY0", "OY0", "AW0", "OW0",
    "AH1", "AA1", "AO1", "ER1", "IH1", "IY1", "UH1", "UW1", "EH1", "AE1", "AY1",
    "EY1", "OY1", "AW1", "OW1", "AH2", "AA2", "AO2", "ER2", "IH2", "IY2", "UH2",
    "UW2", "EH2", "AE2", "AY2", "EY2", "OY2", "AW2", "OW2", "AH3", "AA3", "AO3",
    "ER3", "IH3", "IY3", "UH3", "UW3", "EH3", "AE3", "AY3", "EY3", "OY3", "AW3",
    "OW3", "D-1", "T-1", "P*", "B*", "T*", "D*", "K*", "G*", "M*", "N*", "NG*",
    "L*", "S*", "Z*", "Y*", "TH*", "DH*", "SH*", "ZH*", "CH*", "JH*", "V*",
    "W*", "F*", "R*", "HH*", "sp", "sil", "or", "ar", "aor", "our", "angr",
    "eir", "engr", "air", "ianr", "iaor", "ir", "ingr", "ur", "iiir", "uar",
    "uangr", "uenr", "iir", "ongr", "uor", "ueir", "iar", "iangr", "inr",
    "iour", "vr", "uanr", "ruai", "TR", "rest",
    # opencpop
    'w', 'SP', 'AP', 'un', 'y', 'ui', 'iu',
    # opencpop-strict
    'i0', 'E', 'En',
    # japanese-common
    'ts.', 'f.', 'sh.', 'ry.', 'py.', 'h.', 'p.', 'N.', 'a.', 'm.', 'w.', 'ky.',
    'n.', 'd.', 'j.', 'cl.', 'ny.', 'z.', 'o.', 'y.', 't.', 'u.', 'r.', 'pau',
    'ch.', 'e.', 'b.', 'k.', 'g.', 's.', 'i.',
    # japanese-unique
    'gy.', 'my.', 'hy.', 'br', 'by.', 'v.', 'ty.', 'xx.', 'U.', 'I.', 'dy.'
]
ttsing_phone_to_int = {}
int_to_ttsing_phone = {}
for idx, item in enumerate(ttsing_phone_set):
    ttsing_phone_to_int[item] = idx
    int_to_ttsing_phone[idx] = item
LRELU_SLOPE = 0.1


class SynthesizerTrn(nn.Module):
    """
    Model
    """

    def __init__(self, hps):
        super().__init__()
        self.hps = hps

        self.pre_net = nn.Conv1d(hps.data.acoustic_dim, hps.model.prior_hidden_channels, 1)
        self.proj = nn.Conv1d(hps.model.prior_hidden_channels, len(ttsing_phone_set), 1)
        self.encoder = attentions.Encoder(
            hps.model.prior_hidden_channels,
            hps.model.prior_filter_channels,
            hps.model.prior_n_heads,
            hps.model.prior_n_layers,
            hps.model.prior_kernel_size,
            hps.model.prior_p_dropout)

    def forward(self, mel,phone=None, phone_lengths=None):
        units = mel
        x = self.pre_net(units)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        x = self.proj(x)
        if phone is not None:
            loss_all = F.cross_entropy(x, phone)
        else:
            loss_all = None

        return x, loss_all

def remove_consecutive_duplicates(lst):
    sr = 16000
    hop = 320
    new_lst = []
    dur_lst = []
    previous = None
    count = 1
    for item in lst:
        if item == previous:
            count += 1
        else:
            if previous:
                new_lst.append(f"{previous}")
                dur_lst.append(count*hop/sr)
            previous = item
            count = 1
    new_lst.append(f"{previous}")
    dur_lst.append(count*hop/sr)
    return new_lst, dur_lst

def convert_x_to_phones(x):
    phoneme_ids = torch.argmax(x, dim=1)
    phones, durs = remove_consecutive_duplicates([int_to_ttsing_phone[int(i)] for i in phoneme_ids[0, :]])
    return phones, durs

def load_phoneme_asr_model():
    config_json = "utils/config.json"
    checkpoint_path = f"assets/G_18000.pth"
    hps = ut.get_hparams_from_file(config_json)
    net_g = SynthesizerTrn(hps)
    _ = net_g.eval()
    _ = ut.load_checkpoint(checkpoint_path, net_g, None)
    model = ut.load_whisper_model()
    return (model, net_g)

def get_asr_result(m, wavpath):
    model, net_g = m
    units = ut.get_whisper_units(model, wavpath, False)
    x, _ = net_g(units, phone_lengths=torch.LongTensor([units.shape[2]]))
    phones, durs = convert_x_to_phones(x)
    return phones, durs




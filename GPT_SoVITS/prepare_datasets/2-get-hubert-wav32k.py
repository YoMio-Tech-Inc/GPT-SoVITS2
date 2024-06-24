# -*- coding: utf-8 -*-

import os
import shutil
import traceback
from time import time as ttime

import librosa
import numpy as np
import torch
from GPT_SoVITS.feature_extractor import hubert
from my_utils import load_audio
from scipy.io import wavfile
from pyloudnorm import Meter

inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("_CUDA_VISIBLE_DEVICES")

opt_dir = os.environ.get("opt_dir")
hubert.cnhubert_base_path = os.environ.get("cnhubert_base_dir")
is_half = eval(os.environ.get("is_half", "True"))


def my_save(fea, path):  #####fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path = "%s%s.pth" % (ttime(), i_part)
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


hubert_dir = "%s/4-cnhubert" % (opt_dir)
wav32dir = "%s/5-wav32k" % (opt_dir)
os.makedirs(opt_dir, exist_ok=True)
os.makedirs(hubert_dir, exist_ok=True)
os.makedirs(wav32dir, exist_ok=True)

maxx = 0.95
alpha = 0.5
if torch.cuda.is_available():
    device = "cuda:0"
# elif torch.backends.mps.is_available():
#     device = "mps"
else:
    device = "cpu"
model = hubert.get_model().half().to(device)

nan_fails = []


def balance_loudness(audio, target_loudness=-23.0):
    meter = Meter(44100)  # 创建一个响度计量器

    # 测量积分响度
    loudness = meter.integrated_loudness(audio)

    # 计算增益
    gain_db = target_loudness - loudness
    gain_linear = 10 ** (gain_db / 20.0)

    # 应用增益
    balanced_audio = audio * gain_linear

    # 应用软限幅以防止削波
    balanced_audio = np.tanh(balanced_audio)

    return balanced_audio


def name2go(wav_name, wav_path):
    hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
    if os.path.exists(hubert_path):
        return
    audio = load_audio(wav_path, 32000)
    audio = balance_loudness(audio)
    audio_16k = librosa.resample(audio, orig_sr=32000, target_sr=16000)
    tensor_wav16 = torch.from_numpy(audio_16k)
    tensor_wav16 = tensor_wav16.half().to(device)
    ssl = (
        model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"]
        .transpose(1, 2)
        .cpu()
    )  # torch.Size([1, 768, 215])
    if np.isnan(ssl.detach().numpy()).sum() != 0:
        nan_fails.append((wav_name, wav_path))
        print("nan filtered:%s" % wav_name)
        return
    wavfile.write(
        "%s/%s" % (wav32dir, wav_name),
        32000,
        audio,
    )
    my_save(ssl, hubert_path)


with open(inp_text, "r", encoding="utf8") as f:
    lines = f.read().strip("\n").split("\n")

for line in lines[int(i_part) :: int(all_parts)]:
    try:
        # wav_name,text=line.split("\t")
        wav_name, spk_name, language, text = line.split("|")
        if inp_wav_dir:
            wav_name = os.path.basename(wav_name)
            wav_path = "%s/%s" % (inp_wav_dir, wav_name)

        else:
            wav_path = wav_name
            wav_name = os.path.basename(wav_name)
        name2go(wav_name, wav_path)
    except:
        print(line, traceback.format_exc())

if len(nan_fails) > 0 and is_half == True:
    is_half = False
    model = model.float()
    for wav in nan_fails:
        try:
            name2go(wav[0], wav[1])
        except:
            print(wav_name, traceback.format_exc())

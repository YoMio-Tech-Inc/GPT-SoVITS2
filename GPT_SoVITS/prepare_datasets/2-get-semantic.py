# -*- coding: utf-8 -*-

import os

import librosa
import numpy as np
import onnxruntime
import torch
import torchaudio
import whisper
from tqdm import tqdm  # 修改这一行

max_val = 0.8


def load_audio(audio, target_sr):
    speech, sample_rate = torchaudio.load(audio)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(speech)
    return speech


def postprocess(speech, target_sr, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db, frame_length=win_length, hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


option = onnxruntime.SessionOptions()
option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
option.intra_op_num_threads = 1
dataset_dir = "./dataset"

speech_tokenizer_session = onnxruntime.InferenceSession(
    "./pretrained_models/speech_tokenizer_v1.onnx",
    sess_options=option,
    providers=["CUDAExecutionProvider"],
)


def process(data):
    with tqdm(total=len(data), desc="处理Semantic特征") as pbar:
        for data in data:
            _, wav_name, _, index_folder = data
            wav_path = os.path.join(dataset_dir, index_folder, wav_name)

            audio_16 = load_audio(wav_path, 16000)
            audio_16 = postprocess(audio_16, 16000)

            feat = whisper.log_mel_spectrogram(audio_16, n_mels=128)
            speech_token = (
                speech_tokenizer_session.run(
                    None,
                    {
                        speech_tokenizer_session.get_inputs()[0].name: feat.detach()
                        .cpu()
                        .numpy(),
                        speech_tokenizer_session.get_inputs()[1].name: np.array(
                            [feat.shape[2]], dtype=np.int32
                        ),
                    },
                )[0]
                .flatten()
                .tolist()
            )
            speech_token = np.array(speech_token)
            # 保存speech token
            speech_token_path = wav_path + ".npy"
            np.save(speech_token_path, speech_token)
            pbar.update(1)


todo = []
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".txt"):
            index_folder = os.path.relpath(root, dataset_dir)
            file_path = os.path.join(root, file)

            # 尝试不同的编码
            encodings = ["utf-8", "gbk", "gb2312", "utf-16"]
            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        lines = f.readlines()
                    break  # 如果成功读取，跳出循环
                except UnicodeDecodeError:
                    continue  # 如果解码失败，尝试下一个编码
            else:
                print(f"无法解码文件 {file_path}，跳过此文件")
                continue  # 如果所有编码都失败，跳过此文件

            for line in lines:
                try:
                    spk_name, wav_name, text = line.split("|")
                    todo.append([spk_name, wav_name, text, index_folder])
                except Exception:
                    print(line)

process(todo)

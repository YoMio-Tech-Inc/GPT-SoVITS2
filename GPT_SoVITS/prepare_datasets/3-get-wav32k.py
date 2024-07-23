import os
import numpy as np
import torch
import torchaudio
import librosa
from scipy.io import wavfile
from tqdm import tqdm


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
    max_val = 0.8
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


def rewrite_wavfile(wav_path, text, target_sr=32000):
    audio = load_audio(wav_path, target_sr)
    audio = postprocess(audio, target_sr)
    audio_np = audio.squeeze().numpy()
    new_wav_path = wav_path + ".wav"
    wavfile.write(new_wav_path, target_sr, (audio_np * 32768).astype(np.int16))
    # os.remove(wav_path)


def is_valid_audio(wav_path, text):
    try:
        duration = librosa.get_duration(filename=wav_path)
        return duration >= 0.5 and len(text) > 2
    except Exception as e:
        print(f"无法读取音频文件 {wav_path}：{str(e)}")
        return False


def process_dataset(dataset_dir):
    todo = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                encodings = ["utf-8", "gbk", "gb2312", "utf-16"]
                for encoding in encodings:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            lines = f.readlines()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    print(f"无法解码文件 {file_path}，跳过此文件")
                    continue

                for line in lines:
                    try:
                        _, wav_name, text = line.strip().split("|")
                        wav_path = os.path.join(root, wav_name)
                        if is_valid_audio(wav_path, text):
                            todo.append((wav_path, text))
                        else:
                            print(f"跳过音频 {wav_path}：音频过短或文本过短")
                    except Exception as e:
                        print(f"处理行时出错: {line}. 错误: {e}")

    with tqdm(total=len(todo), desc="重写 WAV 文件") as pbar:
        for wav_path, text in todo:
            rewrite_wavfile(wav_path, text)
            pbar.update(1)

dataset_dir = "./dataset"
process_dataset(dataset_dir)
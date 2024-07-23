import os
import numpy as np
import torch
import torchaudio
import librosa
from scipy.io import wavfile
from tqdm import tqdm





def delete_wavfile(wav_path):
    os.remove(wav_path)


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
                        _, wav_name, _ = line.strip().split("|")
                        wav_path = os.path.join(root, wav_name)
                        todo.append(wav_path)
                    except Exception as e:
                        print(f"处理行时出错: {line}. 错误: {e}")

    for wav_path in todo:
        try:
            delete_wavfile(wav_path)
        except Exception:
            pass


dataset_dir = "./dataset"
process_dataset(dataset_dir)

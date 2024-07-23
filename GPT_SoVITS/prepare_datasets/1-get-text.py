# -*- coding: utf-8 -*-
import os
import os.path
import traceback
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer

dataset_dir = "./dataset"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print("正在加载模型, 设备: ", device)
bert_model = SentenceTransformer(
    "BAAI/bge-m3", cache_folder="./pretrained_models", device=device
).half()

def process(data):
    batch_size = 48
    total_batches = (len(data) + batch_size - 1) // batch_size  # 计算总批次数
    
    with tqdm(total=total_batches, desc="处理BERT特征") as pbar:
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            texts = [item[2] for item in batch]
            # print("此批次的第一句话: ",texts[0])
            bert_features = bert_model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                output_value="token_embeddings",
                batch_size=batch_size
            )
            
            for j, (spk_name, wav_name, text, index_folder) in enumerate(batch):
                try:
                    bert_dir = os.path.dirname(wav_name)
                    name = os.path.basename(wav_name)
                    path_bert = os.path.join(dataset_dir, index_folder, bert_dir, f"{name}.pt")
                    torch.save(bert_features[j].cpu(), path_bert)
                    
                except Exception:
                    print(spk_name, wav_name, text, traceback.format_exc())
            
            pbar.update(1)  # 更新进度条

todo = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".txt"):
            index_folder = os.path.relpath(root, dataset_dir)
            file_path = os.path.join(root, file)
            
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
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
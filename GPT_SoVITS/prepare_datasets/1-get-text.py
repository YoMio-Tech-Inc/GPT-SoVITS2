# -*- coding: utf-8 -*-
import os
import os.path
import shutil
import traceback
from time import time as ttime

import torch
from sentence_transformers import SentenceTransformer

inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("_CUDA_VISIBLE_DEVICES")
opt_dir = os.environ.get("opt_dir")
is_half = eval(os.environ.get("is_half", "True"))


def my_save(fea, path):
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s%s.pth" % (ttime(), i_part)
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
if not os.path.exists(txt_path):
    bert_dir = "%s/3-bert" % (opt_dir)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    bert_model = SentenceTransformer(
        "BAAI/bge-m3", cache_folder="./pretrained_models", device=device
    )

    def get_bert_feature(text, is_half=True):
        return bert_model.encode(
            text, convert_to_tensor=True, normalize_embeddings=True
        ).half()

    def process(data, res):
        # for name, text, lan in data:
        for name, text in data:  # ! 去掉了language
            try:
                name = os.path.basename(name)
                path_bert = "%s/%s.pt" % (bert_dir, name)
                if not os.path.exists(
                    path_bert
                ):  # ! 去掉language是中文才提取bert的设定。
                    bert_feature = get_bert_feature(text)  # ! 去掉phoneme
                    my_save(bert_feature, path_bert)
                res.append([name, text])
            except Exception:
                print(name, text, traceback.format_exc())

    todo = []
    res = []
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")
    for line in lines[int(i_part) :: int(all_parts)]:
        try:
            wav_name, spk_name, language, text = line.split("|")
            todo.append([wav_name, text])  # ! 去掉了language
        except Exception:
            print(line, traceback.format_exc())

    process(todo, res)
    opt = []
    for name, text in res:
        opt.append("%s\t%s" % (name, text))
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")

# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
# ruff:noqa
import os
import sys

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
grand_parent_dir = os.path.dirname(parent_dir)
# 将当前目录添加到Python的模块搜索路径中
sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(grand_parent_dir)

import torch
from AR.models.utils import (
    dpo_loss,
    get_batch_logps,
    make_pad_mask,
    make_reject_y,
    sample,
    topk_sampling,
)
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
from transformers import AutoConfig
from transformers.models import qwen2

qwen_config = AutoConfig.from_pretrained(
    "Qwen/Qwen2-0.5B",
)
qwen_config.hidden_size = 768
qwen_config.num_attention_heads = 16
qwen_config.num_hidden_layers = 20
qwen_config.max_window_layers = 20
qwen_config.intermediate_size = 4096


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.embedding_dim = config["model"]["embedding_dim"]
        self.text_vocab_size = config["model"]["text_vocab_size"]
        self.speech_vocab_size = config["model"]["speech_vocab_size"]
        self.bert_dim = config["model"]["bert_dim"]
        self.qwen_config = config.get("qwen", qwen_config)
        self.EOS = self.speech_vocab_size - 1
        self.bert_proj = nn.Linear(self.bert_dim, self.embedding_dim)
        self.text_embedding = nn.Embedding(
            self.text_vocab_size,
            self.embedding_dim,
        )

        self.speech_embedding = nn.Embedding(self.speech_vocab_size, self.embedding_dim)

        self.llm = qwen2.Qwen2Model(config=self.qwen_config)

        self.predict_layer = nn.Linear(
            self.embedding_dim, self.speech_vocab_size, bias=False
        )
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")

        self.ar_accuracy_metric = MulticlassAccuracy(
            self.speech_vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )
        
    def forward(
        self,
        x,
        x_lens,
        y,
        y_lens,
        prompt_x,
        prompt_x_lens,
        prompt_y,
        prompt_y_lens,
        x_bert_feature,
        prompt_x_bert_feature,
    ):
        """
        x: target text_ids - [batch_size, x_len]
        y: target semantic_ids - [batch_size, y_len]
        prompt_x: prompt text_ids - [batch_size, prompt_x_len]
        prompt_y: prompt semantic_ids - [batch_size, prompt_y_len]
        """
        batch_size = x.shape[0]
        
        prompt_y_mask = torch.arange(prompt_y.shape[1]).expand(batch_size, -1) < prompt_y_lens.unsqueeze(1)
        prompt_y = torch.where(prompt_y_mask, prompt_y, self.EOS)
        prompt_y = torch.nn.functional.pad(prompt_y, (1, 1), value=self.EOS)  # 在开头和结尾添加 EOS
        prompt_y_lens = prompt_y_lens + 2  # 更新 prompt_y_lens，增加 2

        # 处理 y
        y_mask = torch.arange(y.shape[1]).expand(batch_size, -1) < y_lens.unsqueeze(1)
        y = torch.where(y_mask, y, self.EOS)
        y = torch.nn.functional.pad(y, (1, 1), value=self.EOS)  # 在开头和结尾添加 EOS
        targets = y[:, 1:]  # targets 从第二个元素开始
        y = y[:, :-1]  # y 去掉最后一个元素
        y_lens = y_lens + 1  # 更新 y_lens，增加 1

        prompt_x = self.text_embedding(prompt_x)
        prompt_y = self.speech_embedding(prompt_y)
        x = self.text_embedding(x)
        y = self.speech_embedding(y)

        if x_bert_feature != None and prompt_x_bert_feature != None:
            x_bert_feature = self.bert_proj(x_bert_feature)
            prompt_x_bert_feature = self.bert_proj(prompt_x_bert_feature)
            x += x_bert_feature
            prompt_x += prompt_x_bert_feature

        prompt_lens = prompt_x_lens + prompt_y_lens
        prompt_xy_max_len = prompt_lens.max()
        prompt_xy_embedded = torch.zeros(
            (batch_size, prompt_xy_max_len, self.embedding_dim), device=x.device
        )
        for i in range(batch_size):
            prompt_xy_embedded[i, : prompt_lens[i]] = torch.cat(
                [prompt_x[i, : prompt_x_lens[i]], prompt_y[i, : prompt_y_lens[i]]],
                dim=0,
            )

        xy_lens = x_lens + y_lens
        xy_max_len = xy_lens.max()
        xy_embedded = torch.zeros(
            (batch_size, xy_max_len, self.embedding_dim), device=x.device
        )
        for i in range(batch_size):
            xy_embedded[i, : xy_lens[i]] = torch.cat(
                [x[i, : x_lens[i]], y[i, : y_lens[i]]], dim=0
            )

        all_xy_lens = prompt_lens + xy_lens
        all_xy_max_len = all_xy_lens.max()
        all_xy_embedded = torch.zeros(
            (batch_size, all_xy_max_len, self.embedding_dim),
            device=x.device,
        )
        attention_mask = torch.zeros((batch_size, all_xy_max_len), dtype=torch.bool, device=x.device)

        for i in range(batch_size):
            start_idx = all_xy_max_len - all_xy_lens[i]
            prompt_end = prompt_lens[i]
            xy_end = xy_lens[i]
            
            # 左填充 all_xy_embedded
            all_xy_embedded[i, start_idx:] = torch.cat(
                [prompt_xy_embedded[i, :prompt_end], xy_embedded[i, :xy_end]],
                dim=0,
            )
            
            # 创建对应的 attention_mask
            attention_mask[i, start_idx:] = True
        xy_dec = self.llm(
            inputs_embeds=all_xy_embedded,
            attention_mask=attention_mask,
        ).last_hidden_state

        total_loss = 0
        total_acc = 0
        total_tokens = 0
        for i in range(batch_size):
            # 计算当前样本的音频部分在 all_xy_embedded 中的起始和结束位置
            start_idx = all_xy_max_len - all_xy_lens[i]
            audio_start = start_idx + prompt_lens[i] + x_lens[i]
            audio_end = all_xy_max_len
            # 选择当前样本的音频部分输出
            audio_outputs = xy_dec[i, audio_start:audio_end].unsqueeze(0)  # 增加batch维度

            # 计算logits
            logits = self.predict_layer(audio_outputs)
            logits = logits.view(-1, self.speech_vocab_size)  # 调整形状为 (y_lens[i], speech_vocab_size)

            # 选择对应的targets
            current_targets = targets[i, : y_lens[i]]  # 形状为 (y_lens[i],)
            # 计算损失
            loss = F.cross_entropy(logits, current_targets, reduction="sum")
            total_loss += loss

            # 计算准确率
            acc = self.ar_accuracy_metric(logits.detach(), current_targets)
            total_acc += acc * y_lens[i]  # 乘以序列长度，因为准确率是平均值

            total_tokens += y_lens[i]

        # 计算平均损失和准确率
        avg_loss = total_loss / total_tokens
        avg_acc = total_acc / total_tokens

        return avg_loss, avg_acc
    def infer(
        self,
        x,
        x_lens,
        prompt_x,
        prompt_x_lens,
        prompt_y,
        prompt_y_lens,
        x_bert_feature,
        prompt_x_bert_feature,
        top_k: int = -100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
    ):
        """
        x: target text_ids - [batch_size, x_len]
        prompt_x: prompt text_ids - [batch_size, prompt_x_len]
        prompt_y: prompt semantic_ids - [batch_size, prompt_y_len]
        """
        batch_size = x.shape[0]

        prompt_x = self.text_embedding(prompt_x)
        prompt_y = self.speech_embedding(prompt_y)
        x = self.text_embedding(x)

        if x_bert_feature is not None and prompt_x_bert_feature is not None:
            x_bert_feature = self.bert_proj(x_bert_feature)
            prompt_x_bert_feature = self.bert_proj(prompt_x_bert_feature)
            x += x_bert_feature
            prompt_x += prompt_x_bert_feature

        prompt_lens = prompt_x_lens + prompt_y_lens
        prompt_xy_max_len = prompt_lens.max()
        prompt_xy_embedded = torch.zeros(
            (batch_size, prompt_xy_max_len, self.embedding_dim), device=x.device
        )
        for i in range(batch_size):
            prompt_xy_embedded[i, : prompt_lens[i]] = torch.cat(
                [prompt_x[i, : prompt_x_lens[i]], prompt_y[i, : prompt_y_lens[i]]],
                dim=0,
            )

        xy_lens = x_lens
        xy_max_len = xy_lens.max()
        xy_embedded = torch.zeros(
            (batch_size, xy_max_len, self.embedding_dim), device=x.device
        )
        for i in range(batch_size):
            xy_embedded[i, : xy_lens[i]] = x[i, : x_lens[i]]

        all_xy_lens = prompt_lens + xy_lens
        all_xy_max_len = all_xy_lens.max()
        all_xy_embedded = torch.zeros(
            (batch_size, all_xy_max_len, self.embedding_dim),
            device=x.device,
        )
        attention_mask = torch.zeros(
            (batch_size, all_xy_max_len), dtype=torch.bool, device=x.device
        )

        for i in range(batch_size):
            start_idx = all_xy_max_len - all_xy_lens[i]
            prompt_end = prompt_lens[i]
            xy_end = xy_lens[i]

            # 左填充 all_xy_embedded
            all_xy_embedded[i, start_idx:] = torch.cat(
                [prompt_xy_embedded[i, :prompt_end], xy_embedded[i, :xy_end]],
                dim=0,
            )

            # 创建对应的 attention_mask
            attention_mask[i, start_idx:] = True

        y = torch.LongTensor([[self.BOS]]).repeat(batch_size, 1)
        y_emb = self.speech_embedding(y)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1), dtype=torch.bool, device=attention_mask.device
                ),
            ],
            dim=1,
        )
        
        past_key_values = None
        inputs_embeds = torch.concat([all_xy_embedded, y_emb], dim=1)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        output_lens = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        for i in tqdm(range(1500)):
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

            logits = self.predict_layer(outputs.last_hidden_state[:, -1])
            samples = topk_sampling(logits, top_k=top_k, top_p=1.0, temperature=temperature)
            y = torch.cat([y, samples], dim=1)

            eos_mask = (torch.argmax(logits, dim=-1) == self.EOS) | (
                samples[:, 0] == self.EOS
            )
            finished = finished | eos_mask
            output_lens = torch.where(finished, output_lens, i)

            if early_stop_num != -1 and i > early_stop_num:
                print("use early stop num:", early_stop_num)
                break

            if finished.all():
                break
            inputs_embeds = self.speech_embedding(samples)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (batch_size, 1), dtype=torch.bool, device=attention_mask.device
                    ),
                ],
                dim=1,
            )
            past_key_values = outputs.past_key_values
            
        return y[:, 1:], output_lens

    def infer_panel(
        self,
        x,  #####全部文本token
        x_lens,
        prompts,  ####参考音频token
        bert_feature,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts

        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        # print(1111111,self.num_layers)
        cache = {
            "all_stage": self.num_layers,
            "k": [None] * self.num_layers,  ###根据配置自己手写
            "v": [None] * self.num_layers,
            # "xy_pos":None,##y_pos位置编码每次都不一样的没法缓存，每次都要重新拼xy_pos.主要还是写法原因，其实是可以历史统一一样的，但也没啥计算量就不管了
            "y_emb": None,  ##只需要对最新的samples求emb再拼历史的就行
            # "logits":None,###原版就已经只对结尾求再拼接了，不用管
            # "xy_dec":None,###不需要，本来只需要最后一个做logits
            "first_infer": 1,
            "stage": 0,
        }
        ###################  first step ##########################
        if y is not None:
            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.shape[1]
            prefix_len = y.shape[1]
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            cache["y_emb"] = y_emb
            ref_free = False
        else:
            y_emb = None
            y_len = 0
            prefix_len = 0
            y_pos = None
            xy_pos = x
            y = torch.zeros(x.shape[0], 0, dtype=torch.int, device=x.device)
            ref_free = True

        x_attn_mask_pad = F.pad(
            x_attn_mask,
            (0, y_len),  ###xx的纯0扩展到xx纯0+xy纯1，(x,x+y)
            value=True,
        )
        y_attn_mask = F.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).to(x.device)

        for idx in tqdm(range(1500)):
            xy_dec, _ = self.h((xy_pos, None), mask=xy_attn_mask, cache=cache)
            logits = self.ar_predict_layer(
                xy_dec[:, -1]
            )  ##不用改，如果用了cache的默认就是只有一帧，取最后一帧一样的
            # samples = topk_sampling(logits, top_k=top_k, top_p=1.0, temperature=temperature)
            if idx == 0:  ###第一次跑不能EOS否则没有了
                logits = logits[:, :-1]  ###刨除1024终止符号的概率
            samples = sample(
                logits[0],
                y,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.35,
                temperature=temperature,
            )[0].unsqueeze(0)
            # 本次生成的 semantic_ids 和之前的 y 构成新的 y
            # print(samples.shape)#[1,1]#第一个1是bs
            y = torch.concat([y, samples], dim=1)

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                # print(torch.argmax(logits, dim=-1)[0] == self.EOS, samples[0, 0] == self.EOS)
                stop = True
            if stop:
                # if prompts.shape[1] == y.shape[1]:
                #     y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                #     print("bad zero prediction")
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break

            ####################### update next step ###################################
            cache["first_infer"] = 0
            if cache["y_emb"] is not None:
                y_emb = torch.cat(
                    [cache["y_emb"], self.ar_audio_embedding(y[:, -1:])], dim=1
                )
                cache["y_emb"] = y_emb
                y_pos = self.ar_audio_position(y_emb)
                xy_pos = y_pos[:, -1:]
            else:
                y_emb = self.ar_audio_embedding(y[:, -1:])
                cache["y_emb"] = y_emb
                y_pos = self.ar_audio_position(y_emb)
                xy_pos = y_pos
            y_len = y_pos.shape[1]

            ###最右边一列（是错的）
            # xy_attn_mask=torch.ones((1, x_len+y_len), dtype=torch.bool,device=xy_pos.device)
            # xy_attn_mask[:,-1]=False
            ###最下面一行（是对的）
            xy_attn_mask = torch.zeros(
                (1, x_len + y_len), dtype=torch.bool, device=xy_pos.device
            )
        if ref_free:
            return y[:, :-1], 0
        return y[:, :-1], idx - 1

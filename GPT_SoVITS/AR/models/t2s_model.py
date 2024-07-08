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

    def forward(  # ! 需要一个bos标识符给y。
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

        y_mask = torch.arange(y.shape[1]).expand(batch_size, -1) < y_lens.unsqueeze(1)
        y = torch.where(y_mask, y, self.EOS)
        targets = torch.nn.functional.pad(y, (0, 1), value=self.EOS)
        y, targets = targets[:, :-1], targets[:, 1:]

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
        all_xy_mask = torch.zeros((batch_size, all_xy_max_len), dtype=torch.bool)

        for i in range(batch_size):
            all_xy_embedded[i, : all_xy_lens[i]] = torch.cat(
                [prompt_xy_embedded[i, : prompt_lens[i]], xy_embedded[i, : xy_lens[i]]],
                dim=0,
            )
            all_xy_mask[i, : all_xy_lens[i]] = True
            
        xy_dec = self.llm(
            inputs_embeds=all_xy_embedded,
            attention_mask=all_xy_mask,
        ).last_hidden_state
        total_loss = 0
        total_acc = 0
        total_tokens = 0
        for i in range(batch_size):
            # 选择当前样本的音频部分输出
            audio_start = prompt_lens[i] + x_lens[i]
            audio_end = audio_start + y_lens[i]
            audio_outputs = xy_dec[i, audio_start:audio_end].unsqueeze(
                0
            )  # 增加batch维度

            # 计算logits
            logits = self.predict_layer(audio_outputs)
            logits = logits.view(
                -1, self.speech_vocab_size
            )  # 调整形状为 (y_lens[i], speech_vocab_size)

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
            xy_embedded[i, : x_lens[i]] = x[i, : x_lens[i]]

        all_xy_lens = prompt_lens + xy_lens
        all_xy_max_len = all_xy_lens.max()
        all_xy_embedded = torch.zeros(
            (batch_size, all_xy_max_len, self.embedding_dim),
            device=x.device,
        )
        all_xy_mask = torch.zeros((batch_size, all_xy_max_len), dtype=torch.bool)

        for i in range(batch_size):
            all_xy_embedded[i, : all_xy_lens[i]] = torch.cat(
                [prompt_xy_embedded[i, : prompt_lens[i]], xy_embedded[i, : x_lens[i]]],
                dim=0,
            )
            all_xy_mask[i, : all_xy_lens[i]] = True

        y = torch.zeros((batch_size, 1), dtype=torch.long, device=x.device) # ! 需要一个bos标识符
        stop = False
        for _ in tqdm(range(1500)):
            y_emb = self.speech_embedding(y)
            xy_pos = torch.cat([all_xy_embedded, y_emb], dim=1)
            xy_mask = torch.cat([all_xy_mask, torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)], dim=1)

            xy_dec = self.llm(
                inputs_embeds=xy_pos,
                attention_mask=xy_mask,
            ).last_hidden_state

            logits = self.predict_layer(xy_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature
            )

            if early_stop_num != -1 and y.shape[1] > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                if y.shape[1] == 1:
                    y = torch.cat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{y.shape[1]}]")
                break

            y = torch.cat([y, samples], dim=1)

        return y[:, 1:]
    
    def forward_deprecated(self, x, x_lens, y, y_lens):
        """
        x: bert_feature
        y: semantic_ids
        """
        x = x.transpose(1, 2)  # 确保维度正确
        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        # y_emb = self.ar_audio_embedding(y)
        # y_pos = self.ar_audio_position(y_emb)
        y_pos = y  

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_head, -1, -1)
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask
        print(xy_attn_mask)
        print(xy_attn_mask.shape)
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)
        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(0, 2, 1)
        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, ���以用 sum
        loss = F.cross_entropy(logits, targets, reduction="sum")
        acc = self.ar_accuracy_metric(logits.detach(), targets).item()
        return loss, acc

    # ! GPT-SOVITS2更改
    # ! x维度和prompts做embedding后统一，GPTSOVITS中使用的方法是训练两个embedding，维度都是512，用一个MLP将bert的1024变成512.并且音素的embedding维度也是512
    # ! 而hubert则经过码本后重新再进入embedding变成512维度。
    # ! 在GPT SOVITS2中换了一个方法。
    # ! BGE m3 维度为1024. 并且GPT SOVITS2中没有音素。
    # ! hubert是768维度。
    # ! 考虑几个选择: 1. 训练一个MLP层维度为(1024, 768)将BGE m3变成768维度。或者直接(1024, 512)和(768, 512)统一到512维度，这样和GPT SOVITS的维度保持一致
    # ! 2. 直接去掉BGE m3末尾的256维度，直接变成768维度，这样和hubert的维度保持一致。
    # ! 3. 用降维方法例如PCA将BGE m3降维到768维度。
    # ! 目前优先选择第一种方法，训练一个MLP层。
    # ! 将码本变大，比如总数16384
    # ! 理论上可以做到把码本完全去掉。问题在于: 去掉之后，如何学习在哪里结束语音。(否决，决定不去掉码本)
    def infer_deprecated(
        self,
        x,  # ! x是全部文本的token (len, 1) 在GPT SOVITS2���无意义。
        x_lens,
        prompts,  # ! prompts是参考音频的码本结果，后续考虑改成hubert直出不用经过encode码本。
        bert_feature,
        top_k: int = -100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
    ):
        # x = self.ar_text_embedding(x)
        # x = x + self.bert_proj(bert_feature.transpose(1, 2))
        # x = self.ar_text_position(x)

        x = self.bert_proj(
            bert_feature.transpose(1, 2)
        )  # ! GPT SOVITS2的处理。记得MLP换成(1024, 768)

        # AR Decoder
        y = prompts
        prefix_len = y.shape[1]
        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        for _ in tqdm(range(1500)):
            y_emb = self.ar_audio_embedding(
                y
            )  # ! 为什么不直接解码码本呢？这个embedding是为了从码本idx到embedding的映射，那么直接解码码本不就行了吗？注意！解码码本不是直接vq_model.decode而是vq_model.quantizer.decode()
            y_pos = self.ar_audio_position(
                y_emb
            )  # ! 换成RoPE，并且每次循环只推理最新的token而不是全部token，这很重要因为GPT从原理上来说为了更好推理下一个token，embedding就会被设计���绝对位置或者这种RoPE。
            # ! ROPE还支持任意长度，这是比绝对位置更好的地方。
            # ! y_emb和y_pos移动到循环外面，然后在下面每次推理出新的token后在末尾添加。这样应该会极快地增加推理速度。
            # x 和逐渐增长的 y 一起输入给模型
            # ! 换成x1,y1,x2,推理y2，能不能让声线融合变得更合理？
            # ! 按照目前的设计是x1,x2,y1,推理y2。这种情况下如果有多个x,y的话就变成了x1,x2,x3,x4,y1,y2,y3推理y4。似乎没有x1,y1,x2,y2,x3,y3,x4,推理y4合理。
            # ! 但是在VALLE的设计中是直接x2,y1,推理y2.所以在GPTSOVITS中才有x1,x2,y1,推理y2的设计。
            # ! !考虑RoPE对整个序列xy_pos做，而不是对x和y分别做。
            xy_pos = torch.concat([x, y_pos], dim=1)
            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).to(
                y.device
            )

            xy_dec, _ = self.h(
                (xy_pos, None),
                mask=xy_attn_mask,
            )
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature
            )
            # ! 按照上面说的，对samples算一次解码+RoPE，然后拼接到y_pos后面，这样就可以一直推理下去。
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                # print(torch.argmax(logits, dim=-1)[0] == self.EOS, samples[0, 0] == self.EOS)
                stop = True
            if stop:
                if prompts.shape[1] == y.shape[1]:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break
            # 本次生成的 semantic_ids 和之前的 y 构成新的 y
            # print(samples.shape)#[1,1]#第一个1是bs
            # import os
            # os._exit(2333)
            y = torch.concat([y, samples], dim=1)
        return y

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(
            y_mask_int, (0, 1), value=1
        )
        # 错位
        return targets[:, :-1], targets[:, 1:]

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
            "y_emb": None,  ##只需要对最新的samples求emb��再拼历史的就行
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

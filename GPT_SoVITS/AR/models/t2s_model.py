# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from AR.models.utils import (
    dpo_loss,
    get_batch_logps,
    logits_to_probs,
    make_pad_mask,
    make_reject_y,
    multinomial_sample_one_no_sync,
    sample,
    topk_sampling,
)
from AR.modules.embedding import SinePositionalEmbedding, TokenEmbedding
from AR.modules.transformer import (
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer,
)

default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024,
}


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        # should be same as num of kmeans bin
        # assert self.EOS == 1024
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        # self.ar_text_embedding = TokenEmbedding( # ! 在GPT SOVITS2中不需要自己训练embedding。
        #     self.embedding_dim, self.phoneme_vocab_size, self.p_dropout
        # )
        # self.ar_text_position = SinePositionalEmbedding(
        #     self.embedding_dim, dropout=0.1, scale=False, alpha=True
        # )
        # self.ar_audio_embedding = TokenEmbedding(
        #     self.embedding_dim, self.vocab_size, self.p_dropout
        # )
        # self.ar_audio_position = SinePositionalEmbedding(
        #     self.embedding_dim, dropout=0.1, scale=False, alpha=True
        # )

        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None,
        )

        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")

        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )

    def make_input_data(self, x, x_lens, y, y_lens, bert_feature):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

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
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)

        return xy_pos, xy_attn_mask, targets

    def forward_dpo(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """

        reject_y, reject_y_lens = make_reject_y(y, y_lens)

        xy_pos, xy_attn_mask, targets = self.make_input_data(
            x, x_lens, y, y_lens, bert_feature
        )

        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        x_len = x_lens.max()
        logits = self.ar_predict_layer(xy_dec[:, x_len:])

        ###### DPO #############
        reject_xy_pos, reject_xy_attn_mask, reject_targets = self.make_input_data(
            x, x_lens, reject_y, reject_y_lens, bert_feature
        )

        reject_xy_dec, _ = self.h(
            (reject_xy_pos, None),
            mask=reject_xy_attn_mask,
        )
        x_len = x_lens.max()
        reject_logits = self.ar_predict_layer(reject_xy_dec[:, x_len:])

        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum

        loss_1 = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction="sum")
        acc = self.ar_accuracy_metric(logits.permute(0, 2, 1).detach(), targets).item()

        A_logits, R_logits = get_batch_logps(
            logits, reject_logits, targets, reject_targets
        )
        loss_2, _, _ = dpo_loss(A_logits, R_logits, 0, 0, 0.2, reference_free=True)

        loss = loss_1 + loss_2

        return loss, acc

    def forward(self, x, x_lens, y, y_lens):
        """
        x: bert_feature
        y: semantic_ids
        """
        x = x.transpose(1, 2) # 确保维度正确
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
        y_pos = y # ! 这里等数据处理的适合

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
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)
        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(0, 2, 1)
        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum
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
    def infer(
        self,
        x,  # ! x是全部文本的token (len, 1) 在GPT SOVITS2中无意义。
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
            )  # ! 换成RoPE，并且每次循环只推理最新的token而不是全部token，这很重要因为GPT从原理上来说为了更好推理下一个token，embedding就会被设计成绝对位置或者这种RoPE。
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
            "y_emb": None,  ##只需要对最新的samples求emb，再拼历史的就行
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

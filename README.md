# GPT-SoVITS2

[**English**](./docs/en/README.md) | [**日本語**](./docs/ja/README.md) | [**한국어**](./docs/ko/README.md) | [**Türkçe**](./docs/tr/README.md) | [**Español**](./docs/es/README.md) | [**Português**](./docs/pt/README.md)

此名字已获GPT-SoVITS作者[花儿不哭](https://space.bilibili.com/5760446?spm_id_from=333.337.0.0)的许可。
### 此项目还在开发当中，基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 改良而来，主要改进点如下：

1. **原生支持多语言**：不局限于中日英，而是世界上任何语言。
2. **不再需要指定语言**：任何时候都是多语言，可以随意地多语言混合说话。
3. **多语言文字情感提取**：对语言的情感分析更加到位，使得说话更加富有情感。
4. **Zero Shot 的提升**：现在不再推荐微调模型，而是只用几秒的目标音频来直接 zero shot。
5. **参考音频融合**：可以上传多条参考音频，得到的声音将是多个音频进行融合的结果。
6. **更快速的推理**: 将positional embedding改成RoPE，每次推理下一个token无需重新计算整个序列的embedding。

### **数据及合作征集**：现正在征集数据。QQ 1715069210，数据集合格的情况下会在项目进行 credit。

#### 目前在整理改动思路于源代码中，搜索# ! 即可找到注释。如果感兴趣希望添加以上QQ交流。

### 改动列表

#### 码本的变动
单码本-> 2码本/4码本
#### GPT改动
换成qwen2-0.3b
#### 音频编码的变动
cnhubert-> ~~w2v-bert-2.0(暂定,此为meta做的目前训练集最夸张4.6m小时多语言预训练。如果结果像外国人说中文就换cnhubert-large)~~/cnhubert-large/mHubert-147
我发现w2v-bert-2.0训练有点难，mHubert-147训练会比较简单，体积差了四倍，而且实测fp16直接会炸，只能fp32.并且mHubert已经够大了(600MB)
#### 文本编码的改动
去掉音素以及相应embedding
cn-roberta ->  BGE-m3
#### 位置编码的改动
文本与语音编码各自做sinusoidal->全局做RoPE embedding.
#### xy结合embedding改动(实验性)
从原本的
x1+x2+y1->y2
变成
x1+y1+x2->y2
并且整条序列共享一个RoPE embedding
理论上来讲这样更加适合扩展更多目标音频来进行声线融合
比如可以
x1+y1+x2+y2+x3+y3+x4+y4+x5->y5
这样感觉会比
x1+x2+x3+x4+x5+y1+y2+y3+y4->y5
更自然一点.无法严格证明
#### 维度的改动
MLP(768, 512) -> ~~无MLP直接1024维度。因为w2v-bert-2.0和bge-m3都是1024维，天生一对啊.~~ MLP(1024, 768)
#### 训练方法改动
纯自回归->自回归+同一speaker下zero shot训练样本回归
#### vits改动
可能想办法把维度扩大.(256->512)
#### 格式
统一~~半精度~~单精度(实测之后发现半精度会炸), hubert 16000采样 vits 32000采样 对所有音频做响度统一
#### 总结
其实总体上来说,改动基本都是
1. 用上了更先进的预训练模型
2. 因为更先进的模型更大,所以将本来的维度也扩大.
3. 因为注重zero shot,所以训练方法会加上zero shot的训练.
4. 原本代码只有中文才拿到bert,换成BGE m3这种多语言embedding很自然地就全语言无缝提取
5. 原本只有单码本甚至码本大小只有1024,导致原本提取hubert特征引导能力不够强.换成双码本信息量直接变成1024^2 = 1048576,四码本就更夸张但应该没那么多数据可以炼,先一步一步尝试
6. 原本速度慢有个原因在于GPT每次都要重新计算整个序列的embedding和计算positional embedding.但如果换成RoPE就没这个缺点了.
7. 原本是没有关注声线融合这件事的,后来我改了个branch在原本的GPT-SoVITS上实现了声线融合,但是很明显最开始设计就完全没有这个目标.花佬视频里面提到的是是否可以在GPT部分用的参考音频转的声音feature和在vits中的参考音频不一样,可以做声线融合.但是我的实现是直接在两个部分都有多个音频.
8. 改了原本看着我认为不合理的地方. 比如都用了hubert作为音频的embedding了,码本作为token了,为什么还要再来一个ar_audio_embedding.并且原来因为有音素所以要给音素配一个embedding,有很多个单独的embedding在那训练然而已经用了bert和hubert了. 并且单独的文本embedding和音频做embedding是不会考虑到上下文的音频和文本的,不如直接输入进GPT靠attention找关系.

都看到这里了,说明你看懂了,欢迎来加入这个项目!

**QQ: 1715069210**

**微信: JunityZ**

#### Quick Note
今天看了很多论文包括VALLE2的论文又有很多新的想法.有一个很重要的事情就是目前这个ar_audio_embedding和ar_text_embedding其实是一个历史遗留问题了。

因为audioLM率先使用了hubert+kmeans获得token，但是因为kmean量化学习不需要学整体数据，而是直接从hubert分布上学习。所以会加一个后续的embedding。

但如果用了vq，vq本身就已经做了学习了，所以vq不需要再加一个embedding，这里历史遗留问题一直都有加embedding，虽然影响应该不会很大，但是去掉的话会合理很多。

还有就是 audio lm 同时用了semantic和acoustic，分别通过hubert和soundstream。但其实GPT SoVITS也有这个，meltransferencoder获得acoustic，而hubert获得semantic，所以非常巧。

而VALLE系一般用的是EnCodec，EnCodec是直接从音频得到token，才会需要再做一个embedding,因为本身就没出embedding，但是很明显用hubert就不需要了，因为hubert的输出就是embedding。

反过来，我们用hubert embedding得到token，而EnCodec是得到token之后反过来做embedding。

所以原版的GPTSoVITS包括之前参考的AUdio LM更像是参考了基于EnCodec系列TTS的做法，但实际上这两个是不一样的。

#### TODO
重写量化, 直接调用vector-quantize-pytorch的Group Residual VQ

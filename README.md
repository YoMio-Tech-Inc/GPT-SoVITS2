# GPT-SoVITS2

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

### 网络的改动

#### 码本的变动
单码本-> 2码本/4码本
#### 音频编码的变动
cnhubert-> cnhubert-large/mHubert-146/w2v-bert-2.0
#### 文本编码的改动
去掉音素以及相应embedding
cn-roberta ->  BGE-m3
#### 位置编码的改动
文本与语音编码各自做sinusoidal->全局做RoPE embedding.
#### 维度的改动
BGE-m3(1024), mHubert-146(768), w2v-bert-2.0(1024) 取低者。
#### 训练方法改动
纯自回归->自回归+同一speaker下one shot训练样本回归
#### vits decoder改动
维度和上述一样变成768/1024取决于用mHubert or w2v-bert-2.0
#### 格式
统一半精度, hubert 16000采样 vits 32000采样
# GPT-SoVITS2

This name has been approved by the author of GPT-SoVITS, [花儿不哭](https://space.bilibili.com/5760446?spm_id_from=333.337.0.0).
### This project is still under development and is an improvement based on [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). The main improvements are as follows:
|GPT-SoVITS|GPT-SoVITS2|
|:----:|:----:|
|**Text**|**Text**|
|Text->Phone|Text->BPE|
|Phone->Embedding|BPE->Embedding|
|Roberta-Chinese|BGE-M3|
|**Speech Encoder**|**Speech Encoder**|
|Hubert|$S^3$|
|VQ|$S^3$->Embedding|
|1024 Speech Token|4096 Speech Token|
|**AR**|**AR**|
|Old-Style GPT|Qwen2-0.3b|
|**Speech Decoder**|**Speech Decoder**|
|VITS|VITS2|
|Hidden Size 192|Hidden Size 256|
|2 Heads|4 Heads|
|Inter Size 768|Inter Size 1024|
|**Training**|**Training**|
|No Zero-Shot Training|Inference Different Speech with same Speaker|
|ZH,EN,JA|Multi-Lingual|
|2k hours|Not Sure Yet|

### **Data and collaboration solicitation**: Data is currently being collected. QQ 1715069210, if the data set meets the requirements, credit will be given in the project.

#### Currently organizing modification ideas in the source code. Search for # ! to find comments. If interested, feel free to contact via the above QQ.

### List of Changes

#### Codebook changes
Single codebook -> 2 codebooks/4 codebooks
#### GPT changes
Replaced with qwen2-0.3b
#### Audio encoding changes
cnhubert -> ~~w2v-bert-2.0 (tentative, this is the most exaggerated 4.6m-hour multilingual pre-training done by meta. If the result sounds like a foreigner speaking Chinese, it will be replaced with cnhubert-large)~~/cnhubert-large/mHubert-147
I found that training w2v-bert-2.0 is a bit difficult, training mHubert-147 is relatively easier, the size difference is fourfold, and in real tests, fp16 directly crashes, so only fp32 can be used. Also, mHubert is already large enough (600MB).
#### Text encoding changes
Remove phonemes and corresponding embeddings
cn-roberta -> BGE-m3
#### Positional encoding changes
Text and speech encoding each do sinusoidal -> globally do RoPE embedding.
#### xy combined embedding changes (experimental)
From the original
x1+x2+y1 -> y2
changed to
x1+y1+x2 -> y2
and the entire sequence shares a RoPE embedding.
Theoretically, this is more suitable for expanding more target audio for voice line fusion.
For example:
x1+y1+x2+y2+x3+y3+x4+y4+x5 -> y5
feels more natural than
x1+x2+x3+x4+x5+y1+y2+y3+y4 -> y5
Cannot be strictly proven.
#### Dimensional changes
MLP(768, 512) -> ~~No MLP directly 1024 dimensions. Since both w2v-bert-2.0 and bge-m3 are 1024 dimensions, a perfect match.~~ MLP(1024, 768)
#### Training method changes
Pure autoregressive -> autoregressive + zero shot training sample regression under the same speaker
#### vits changes
Possibly find a way to increase the dimensions. (256 -> 512) VITS -> VITS2 (mainly adding transformer block to the flow model).
#### Format
Unified ~~half precision~~ single precision (real tests found that half precision crashes), hubert 16000 sampling, vits 32000 sampling, loudness normalization for all audio.
#### Summary
Overall, the changes are mainly:
1. Using more advanced pre-trained models.
2. Since more advanced models are larger, the original dimensions are also expanded.
3. Due to the focus on zero shot, the training method includes zero shot training.
4. The original code only had bert for Chinese, changing to BGE m3, a multilingual embedding, naturally extracts all languages seamlessly.
5. The original only had a single codebook with a size of only 1024, making the original hubert feature extraction guidance insufficient. Changing to double codebooks increases the information amount to 1024^2 = 1048576, and four codebooks are even more exaggerated, but should be tried step by step due to limited data.
6. One reason for the original slow speed is that GPT has to recompute the entire sequence's embedding and positional embedding each time. Switching to RoPE eliminates this drawback.
7. Originally, voice line fusion was not considered, but later I modified a branch in the original GPT-SoVITS to achieve voice line fusion, although it was not the initial design goal. In 花's video, it was mentioned that whether the reference audio used in the GPT part and the reference audio in vits could be different for voice line fusion. However, my implementation has multiple audios in both parts.
8. Changed parts that I thought were unreasonable. For example, since hubert is used as the audio embedding and codebooks as tokens, why add an ar_audio_embedding. Also, since there were phonemes, an embedding had to be assigned to them, resulting in many separate embeddings training, although bert and hubert were already used. Moreover, separate text and audio embeddings would not consider the context of audio and text, better to directly input into GPT and use attention to find relationships.

If you've read this far, it means you understand, and you're welcome to join this project!

**QQ: 1715069210**

**WeChat: JunityZ**

#### Quick Note
Today I read many papers, including the VALLE2 paper, and had many new ideas. One important issue is that the current ar_audio_embedding and ar_text_embedding are historical legacy problems.

Because audioLM first used hubert+kmeans to obtain tokens, but since kmeans quantization learning does not need to learn the overall data but directly learns from the hubert distribution, a subsequent embedding is added.

But if vq is used, vq itself already learns, so there is no need to add another embedding. Here, historical issues have always added embedding, although the impact may not be significant, removing it would be much more reasonable.

Moreover, audio lm used both semantic and acoustic through hubert and soundstream, respectively. But GPT SoVITS also has this, with meltransferencoder obtaining acoustic and hubert obtaining semantic, which is very coincidental.

VALLE series generally use EnCodec, which directly obtains tokens from audio and needs another embedding since it does not output embedding initially. But using hubert does not need this because hubert's output is already embedding.

Conversely, we obtain tokens with hubert embedding, while EnCodec obtains tokens first and then performs embedding.

Therefore, the original GPTSoVITS and the previous referenced AUdio LM seem to be based on EnCodec series TTS methods, but they are actually different.

#### TODO
Rewrite quantization, directly call vector-quantize-pytorch's Group Residual VQ

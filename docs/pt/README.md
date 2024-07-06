# GPT-SoVITS2

Este nome foi autorizado pelo autor do GPT-SoVITS, [花儿不哭](https://space.bilibili.com/5760446?spm_id_from=333.337.0.0).
### Este projeto ainda está em desenvolvimento, baseado no [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), com as seguintes principais melhorias:
|GPT-SoVITS|GPT-SoVITS2|
|:----:|:----:|
|**Texto**|**Texto**|
|Texto->Telefone|Texto->BPE|
|Telefone->Embedding|BPE->Embedding|
|Roberta-Chinês|BGE-M3|
|**Codificador de Fala**|**Codificador de Fala**|
|Hubert|$S^3$|
|VQ|$S^3$->Embedding|
|1024 Token de Fala|4096 Token de Fala|
|**AR**|**AR**|
|GPT Estilo Antigo|Qwen2-0.3b|
|**Decodificador de Fala**|**Decodificador de Fala**|
|VITS|VITS2|
|Tamanho Oculto 192|Tamanho Oculto 256|
|2 Cabeças|4 Cabeças|
|Tamanho Interno 768|Tamanho Interno 1024|
|**Treinamento**|**Treinamento**|
|Sem Treinamento Zero-Shot|Inferência de Voz Diferente com o Mesmo Falante|
|ZH,EN,JA|Multilíngue|
|2000 horas|Ainda Não Definido|

1. **Suporte nativo a múltiplos idiomas**: Não se limita a chinês, japonês e inglês, mas a qualquer idioma do mundo.
2. **Não é necessário especificar o idioma**: Sempre é multilíngue, permitindo misturar idiomas livremente.
3. **Extração de emoções em múltiplos idiomas**: Análise mais precisa das emoções na linguagem, tornando a fala mais expressiva.
4. **Melhoria no Zero Shot**: Agora não se recomenda ajustar modelos, mas usar apenas alguns segundos de áudio-alvo para um Zero Shot direto.
5. **Fusão de áudios de referência**: É possível enviar vários áudios de referência, e a voz resultante será uma fusão desses áudios.
6. **Inferência mais rápida**: Mudança do positional embedding para RoPE, eliminando a necessidade de recalcular todo o embedding da sequência a cada token.

### **Coleta de dados e colaboração**: Atualmente estamos coletando dados. Contato QQ 1715069210. Dados qualificados receberão crédito no projeto.

#### Atualmente estamos organizando as ideias das mudanças no código-fonte, procure por # ! para encontrar comentários. Se estiver interessado, adicione o QQ acima para trocar ideias.

### Lista de mudanças

#### Mudanças no código
~~De um único código -> 2 códigos/4 códigos.~~

O tamanho do vocabulário do $S^3$ é 4096 em um único código.
#### Mudanças no GPT
Mudança para qwen2-0.3b.
#### Mudanças na codificação de áudio
cnhubert -> ~~w2v-bert-2.0 (provisório, o conjunto de treinamento mais extenso de 4.6m horas multilíngues da Meta. Se resultar em pronúncias estranhas, mudar para cnhubert-large)~~ ~~/ cnhubert-large / mHubert-147.~~
~~Descobri que treinar w2v-bert-2.0 é difícil, enquanto que mHubert-147 é mais fácil, embora seja quatro vezes maior e fp16 falhe, só fp32 funciona. Além disso, mHubert é suficientemente grande (600MB).~~

Usando o codificador $S^3$ do CosyVoice, com uma camada de embedding externa.
#### Mudanças na codificação de texto
Eliminar fonemas e seus correspondentes embeddings.

Fonema -> Tokenização BPE.

cn-roberta -> BGE-m3.
#### Mudanças no posicionamento de embeddings
Separar codificação de texto e áudio em sinusoidal -> fazer tudo com RoPE embedding.
#### Mudanças na combinação de embeddings (experimental)
De:
x1 + x2 + y1 -> y2
Para:
x1 + y1 + x2 -> y2
Toda a sequência compartilha um RoPE embedding.
Teoricamente, isso é mais adequado para fundir múltiplos áudios de referência.
Exemplo:
x1 + y1 + x2 + y2 + x3 + y3 + x4 + y4 + x5 -> y5
Isso parece mais natural que:
x1 + x2 + x3 + x4 + x5 + y1 + y2 + y3 + y4 -> y5
Sem uma prova estrita.
#### Mudanças nas dimensões
MLP (768, 512) -> ~~Sem MLP, diretamente 1024 dimensões. w2v-bert-2.0 e bge-m3 são 1024 dimensões, uma combinação perfeita.~~ MLP (1024, 768).
#### Mudanças no método de treinamento
De autoregressivo puro -> autoregressivo + zero shot com amostras do mesmo locutor.
#### Mudanças no vits
Ampliar dimensões se possível (256 -> 512). VITS -> VITS2 (principalmente adicionando um transformer block ao modelo de fluxo)
#### Formato
Unificar ~~precisão média~~ precisão simples (a precisão média falhava), hubert a 16000 amostras, vits a 32000 amostras, unificar a sonoridade de todos os áudios.
#### Resumo
Em resumo, as mudanças são:
1. Uso de modelos pré-treinados mais avançados.
2. Modelos maiores requerem maiores dimensões.
3. Foco em zero shot, adicionando treinamento específico.
4. Substituição de modelos monolíngues por embeddings multilíngues como BGE m3.
5. Aumento de códigos, de um para dois, aumentando significativamente a informação.
6. RoPE elimina a necessidade de recalcular embeddings, acelerando o processo.
7. Fusão de vozes, tanto em GPT quanto em vits, implementando múltiplos áudios de referência.
8. Correção de inconsistências, otimizando embeddings e eliminando redundâncias desnecessárias.

Se você entendeu até aqui, junte-se ao projeto!

**QQ: 1715069210**

**WeChat: JunityZ**

#### Nota rápida
Hoje revisei vários artigos, incluindo o de VALLE2, e tive novas ideias. Um ponto importante é que os embeddings ar_audio e ar_text são problemas históricos.

AudioLM utilizou primeiro hubert+kmeans para obter tokens, e kmeans não requer aprender de todos os dados, apenas do distribuidor hubert, por isso adiciona um embedding posterior.

No entanto, usando vq, já se fez o aprendizado, então vq não precisa de um embedding adicional. Esta redundância histórica, embora de pouco impacto, deveria ser eliminada.

AudioLM utilizou semantic e acoustic com hubert e soundstream, respectivamente. GPT SoVITS também usa meltransferencoder para acústico e hubert para semântico, sendo muito similar.

VALLE geralmente utiliza EnCodec, que obtém tokens diretamente do áudio e depois faz o embedding, mas com hubert isso não é necessário, já que hubert fornece embeddings.

Portanto, GPTSoVITS e Audio LM parecem seguir a abordagem de TTS baseada em EnCodec, mas são diferentes.

#### Tarefas pendentes
Reescrever a quantização, utilizando vector-quantize-pytorch Group Residual VQ.

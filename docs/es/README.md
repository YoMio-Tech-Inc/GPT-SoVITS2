# GPT-SoVITS2

Este nombre ha sido autorizado por el autor de GPT-SoVITS, [花儿不哭](https://space.bilibili.com/5760446?spm_id_from=333.337.0.0).
### Este proyecto aún está en desarrollo, basado en [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), con las principales mejoras siguientes:

1. **Soporte nativo multilingüe**: No limitado a chino, japonés e inglés, sino a cualquier idioma del mundo.
2. **No es necesario especificar el idioma**: Siempre es multilingüe, permitiendo mezclar idiomas libremente.
3. **Extracción de emociones en múltiples idiomas**: Mejor análisis de las emociones en el lenguaje, haciendo que el habla sea más expresiva.
4. **Mejora en Zero Shot**: Ahora no se recomienda ajustar modelos, sino usar solo unos segundos de audio objetivo para un Zero Shot directo.
5. **Fusión de audios de referencia**: Se pueden subir múltiples audios de referencia y la voz resultante será una fusión de estos.
6. **Inferencia más rápida**: Cambiar el positional embedding a RoPE, eliminando la necesidad de recalcular todo el embedding de la secuencia con cada token.

### **Recopilación de datos y colaboración**: Actualmente estamos recopilando datos. Contacto QQ 1715069210. Los datos calificados recibirán crédito en el proyecto.

#### Actualmente estamos organizando las ideas de los cambios en el código fuente, busca # ! para encontrar comentarios. Si estás interesado, añade el QQ anterior para intercambiar ideas.

### Lista de cambios

#### Cambios en el código
De un solo código -> 2 códigos/4 códigos.
#### Cambios en GPT
Cambiar a qwen2-0.3b.
#### Cambios en la codificación de audio
cnhubert -> ~~w2v-bert-2.0 (provisional, el conjunto de entrenamiento más extenso de 4.6m horas multilingües de Meta. Si resulta en pronunciaciones extrañas, cambiar a cnhubert-large)~~ / cnhubert-large / mHubert-147.
Descubrí que entrenar w2v-bert-2.0 es difícil, mientras que mHubert-147 es más sencillo, aunque es cuatro veces más grande y fp16 falla, solo fp32 funciona. Además, mHubert es suficientemente grande (600MB).
#### Cambios en la codificación de texto
Eliminar fonemas y sus correspondientes embeddings.
cn-roberta -> BGE-m3.
#### Cambios en el posicionamiento de embeddings
Separar codificación de texto y audio en sinusoidal -> hacer todo con RoPE embedding.
#### Cambios en la combinación de embeddings (experimental)
De:
x1 + x2 + y1 -> y2
A:
x1 + y1 + x2 -> y2
Toda la secuencia comparte un RoPE embedding.
Teóricamente, esto es más adecuado para fusionar múltiples audios de referencia.
Ejemplo:
x1 + y1 + x2 + y2 + x3 + y3 + x4 + y4 + x5 -> y5
Esto parece más natural que:
x1 + x2 + x3 + x4 + x5 + y1 + y2 + y3 + y4 -> y5
Sin una prueba estricta.
#### Cambios en las dimensiones
MLP (768, 512) -> ~~Sin MLP, directamente 1024 dimensiones. w2v-bert-2.0 y bge-m3 son 1024 dimensiones, una pareja perfecta.~~ MLP (1024, 768).
#### Cambios en el método de entrenamiento
De autoregresivo puro -> autoregresivo + zero shot con muestras del mismo speaker.
#### Cambios en vits
Ampliar dimensiones si es posible (256 -> 512).
#### Formato
Unificar ~~precisión media~~ precisión simple (la precisión media fallaba), hubert a 16000 muestras, vits a 32000 muestras, unificar la sonoridad de todos los audios.
#### Resumen
En resumen, los cambios son:
1. Uso de modelos preentrenados más avanzados.
2. Modelos más grandes requieren mayores dimensiones.
3. Enfoque en zero shot, añadiendo entrenamiento específico.
4. Sustitución de modelos monolingües por embeddings multilingües como BGE m3.
5. Aumento de códigos, de uno a dos, incrementando significativamente la información.
6. RoPE elimina la necesidad de recalcular embeddings, acelerando el proceso.
7. Fusión de voces, tanto en GPT como en vits, implementando múltiples audios de referencia.
8. Corrección de inconsistencias, optimizando embeddings y eliminando redundancias innecesarias.

Si has entendido hasta aquí, únete al proyecto!

**QQ: 1715069210**

**WeChat: JunityZ**

#### Nota rápida
Hoy revisé varios artículos, incluido el de VALLE2, y tuve nuevas ideas. Un punto importante es que los embeddings ar_audio y ar_text son problemas heredados históricos.

AudioLM utilizó primero hubert+kmeans para obtener tokens, y kmeans no requiere aprender de todos los datos, solo del distribuidor hubert, por lo que añade un embedding posterior.

Sin embargo, usando vq, ya se ha hecho el aprendizaje, por lo que vq no necesita un embedding adicional. Esta redundancia histórica, aunque de poco impacto, se debería eliminar.

AudioLM utilizó semantic y acoustic con hubert y soundstream, respectivamente. GPT SoVITS también usa meltransferencoder para acústico y hubert para semántico, siendo muy similar.

VALLE generalmente utiliza EnCodec, que obtiene tokens directamente del audio y luego hace el embedding, pero con hubert esto no es necesario, ya que hubert proporciona embeddings.

Por lo tanto, GPTSoVITS y AUdio LM parecen seguir el enfoque de TTS basado en EnCodec, pero son diferentes.

#### Tareas pendientes
Reescribir la cuantización, utilizando vector-quantize-pytorch Group Residual VQ.

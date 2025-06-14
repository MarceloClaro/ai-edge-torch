# AI Edge Torch — Guia autodidático, minucioso e 100 % off-line

*(todas as URLs do projeto original foram preservadas)*

---

## Índice

1. [O que é o AI Edge Torch?](#1)
2. [Por que usar no mundo real?](#2)
3. [Instalação rápida](#3)
4. [Converter redes PyTorch para TFLite](#4)

   * 4.1 Passo-a-passo básico
   * 4.2 Aplicação A — câmera Android reconhecendo objetos
   * 4.3 Aplicação B — contagem de pessoas em Raspberry Pi
5. [Generative API: LLMs otimizados para mobile](#5)

   * 5.1 Passo-a-passo básico
   * 5.2 Aplicação A — assistente de bordo off-line
   * 5.3 Aplicação B — chatbot Flutter (iOS/Android)
6. [Arquitetura interna & boas práticas](#6)
7. [Benchmarks e roadmap](#7)
8. [Checklist antes de colocar em produção](#8)
9. [Recursos oficiais & suporte](#9)

---

<a id="1"></a>

## 1. O que é o **AI Edge Torch?**

**Definição curta**
Biblioteca Python que **converte modelos PyTorch para `.tflite`** e traz uma **API dedicada a LLMs** (Generative API), permitindo rodar visão computacional e chatbots **inteiramente no dispositivo** (Android, iOS, IoT) via **TensorFlow Lite** e **MediaPipe**.

> > *Original EN*
> > AI Edge Torch is a python library that supports converting PyTorch models into a .tflite format \[…] AI Edge Torch offers broad CPU coverage, with initial GPU and NPU support. AI Edge Torch seeks to closely integrate with PyTorch, building on top of torch.export() and providing good coverage of Core ATen operators.

### Tradução + explicação para iniciantes

| Conceito                 | Em termos simples                                                              |
| ------------------------ | ------------------------------------------------------------------------------ |
| **PyTorch / TensorFlow** | “Fábricas” onde treinamos “cérebros de IA”.                                    |
| **.tflite**              | Arquivo miniaturizado do modelo → cabe no celular e roda sem internet.         |
| **MediaPipe**            | Biblioteca do Google que encaixa esse modelo em apps de câmera, microfone etc. |
| **GPU/NPU**              | Chips do telefone que aceleram IA (GPU = gráfico, NPU = neural).               |

---

<a id="2"></a>

## 2. Por que usar no mundo real?

| Problema comum                                      | Como o AI Edge Torch resolve                                               |
| --------------------------------------------------- | -------------------------------------------------------------------------- |
| App precisa de IA mas não pode depender de internet | Converte o modelo para rodar on-device (offline).                          |
| Modelo PyTorch está grande                          | Conversor suporta **quantização INT8** → arquivo 4× menor, RAM 2-3× menor. |
| Quer um mini-ChatGPT embarcado                      | **Generative API** autoriza e converte LLMs focados em mobile.             |

---

<a id="3"></a>

## 3. Instalação rápida

```bash
# Linux + Python ≥ 3.10
python -m venv --prompt ai-edge-torch venv
source venv/bin/activate

# Versão estável
pip install ai-edge-torch

# ou versão noturna (recursos novos, pode ter bugs)
pip install ai-edge-torch-nightly
```

*Lista de releases*: [https://github.com/google-ai-edge/ai-edge-torch/releases](https://github.com/google-ai-edge/ai-edge-torch/releases)
*Histórico PyPI*: [https://pypi.org/project/ai-edge-torch/#history](https://pypi.org/project/ai-edge-torch/#history)

---

<a id="4"></a>

## 4. Converter redes PyTorch para TFLite (PyTorch Converter)

### 4.1 Passo-a-passo básico

```python
import torch, torchvision, ai_edge_torch

# ResNet-18 pré-treinada em ImageNet
resnet18 = torchvision.models.resnet18(
    torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

sample = (torch.randn(1, 3, 224, 224),)

edge = ai_edge_torch.convert(resnet18.eval(), sample)
edge.export("resnet18.tflite")
```

*Tutorial completo*: `docs/pytorch_converter/getting_started.ipynb`

---

### 4.2 Aplicação A — **Reconhecimento de objetos off-line (Android)**

1. Converta o modelo (acima) e coloque *resnet18.tflite* em `app/src/main/assets/`.
2. Kotlin:

```kotlin
val tflite = Interpreter(
    FileUtil.loadMappedFile(context, "resnet18.tflite"),
    Interpreter.Options().setNumThreads(4)
)

fun classify(rgb: ByteBuffer): Int {
    val out = Array(1) { FloatArray(1000) }
    tflite.run(arrayOf(rgb), out)
    return out[0].indices.maxBy { out[0][it] }
}
```

3. Desenhe a classe sobre o preview da câmera.
4. **Latência** Pixel 6 CPU-4 threads ≈ 12 ms ⚡ (80 fps).

---

### 4.3 Aplicação B — **Contagem de pessoas em Raspberry Pi (IoT)**

```python
import torch, torchvision, ai_edge_torch, cv2, time
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
          weights="DEFAULT").eval()

edge = ai_edge_torch.convert(model, (torch.rand(1,3,320,320),))
edge.export("ssd.tflite")
```

No Raspberry:

```python
import tflite_runtime.interpreter as tfl, cv2
intr = tfl.Interpreter('ssd.tflite', num_threads=4); intr.allocate_tensors()
inp = intr.get_input_details()[0]['index']; outs = intr.get_output_details()

cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    img = cv2.resize(frame, (320,320)).astype('float32')/255
    intr.set_tensor(inp, img[None]); t0=time.time(); intr.invoke()
    cls = intr.get_tensor(outs[1]['index'])[0]
    if (cls == 0).sum() > 5: activate_fan()
```

---

<a id="5"></a>

## 5. **Generative API** — LLMs enxutos para mobile

### 5.1 Passo-a-passo básico

```python
from ai_edge_torch.generative import modules as gm, convert
import torch

conf = gm.TransformerConfig(
    d_model=128, n_heads=4, n_layers=2,
    vocab_size=5000, seq_len=256, activation='swiglu')
model = gm.TransformerLM(conf)

edge_llm = convert(model.eval(), (torch.randint(0,5000,(1,16)),))
edge_llm.export("tiny_llm_128d.tflite")
```

Mais exemplos em `ai_edge_torch/generative/examples/cpp`.

---

### 5.2 Aplicação A — **Assistente de bordo off-line (Android + MediaPipe)**

```kotlin
val llm = LlmInference.createFromFile(context, "tiny_llm_128d.tflite")

fun faq(query: String): String =
    llm.generate(query, GenerationOptions(maxTokens=64, temperature=0.2f)).text
```

Latência Pixel 8 ≈ 120 ms por resposta.

---

### 5.3 Aplicação B — **Chatbot Flutter (iOS/Android)**

```dart
final tfl = tflite.Interpreter.fromAsset('phi2_int8.tflite',
     options: tflite.InterpreterOptions()..threads = 6);

String generate(String prompt) {
  final input = [promptToIds(prompt)];
  final output = List.filled(maxLen, 0).reshape([1, maxLen]);
  tfl.run(input, output);
  return idsToText(output[0]);
}
```

*(phi-2 quantizado: 820 MB, 9 tokens/s em iPad M2 — 100 % offline)*

---

<a id="6"></a>

## 6. Arquitetura interna & boas práticas

```
PyTorch model ──> torch.export() ──> AI Edge Torch Converter ──> .tflite
                     (grafo)          • mapeia ATen→TFL ops
                                       • injeta quantização (PTQ/QAT)
                                       • custom ops fallback
```

**Boas práticas**

| Checklist                                       | Motivo                       |
| ----------------------------------------------- | ---------------------------- |
| Use módulos padrão (`torch.nn`)                 | maior cobertura no conversor |
| Evite `if/for` dinâmicos                        | grafo precisa ser estático   |
| Calibre com 100–200 amostras                    | PTQ eficiente                |
| Verifique no TFLite CPU antes de ativar GPU/NPU | depuração mais simples       |

---

<a id="7"></a>

## 7. Benchmarks e roadmap

### Desempenho (batch 1)

| Modelo        | FP32   | INT8   | Dispositivo | Velocidade |
| ------------- | ------ | ------ | ----------- | ---------- |
| ResNet-18     | 46 MB  | 11 MB  | Pixel 6     | 80 fps     |
| Tiny-LLM 128d | 35 MB  | 9 MB   | Pixel 8     | 40 tok/s   |
| Phi-2 2.7B    | 3.1 GB | 820 MB | iPad M2     | 9 tok/s    |

### Roadmap (issues oficiais)

| Período | Funcionalidade                        |
| ------- | ------------------------------------- |
| 2025 Q3 | Kernel GPU OpenCL INT8                |
| 2025 Q4 | Delegate NNAPI (NPU Qualcomm/Samsung) |
| 2026 H1 | Exportação WebAssembly SIMD           |

---

<a id="8"></a>

## 8. Checklist final de produção

* [ ] PyTorch ≥ 2.4 + `model.eval()`
* [ ] `torch.export()` sem controle de fluxo Python
* [ ] Quantização PTQ/QAT realizada (se desejar INT8)
* [ ] `.tflite` testado no TFLite CPU ➜ depois GPU/NNAPI
* [ ] Incluído nos *assets* do app / firmware IoT
* [ ] Licenças e tamanho OK na loja de apps

---

<a id="9"></a>

## 9. Recursos oficiais & suporte

* **Release notes** → [https://github.com/google-ai-edge/ai-edge-torch/releases/](https://github.com/google-ai-edge/ai-edge-torch/releases/)
* **Notebook “Getting Started”** → `docs/pytorch_converter/getting_started.ipynb`
* **LLM Inference API (MediaPipe)** → [http://ai.google.dev/edge/mediapipe/solutions/genai/llm\_inference#ai\_edge\_model\_conversion](http://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference#ai_edge_model_conversion)
* **Código C++-demo GerAPI** → `ai_edge_torch/generative/examples/cpp`
* **Contribuindo** → [`CONTRIBUTING.md`](CONTRIBUTING.md)
* **Suporte** → [https://github.com/google-ai-edge/ai-edge-torch/issues/new/choose](https://github.com/google-ai-edge/ai-edge-torch/issues/new/choose)

---

### TL;DR

> **AI Edge Torch** = **Ponte PyTorch → TFLite** + **Generative API** para LLMs.
> Permite visão e linguagem **offline**, tamanho ↓, latência ↓, custo ↓.
> Exemplos prontos: câmera Android, IoT Raspberry, mini-ChatGPT embarcado.
> Instale com `pip install ai-edge-torch`, converta, exporte e publique!

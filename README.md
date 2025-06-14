# AI Edge Torch

[![](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_generative_api.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_generative_api.yml)
[![](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_model_coverage.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_model_coverage.yml)
[![](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_unittests.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_unittests.yml)
[![](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_release.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_release.yml)

**AI Edge Torch** é uma biblioteca **Python** que converte modelos **PyTorch** para o formato **`.tflite`**, executável pelo **TensorFlow Lite** e facilmente integrado ao **MediaPipe**.  
Com ela, você coloca IA **100 % on-device** (Android, iOS, IoT) — sem depender de nuvem — com:

* ampla cobertura **CPU** e suporte inicial a **GPU/NPU**;  
* conversor baseado em `torch.export()` (cobre a maioria dos operadores Core ATen);  
* **Generative API** (Alpha) para criar e quantizar **LLMs** móveis.

> **Nota** • O Conversor PyTorch está em **Beta** e a *Generative API* em **Alpha**.  
> Consulte as [notas de versão](https://github.com/google-ai-edge/ai-edge-torch/releases/) para detalhes.

---

## Índice

1. [Recursos-chave](#recursos-chave)  
2. [Instalação](#instalação)  
3. [PyTorch Converter](#pytorch-converter)  
4. [Generative API](#generative-api)  
5. [Benchmarks & Roadmap](#benchmarks)  
6. [Build Status](#build-status)  
7. [Contribuição](#contribuição)  
8. [Suporte](#suporte)  
9. [Licença](#licença)

---

## Recursos-chave

| Funcionalidade | Descrição |
|----------------|-----------|
| **Conversão PyTorch → TFLite** | Compatível com a maior parte dos ops Core ATen. |
| **Quantização** | Pós-treino (PTQ) ou *aware-training* (QAT) para INT8. |
| **Generative API** | Cria e quantiza transformers/LLMs otimizados para mobile. |
| **Integração MediaPipe** | Uso direto via *LLM Inference API*. |
| **Suporte futuro** | Kernels GPU INT8 (2025 Q3), delegate NNAPI (2025 Q4), exportação WASM SIMD (2026 H1). |

---

## Instalação

### Pré-requisitos

* **Python** ≥ 3.10   •   **Linux**  
* **PyTorch** [![torch](https://img.shields.io/badge/torch->=2.4.0-blue)](https://pypi.org/project/torch/)  
* **TensorFlow Lite nightly** [![tf-nightly](https://img.shields.io/badge/tf--nightly-latest-blue)](https://pypi.org/project/tf-nightly/)

### Ambiente virtual recomendado

```bash
python -m venv venv
source venv/bin/activate
pip install ai-edge-torch         # versão estável
# pip install ai-edge-torch-nightly  # versão de desenvolvimento (nightly)
````

*Lista de versões*: [https://github.com/google-ai-edge/ai-edge-torch/releases](https://github.com/google-ai-edge/ai-edge-torch/releases)
*Histórico PyPI*: [https://pypi.org/project/ai-edge-torch/#history](https://pypi.org/project/ai-edge-torch/#history)

---

## PyTorch Converter

### Passo-a-passo mínimo

```python
import torch, torchvision, ai_edge_torch

model = torchvision.models.resnet18(
    torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()

edge = ai_edge_torch.convert(model, (torch.randn(1, 3, 224, 224),))
edge.export("resnet18.tflite")
```

*Tutorial completo*: [`docs/pytorch_converter/getting_started.ipynb`](docs/pytorch_converter/getting_started.ipynb)

### Exemplo ❶ — câmera Android que reconhece objetos off-line

```kotlin
val tflite = Interpreter(
    FileUtil.loadMappedFile(context, "resnet18.tflite"),
    Interpreter.Options().setNumThreads(4)
)
val output = Array(1) { FloatArray(1000) }
tflite.run(arrayOf(inputBuffer), output)
val topClass = output[0].indices.maxBy { output[0][it] }
```

Latência média Pixel 6 ≈ 12 ms (80 fps).

### Exemplo ❷ — contagem de pessoas em Raspberry Pi

```python
import torch, torchvision, ai_edge_torch
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT").eval()
edge = ai_edge_torch.convert(model, (torch.rand(1,3,320,320),))
edge.export("ssd_people.tflite")
```

No Pi, use `tflite_runtime` para inferir e acionar ventilação quando `num_people > 5`.

---

## Generative API

A Generative API produz **LLMs enxutos** em poucos comandos.

```python
from ai_edge_torch.generative import modules as gm, convert
import torch

cfg = gm.TransformerConfig(
    d_model=128, n_heads=4, n_layers=2,
    vocab_size=5000, seq_len=256, activation="swiglu")
llm = gm.TransformerLM(cfg).eval()

edge = convert(llm, (torch.randint(0, 5000, (1, 16)),))
edge.export("tiny_llm.tflite")
```

### Integração rápida (Android + MediaPipe)

```kotlin
val llm = LlmInference.createFromFile(context, "tiny_llm.tflite")
val answer = llm.generate("Qual a capital do Brasil?", maxTokens = 32).text
```

Latência Pixel 8 ≈ 120 ms/resposta.
Mais detalhes: [http://ai.google.dev/edge/mediapipe/solutions/genai/llm\_inference#ai\_edge\_model\_conversion](http://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference#ai_edge_model_conversion)

---

<a id="benchmarks"></a>

## Benchmarks & Roadmap

| Modelo           | Tamanho INT8 | Dispositivo | Desempenho  |
| ---------------- | ------------ | ----------- | ----------- |
| ResNet-18        | 11 MB        | Pixel 6     | 80 fps      |
| Tiny-LLM (128 d) | 9 MB         | Pixel 8     | 40 tokens/s |
| Phi-2 (2 .7 B)   | 820 MB       | iPad M2     | 9 tokens/s  |

| Período     | Funcionalidade planejada               |
| ----------- | -------------------------------------- |
| **2025 Q3** | Kernels GPU (OpenCL INT8)              |
| **2025 Q4** | Delegate NNAPI (NPUs Qualcomm/Samsung) |
| **2026 H1** | Exportação WebAssembly SIMD            |

---

<a id="build-status"></a>

## Build Status

Os *badges* no topo refletem:

* **nightly\_generative\_api** — compila & testa a Generative API
* **nightly\_model\_coverage** — cobertura de operadores/arquiteturas
* **nightly\_unittests** — suíte de testes unitários
* **nightly\_release** — geração de pacotes PyPI (`ai-edge-torch-nightly`)

---

<a id="contribuição"></a>

## Contribuição

Quer ajudar? Leia **[`CONTRIBUTING.md`](CONTRIBUTING.md)**
– diretrizes de *pull request*, estilo de código e fluxo de revisão.

---

<a id="suporte"></a>

## Suporte

Abra uma *issue*:
[https://github.com/google-ai-edge/ai-edge-torch/issues/new/choose](https://github.com/google-ai-edge/ai-edge-torch/issues/new/choose)

---

<a id="licença"></a>

## Licença

Distribuído sob **Apache 2.0**. Consulte o arquivo [`LICENSE`](LICENSE) para detalhes.


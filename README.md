# AI Edge Torch

AI Edge Torch é uma biblioteca Python que suporta a conversão de modelos PyTorch para o formato
.tflite, que pode então ser executado com o TensorFlow Lite e o MediaPipe.
Isso possibilita que aplicações para Android, iOS e IoT executem modelos
completamente no dispositivo. O AI Edge Torch oferece ampla cobertura para CPU, com suporte
inicial para GPU e NPU. O AI Edge Torch busca se integrar profundamente com o PyTorch,
baseando-se no `torch.export()` e fornecendo uma boa cobertura dos operadores
Core ATen.

Para começar a converter modelos PyTorch para TF Lite, veja detalhes adicionais na
seção [Conversor PyTorch](https://www.google.com/search?q=%23conversor-pytorch). Para o caso particular de
Modelos de Linguagem Grandes (LLMs) e modelos baseados em transformadores, a [API
Generativa](https://www.google.com/search?q=%23api-generativa) suporta a criação e quantização de modelos para permitir
um desempenho aprimorado no dispositivo.

Embora façam parte do mesmo pacote PyPi, o conversor PyTorch é uma versão Beta,
enquanto a API Generativa é uma versão Alfa. Por favor, veja as [notas de
lançamento](https://github.com/google-ai-edge/ai-edge-torch/releases/) para informações
adicionais.

## Conversor PyTorch

Aqui estão os passos necessários para converter um modelo PyTorch para um flatbuffer TFLite:

```python
import torch
import torchvision
import ai_edge_torch

# Usa a resnet18 com pesos pré-treinados.
resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
sample_inputs = (torch.randn(1, 3, 224, 224),)

# Converte e serializa o modelo PyTorch para um flatbuffer tflite. Note que
# estamos definindo o modelo para o modo de avaliação antes da conversão.
edge_model = ai_edge_torch.convert(resnet18.eval(), sample_inputs)
edge_model.export("resnet18.tflite")
```

O notebook Jupyter de [introdução](https://www.google.com/search?q=docs/pytorch_converter/getting_started.ipynb)
oferece um passo a passo inicial do processo de conversão e pode ser experimentado
com o Google Colab.

Detalhes técnicos adicionais do Conversor PyTorch estão [aqui](https://www.google.com/search?q=docs/pytorch_converter/README.md).

## API Generativa

A API Generativa do AI Edge Torch é uma biblioteca nativa do Torch para a criação
de modelos Transformer do PyTorch otimizados para dispositivos móveis, que podem ser convertidos para TFLite,
permitindo que os usuários implementem facilmente Modelos de Linguagem Grandes (LLMs) em dispositivos móveis.
Os usuários podem converter os modelos usando o Conversor PyTorch do AI Edge Torch
e executá-los através do runtime do TensorFlow Lite. Veja
[aqui](https://www.google.com/search?q=ai_edge_torch/generative/examples/cpp).

Desenvolvedores de aplicativos móveis também podem usar a API Generativa do Edge para integrar
LLMs do PyTorch diretamente com a API de Inferência de LLM do MediaPipe para uma fácil integração
em seu código de aplicação. Veja
[aqui](http://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference#ai_edge_model_conversion).

Documentação mais detalhada pode ser encontrada [aqui](https://www.google.com/search?q=ai_edge_torch/generative).

A API Generativa atualmente suporta apenas CPU, com suporte planejado para GPU e NPU.
Uma direção futura é colaborar com a comunidade PyTorch para
garantir que abstrações de transformadores frequentemente usadas possam ser diretamente suportadas
sem a necessidade de recriação.

## Status de Build

| Tipo de Build         | Status                                                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| API Generativa (Linux) | [](https://www.google.com/search?q=%5Bhttps://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_generative_api.yml%5D\(https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_generative_api.yml\)) |
| Cobertura de Modelos (Linux) | [](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_model_coverage.yml) |
| Testes Unitários (Linux) | [](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_unittests.yml)       |
| Lançamento Noturno     | [](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_release.yml)           |

## Instalação

### Requisitos e Dependências

  * Versões do Python: \>=3.10
  * Sistema operacional: Linux
  * PyTorch: [](https://pypi.org/project/torch/)
  * TensorFlow: [](https://pypi.org/project/tf-nightly/)

### Ambiente Virtual Python

Configure um ambiente virtual Python:

```bash
python -m venv --prompt ai-edge-torch venv
source venv/bin/activate
```

A última versão estável pode ser instalada com:

```bash
pip install ai-edge-torch
```

Alternativamente, a versão noturna (nightly) pode ser instalada com:

```bash
pip install ai-edge-torch-nightly
```

  * A lista de lançamentos versionados pode ser vista [aqui](https://github.com/google-ai-edge/ai-edge-torch/releases).
  * A lista completa de lançamentos do PyPi (incluindo builds noturnos) pode ser vista [aqui](https://pypi.org/project/ai-edge-torch/#history).

# Contribuição

Veja nossa [documentação de contribuição](https://www.google.com/search?q=CONTRIBUTING.md).

# Obtendo Ajuda

Por favor, [crie uma issue no GitHub](https://github.com/google-ai-edge/ai-edge-torch/issues/new/choose) com qualquer dúvida.

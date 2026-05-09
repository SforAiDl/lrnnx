<!---
Copyright 2025 SAiDL Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance
with the License. You may obtain a copy of the License in the LICENSE file.
-->

# lrnnx: A library for Linear RNNs
<p>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <a href="https://pypi.org/project/lrnnx/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lrnnx?color=yellow&logo=pypi&logoColor=white"></a>
    <a href="https://arxiv.org/abs/2602.08810"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2602.08810-b31b1b.svg?logo=arxiv&logoColor=white"></a>
</p>

A unified PyTorch library providing easy access to state-of-the-art Linear RNN architectures for sequence modeling.
The technical report of this system was accepted to [EACL Student Research Workshop 2026](https://aclanthology.org/2026.eacl-srw.60/).
We recommend reading the report before using / contributing to the library.

## Installation

### From PyPI
```bash
# standard installation
pip install lrnnx
# with optional causal-conv1d
pip install "lrnnx[causal-conv1d]"
# for development
pip install "lrnnx[dev]"
```

We recommend installing PyTorch first, matching your specific CUDA version. After that, install our library using `--no-build-isolation`.
```bash
pip install lrnnx --no-build-isolation
```

### From Source
We recommend installation with [`uv`](https://docs.astral.sh/uv/getting-started/installation/), though standard `pip` is also supported.

#### Using uv
```bash
git clone https://github.com/SforAiDl/lrnnx.git
cd lrnnx
# standard installation
uv sync
# with optional causal-conv1d
uv sync --extra causal-conv1d
# for development
uv sync --extra dev
```

#### Using pip
```bash
git clone https://github.com/SforAiDl/lrnnx.git
cd lrnnx
# standard installation
pip install -e . --no-build-isolation
# with optional causal-conv1d
pip install -e ".[causal-conv1d]" --no-build-isolation
# for development
pip install -e ".[dev]" --no-build-isolation
```

Note that since our library builds several custom CUDA kernels, it can take time for this installation to finish.
Along with `causal-conv1d` the full installation can take about 30 minutes, depending on the number of CPUs available.

## Model Zoo
Our library provides implementations of the following Linear RNN architectures:
- [S4](https://openreview.net/forum?id=uYLFoz1vlAC)
- [S4D](https://dl.acm.org/doi/10.5555/3600270.3602877)
- [S5](https://openreview.net/forum?id=Ai8Hw3AXqks)
- [Event-SSM](https://www.computer.org/csdl/proceedings-article/icons/2024/686500a124/22lEawhJ0Va) (inside `S5`, use by passing `integration_timesteps`)
- [LRU](https://dl.acm.org/doi/10.5555/3618408.3619518)
- [S6](https://openreview.net/forum?id=tEYskw1VY2) (we implemented other discretizations)
- [STREAM](https://arxiv.org/abs/2411.12603) (inside `S6`, use by passing `integration_timesteps`)
- [RG-LRU](https://arxiv.org/abs/2402.19427)
- [S7](https://arxiv.org/abs/2410.03464)
- [aTENNuate](https://www.isca-archive.org/interspeech_2025/pei25_interspeech.html)

We expose several levels of API for each model, including a scan, a recurrent step, and a full layer API matching the paper.
For S5 we implement both a convolution based approach and a parallel scan approach.
The latter is more stable and faster for most use cases, but the convolution based approach can be faster for very long sequences.

## Usage

### Training
It is easy to instantiate a model from our library
```python
from lrnnx.models.lti import LRU
from lrnnx.models.ltv import Mamba

model_lti = LRU(d_model, d_state).cuda()
x = torch.randn(
	batch_size, seq_len, d_model, dtype=torch.float32, device="cuda"
)
output = model_lti(x)

model_ltv = Mamba(d_model, d_state).cuda()
x = torch.randn(
	batch_size, seq_len, d_model, dtype=torch.float32, device="cuda"
)
output = model_ltv(x)
```

### Inference
Linear RNNs in torch require special handling during inference, following [Mamba](https://github.com/state-spaces/mamba), we also implement CUDA graphs based inference which reduces CPU overheads, this leads to > 10x speedup compared to using a simple for loop over the sequence length.
The main file is [generation.py](https://github.com/SforAiDl/lrnnx/blob/main/lrnnx/utils/generation.py) which provides a simple API for autoregressive generation with any of the models in our library.
You can see a simple way to use it in our [benchmarking script](https://github.com/SforAiDl/lrnnx/blob/main/benchmarks/benchmark_inference.py).

### Reproducing the Benchmarks from the paper
This script will run both training and inference benchmarks.
```bash
python -m benchmarks.run_all
```

### Architectures
We also implement some common architectures based on the models in our library, such as a U-Net (inspired from [aTENNuate](https://www.isca-archive.org/interspeech_2025/pei25_interspeech.html) ) and a hierarchical classifier (inspired from [Event-SSM](https://www.computer.org/csdl/proceedings-article/icons/2024/686500a124/22lEawhJ0Va)).
Additionally, there is a [Language Model](https://github.com/SforAiDl/lrnnx/blob/main/lrnnx/models/language_model.py) architecture inspired from [Mamba](https://github.com/state-spaces/mamba) and [RG-LRU](https://arxiv.org/abs/2402.19427) which can be used for language modeling tasks, with replaceable LRNN and attention layers.
This can be used as
```python
from lrnnx.models.language_model import LRNNLMHeadModel

model = LRNNLMHeadModel(
	d_model, d_state, num_layers, vocab_size, mixer_types=["s5", "s6", "attn"]
)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
logits = model(input_ids)
```

### Tutorial Overview

Based on the architectures, there are tutorials on how to use them for two very popular use cases:
1. [U-Net Seq2Seq for audio denoising Tutorial](https://github.com/SforAiDl/lrnnx/blob/main/tutorials/notebooks/01_UNet.ipynb)
2. [Hierarchical Classification Tutorial](https://github.com/SforAiDl/lrnnx/blob/main/tutorials/notebooks/02_hierarchical_classifier.ipynb)

## Contributing

Please check out our [Contributing Guide](https://github.com/SforAiDl/lrnnx/blob/main/CONTRIBUTING.rst) for details on how to contribute to this project.

## Citation

If you use lrnnx in your research, please cite:

```bibtex
@inproceedings{bania-etal-2026-lrnnx,
    title = "lrnnx: A library for Linear {RNN}s",
    author = "Bania, Karan  and
      Kalburgi, Soham  and
      Tanwar, Manit  and
      Dhruthi  and
      Nagarsekar, Aditya  and
      Mestha, Harshvardhan  and
      Chibber, Naman  and
      Deshmukh, Raj  and
      Sathyanarayanan, Anish  and
      Rathore, Aarush  and
      Chheda, Pratham",
    editor = "Baez Santamaria, Selene  and
      Somayajula, Sai Ashish  and
      Yamaguchi, Atsuki",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 4: Student Research Workshop)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-srw.60/",
    doi = "10.18653/v1/2026.eacl-srw.60",
    pages = "811--817",
    ISBN = "979-8-89176-383-8",
    abstract = "Linear recurrent neural networks (LRNNs) provide a structured approach to sequence modeling that bridges classical linear dynamical systems and modern deep learning, offering both expressive power and theoretical guarantees on stability and trainability. In recent years, multiple LRNN-based architectures have been proposed, each introducing distinct parameterizations, discretization schemes, and implementation constraints. However, existing implementations are fragmented across different software frameworks, often rely on framework-specific optimizations, and in some cases require custom CUDA kernels or lack publicly available code altogether. As a result, using, comparing, or extending LRNNs requires substantial implementation effort. To address this, we introduce $\texttt{lrnnx}$, a unified software library that implements several modern LRNN architectures under a common interface. The library exposes multiple levels of control, allowing users to work directly with core components or higher-level model abstractions. $\texttt{lrnnx}$ aims to improve accessibility, reproducibility, and extensibility of LRNN research and applications. We make our code available under a permissive MIT license."
}
```

## License

MIT

## Acknowledgments

This library builds upon the excellent work of researchers who developed the individual LRNN models.
Please see individual model documentation for proper citations of the original papers.

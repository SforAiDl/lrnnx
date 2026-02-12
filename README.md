<!---
Copyright 2025 SAiDL Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance
with the License. You may obtain a copy of the License in the LICENSE file.
-->

# lrnnx: A library for Linear RNNs
<p>
	<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
	<a href="https://arxiv.org/abs/2602.08810"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2602.08810-b31b1b.svg"></a>
</p>

A unified PyTorch library providing easy access to state-of-the-art Linear RNN architectures for sequence modeling.
The technical report of this system was accepted to [EACL Student Research Workshop 2026](https://2026.eacl.org/calls/srw/).
We recommend reading the report before using / contributing to the library.

## Installation

### From PyPI
```bash
# standard installation
pip install lrnnx
# with optional causal-conv1d
pip install "lrnnx[conv1d]"
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
uv sync --extra conv1d
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
pip install -e ".[conv1d]" --no-build-isolation
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
Linear RNNs in torch require special handling during inference, following [mamba](https://github.com/state-spaces/mamba), we also implement CUDA graphs based inference which reduces CPU overheads, this leads to > 10x speedup compared to using a simple for loop over the sequence length.
The main file is [generation.py](lrnnx/generation.py) which provides a simple API for autoregressive generation with any of the models in our library.
You can see a simple way to use it in our [benchmarking script](benchmarks/benchmark_inference.py).

### Reproducing the Benchmarks from the paper
This script will run both training and inference benchmarks.
```bash
python -m benchmarks.run_all
```

### Architectures
We also implement some common architectures based on the models in our library, such as a U-Net (inspired from [aTENNuate](https://www.isca-archive.org/interspeech_2025/pei25_interspeech.html) ) and a hierarchical classifier (inspired from [Event-SSM](https://www.computer.org/csdl/proceedings-article/icons/2024/686500a124/22lEawhJ0Va)).
Additionally, there is a [Language Model](lrnnx/models/language_model.py) architecture inspired from [Mamba](https://github.com/state-spaces/mamba) and [RG-LRU](https://arxiv.org/abs/2402.19427) which can be used for language modeling tasks, with replaceable LRNN and attention layers.
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

Based on the architectures, there are tutorials on how to use them for 2 very popular use cases:
1. [U-Net Seq2Seq for audio denoising Tutorial](tutorials/notebooks/01_UNet.ipynb)
2. [Hierarchical Classification Tutorial](tutorials/notebooks/02_hierarchical_classifier.ipynb)

## Contributing

Please check out our [Contributing Guide](CONTRIBUTING.rst) for details on how to contribute to this project.

## Citation

If you use lrnnx in your research, please cite:

```bibtex
@misc{bania2026textttlrnnxlibrarylinearrnns,
	title={$\texttt{lrnnx}$: A library for Linear RNNs}, 
	author={Karan Bania and Soham Kalburgi and Manit Tanwar and Dhruthi and Aditya Nagarsekar and Harshvardhan Mestha and Naman Chibber and Raj Deshmukh and Anish Sathyanarayanan and Aarush Rathore and Pratham Chheda},
	year={2026},
	eprint={2602.08810},
	archivePrefix={arXiv},
	primaryClass={cs.LG},
	url={https://arxiv.org/abs/2602.08810}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This library builds upon the excellent work of researchers who developed the individual LRNN models.
Please see individual model documentation for proper citations of the original papers.

# SlimFit: Memory-Efficient Fine-Tuning of Transformer-based Models Using Training Dynamics

## Paper
[ArXiv](https://arxiv.org/abs/2305.18513)


## Abstract
Transformer-based models, such as BERT and ViT, have achieved state-of-the-art results across different natural language processing (NLP) and computer vision (CV) tasks. However, these models are extremely memory intensive during their fine-tuning process, making them difficult to deploy on GPUs with limited memory resources. To address this issue, we introduce a new tool called SLIMFIT that reduces the memory requirements of these models by dynamically analyzing their training dynamics and freezing less-contributory layers during fine-tuning. The layers to freeze are chosen using a runtime inter-layer scheduling algorithm. SLIMFIT adopts quantization and pruning for particular layers to balance the load of dynamic activations and to minimize the memory footprint of static activations, where static activations refer to those that cannot be discarded regardless of freezing. This allows SLIMFIT to freeze up to 95% of layers and reduce the overall on-device GPU memory usage of transformerbased models such as ViT and BERT by an average of 2.2x, across different NLP and CV benchmarks/datasets such as GLUE, SQuAD 2.0, CIFAR-10, CIFAR-100 and ImageNet with an average degradation of 0.2% in accuracy. For such NLP and CV tasks, SLIMFIT can reduce up to 3.1x the total on-device memory usage with an accuracy degradation of only up to 0.4%. As a result, while fine-tuning of ViT
on ImageNet and BERT on SQuAD 2.0 with a batch size of 128 requires 3 and 2 32GB GPUs respectively, SLIMFIT enables their fine-tuning on a single 32GB GPU without any significant accuracy degradation.


## This repo contains the preliminary version of our code. To run the code, simply execute the following command by replacing "task.py" with your desired task:
CUDA_VISIBLE_DEVICES=0 python -W ignore task.py

## Example:
CUDA_VISIBLE_DEVICES=0 python -W ignore CIFAR10.py


## MIT License

Copyright (c) 2022 Arash Ardakani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

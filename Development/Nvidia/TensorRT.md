# Overview
The core of NVIDIA® TensorRT™ is a C++ library that facilitates high-performance inference on NVIDIA graphics processing units (GPUs). TensorRT takes a trained network, which consists of a network definition and a set of trained parameters, and produces a highly optimized runtime engine that performs inference for that network.
TensorRT provides APIs via C++ and Python that help to express deep learning models via the Network Definition API or load a pre-defined model via the parsers that allow TensorRT to optimize and run them on an NVIDIA GPU. TensorRT applies graph optimizations, layer fusions, among other optimizations, while also finding the fastest implementation of that model leveraging a diverse collection of highly optimized kernels. TensorRT also supplies a runtime that you can use to execute this network on all of NVIDIA’s GPU’s from the NVIDIA Pascal™ generation onwards.

TensorRT also includes optional high speed mixed precision capabilities with the NVIDIA Pascal, NVIDIA Volta™, NVIDIA Turing™, NVIDIA Ampere architecture, NVIDIA Ada Lovelace architecture, and NVIDIA Hopper™ Architectures.
# Runtime
In NVIDIA TensorRT, the term "runtime" refers to the TensorRT runtime library. It is a crucial component used for executing deep learning models during the inference stage. The TensorRT runtime library provides an efficient way to deploy and run deep learning models, particularly on NVIDIA GPUs.

The TensorRT runtime library includes optimizations and acceleration techniques for deep learning models to achieve low latency and high throughput in production environments. It leverages the parallel computing power of GPUs to accelerate model inference by reducing computation and memory overhead.

The TensorRT runtime library supports model import from various deep learning frameworks, including TensorFlow, PyTorch, ONNX, and others, enabling you to deploy models trained in these frameworks within TensorRT and harness the performance advantages of NVIDIA GPUs.

In summary, the TensorRT runtime library is a critical component for efficiently running deep learning inference on GPUs, accelerating model inference, and achieving better performance.
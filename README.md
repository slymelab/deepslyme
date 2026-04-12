<div align="center">
  <img src="https://raw.githubusercontent.com/slymelab/deepslyme/main/assets/images/logo.jpg" alt="DeepSlyme Logo" width="200" style="max-width: 50%;" />

  <p><em>A powerful LLM training framework built natively on Slyme.</em></p>

  <p>
    <a href="https://pypi.org/project/deepslyme/"><img src="https://img.shields.io/pypi/v/deepslyme.svg?label=PyPI" alt="PyPI version"></a>
    <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python version">
    <a href="https://slymelab.github.io/deepslyme/"><img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation"></a>
    <a href="https://github.com/slymelab/deepslyme/blob/main/LICENSE"><img src="https://img.shields.io/github/license/slymelab/deepslyme" alt="License"></a>
  </p>

  <p>
    <b>English</b> | 
    <a href="https://github.com/slymelab/deepslyme/blob/main/i18n/README_zh.md">简体中文</a>
  </p>
</div>

## About DeepSlyme

DeepSlyme is a powerful and highly customizable LLM training framework built natively on [Slyme](https://slymelab.github.io/slyme/). It is designed to help researchers and developers seamlessly orchestrate distributed training and accelerate model optimization and iteration.

The `Trainer` class in traditional deep learning frameworks is often a highly encapsulated "black box", filled with intricate object nesting internally. DeepSlyme abandons this paradigm, turning instead to the **node-based** and **functional context** features of the Slyme framework, presenting you with an extremely clear model training pipeline.

## Installation

DeepSlyme requires **Python 3.9+**. You can install it directly via pip:

```bash
pip install deepslyme
```

*(Note: If you are new to the Slyme series, it is highly recommended that you first read the [official Slyme documentation](https://slymelab.github.io/slyme/).)*

## Core Advantages

**Transparent Execution Flow:** Break free from the black box constraints of traditional Trainers. DeepSlyme eliminates complex object nesting and endless code jumping. Data loading, model forward passes, gradient backpropagation, and parameter updates—every step is clearly visible, giving you absolute control over the entire training loop.

**High Extensibility:** Focus your energy on the algorithm itself rather than being constrained by tedious boilerplate code. DeepSlyme adopts a completely decoupled architecture, making the process of testing new ideas significantly faster and smoother.

**Universal Composability:** Thanks to the unified ecosystem of the underlying Slyme framework, you can seamlessly extract and reuse custom Nodes across different projects. Meanwhile, you can easily integrate and call out-of-the-box modules from the entire ecosystem, merging your breakthrough research into the broader community.

## Documentation

To dive deeper into DeepSlyme's architecture and usage, please refer to our official documentation site.

**👉 [Read the Official DeepSlyme Documentation](https://slymelab.github.io/deepslyme/)**

## License

This project is licensed under the Apache-2.0 License.

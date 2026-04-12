# What is DeepSlyme?

DeepSlyme is a powerful and highly customizable LLM training framework built natively on [Slyme](https://slymelab.github.io/slyme/). It is designed to help researchers and developers seamlessly orchestrate distributed training and accelerate model optimization and iteration.

The `Trainer` class in traditional deep learning frameworks is often a highly encapsulated "black box", filled with intricate object nesting internally. DeepSlyme abandons this paradigm, turning instead to the **node-based** and **functional context** features of the Slyme framework, presenting you with an extremely clear model training pipeline.

::: tip
If you are new to the Slyme series, it is highly recommended that you first read the [official Slyme documentation](https://slymelab.github.io/slyme/), which serves as the foundation for the entire DeepSlyme project.
:::

::: tip
We are continuously optimizing and improving the DeepSlyme ecosystem, and we welcome community contributions and feedback. If you have any suggestions or questions, feel free to submit a Pull Request or open an Issue on GitHub.

The documentation for the DeepSlyme project is constantly being improved, so stay tuned! You can also check out the [project source code](https://github.com/slymelab/deepslyme) first.
:::

## Core Advantages

- **Transparent Execution Flow**: Break free from the black box constraints of traditional Trainers. DeepSlyme eliminates complex object nesting and endless code jumping. Data loading, model forward passes, gradient backpropagation, and parameter updates—every step is clearly visible, giving you absolute control over the entire training loop.

- **High Extensibility**: Focus your energy on the algorithm itself rather than being constrained by tedious boilerplate code. DeepSlyme adopts a completely decoupled architecture, making the process of testing new ideas significantly faster and smoother.

- **Universal Composability**: Thanks to the unified ecosystem of the underlying Slyme framework, you can seamlessly extract and reuse custom [Node](https://slymelab.github.io/slyme/guide/essentials/node) across different projects. Meanwhile, you can easily integrate and call out-of-the-box modules from the entire ecosystem, merging your breakthrough research into the broader community.

## Quick Installation

Get the latest version of DeepSlyme via `pip`:

```bash
pip install deepslyme
```

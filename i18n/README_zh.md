<div align="center">
  <img src="https://raw.githubusercontent.com/slymelab/deepslyme/main/assets/images/logo.jpg" alt="DeepSlyme Logo" width="200" style="max-width: 50%;" />

  <p><em>一款基于 Slyme 原生构建的强大 LLM 训练框架。</em></p>

  <p>
    <a href="https://pypi.org/project/deepslyme/"><img src="https://img.shields.io/pypi/v/deepslyme.svg?label=PyPI" alt="PyPI version"></a>
    <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python version" />
    <a href="https://slymelab.github.io/deepslyme/zh/"><img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation"></a>
    <a href="https://github.com/slymelab/deepslyme/blob/main/LICENSE"><img src="https://img.shields.io/github/license/slymelab/deepslyme" alt="License"></a>
  </p>

  <p>
    <a href="https://github.com/slymelab/deepslyme/blob/main/README.md">English</a> | 
    <b>简体中文</b>
  </p>
</div>

## 关于 DeepSlyme

DeepSlyme 是一款基于 [Slyme](https://slymelab.github.io/slyme/zh/) 原生构建的强大且高度可定制的 LLM 训练框架。它旨在帮助研究人员和开发者无缝编排分布式训练，并加速模型的优化与迭代。

传统深度学习框架中的 `Trainer` 类往往是一个高度封装的“黑盒”，内部充斥着错综复杂的对象嵌套。DeepSlyme 摒弃了这一模式，转而利用 Slyme 框架的**节点化**与**函数式上下文**特性，为你展现一个极致清晰的模型训练流程。

## 安装

DeepSlyme 需要 **Python 3.9+**。您可以通过 pip 直接安装：

```bash
pip install deepslyme
```

*（注意：如果您初次接触 Slyme 系列框架，强烈推荐您首先阅读 [Slyme 官方文档](https://slymelab.github.io/slyme/zh/)。）*

## 核心优势

**透明的执行流程：** 摆脱传统 Trainer 的黑盒束缚。DeepSlyme 消除了错综复杂的对象嵌套和无休止的代码跳转。数据加载、模型前向、梯度回传、参数更新，每一个步骤都清晰可见，让你对整个训练循环拥有绝对的掌控力。

**高可扩展性：** 将精力集中于算法本身，而非被繁琐的样板代码所限制。DeepSlyme 采用彻底解耦的架构，测试新想法的过程会显著提速且丝滑无阻。

**通用的可组合性：** 得益于底层 Slyme 的统一生态，你可以在不同项目间无缝提取与复用自定义节点。同时，也能轻松整合并调用整个生态系统中开箱即用的模块，将你的突破性研究融入更广泛的社区。

## 文档

要深入了解 DeepSlyme 的架构与使用，请参阅我们的官方文档站点。

**👉 [阅读 DeepSlyme 官方文档](https://slymelab.github.io/deepslyme/zh/)**

## 许可证

本项目采用 Apache-2.0 许可证。

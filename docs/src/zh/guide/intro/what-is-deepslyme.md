# DeepSlyme 是什么？

DeepSlyme 是一款基于 [Slyme](https://slymelab.github.io/slyme/zh/) 原生构建的强大且高度可定制的 LLM 训练框架。它旨在帮助研究人员和开发者无缝编排分布式训练，并加速模型的优化与迭代。

传统深度学习框架中的 `Trainer` 类往往是一个高度封装的“黑盒”，内部充斥着错综复杂的对象嵌套。DeepSlyme 摒弃了这一模式，转而利用 Slyme 框架的**节点化**与**函数式上下文**特性，为你展现一个极致清晰的模型训练流程。

::: tip
如果你初次接触 Slyme 系列框架，强烈推荐你首先阅读 [Slyme 官方文档](https://slymelab.github.io/slyme/zh/)，它是整个 DeepSlyme 项目的基础。
:::

::: tip
我们正在持续优化和完善 DeepSlyme 生态，欢迎社区贡献和反馈。如果你有任何建议或问题，请随时提交 Pull Request 或在 GitHub 上开 Issue。

DeepSlyme 项目的文档正在持续完善中，敬请关注！可优先查看[项目源码](https://github.com/slymelab/deepslyme)。
:::

## 核心优势

- **透明的执行流程**：摆脱传统 Trainer 的黑盒束缚。DeepSlyme 消除了错综复杂的对象嵌套和无休止的代码跳转。数据加载、模型前向、梯度回传、参数更新，每一个步骤都清晰可见，让你对整个训练循环拥有绝对的掌控力。

- **高可扩展性**：将精力集中于算法本身，而非被繁琐的样板代码所限制。DeepSlyme 采用彻底解耦的架构，测试新想法的过程会显著提速且丝滑无阻。

- **通用的可组合性**：得益于底层 Slyme 的统一生态，你可以在不同项目间无缝提取与复用自定义 [Node](https://slymelab.github.io/slyme/zh/guide/essentials/node)。同时，也能轻松整合并调用整个生态系统中开箱即用的模块，将你的突破性研究融入更广泛的社区。

## 快速安装

通过 `pip` 获取最新版本的 DeepSlyme：

```bash
pip install deepslyme
```

# TrickCatcher shared baseline engine

本目录提供数据发现、模型调用、输入生成、变体筛选、统一执行、参考输出和论文指标分析等共享实现。论文实验入口是根目录 `run-paired.ps1`，不要在本目录单独启动另一套实验。

指标按候选证据逐条分类；Precision 为有 TP/FP 的 PUT 上的宏平均；Recall 的分母是全部 846 个缺陷 PUT；参考执行失败或不一致记为 `unresolved_reference`。K=6 与 K=10 的原始数据来自两次独立运行。

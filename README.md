# RACE-Vote 实验流程

本项目用于执行四项实验：RQ1 检测效果、RQ2 组件消融、RQ3 变体质量和 RQ4 Buggy 变体价值。实验集为 TrickyBugs 中的部分 Python PUT；

## 1. 配置

复制 `.env.example` 为 `.env`，填写 API Key 和 Base URL。论文模型为 `deepseek-v4-flash`。运行时使用温度 0.8、top-p 0.95、最多 1024 输出 token、随机种子 42、PUT 超时 5 秒、生成器超时 30 秒。

## 2. RQ1：TrickCatcher 与 RACE-Vote

K=6 和 K=10 是两次独立生成实验，不能从 K=10 的池中抽取组合代替 K=6。建议按 25 个 PUT 分批执行：

```powershell
.\run-paired.ps1 --k 6 --preflight
.\run-paired.ps1 --k 6 --start 0 --limit 25
.\run-paired.ps1 --k 10 --start 0 --limit 25
```

继续下一批时把 `--start` 改为 25、50、75……；已完成 PUT 会自动跳过。结果分别写入：

- `Baseline/K6/Ablation-TrickCatcher` 与 `Baseline/K6/RACE-Vote`
- `Baseline/K10/Ablation-TrickCatcher` 与 `Baseline/K10/RACE-Vote`

## 3. 人工核验与分析

四个独立运行各自生成一份候选输入核验表：

```powershell
.\Ablation-TrickCatcher\analyze.ps1 prepare-review --k 6
.\RACE-Vote\analyze.ps1 prepare-review --k 6
.\Ablation-TrickCatcher\analyze.ps1 prepare-review --k 10
.\RACE-Vote\analyze.ps1 prepare-review --k 10
```

在对应目录的 `manual/input_validity.csv` 中填写 `input_valid`（合法填 1，非法填 0），然后执行：

```powershell
.\Ablation-TrickCatcher\analyze.ps1 analyze --k 6
.\RACE-Vote\analyze.ps1 analyze --k 6
.\Ablation-TrickCatcher\analyze.ps1 analyze --k 10
.\RACE-Vote\analyze.ps1 analyze --k 10
```

## 4. RQ2：K=10 消融

消融按同一批次范围运行。w/o RC-Weight 和 w/o LLM-ReVote 复用完整方法的变体、输入和执行记录；w/o CE-CoEvo 复用第一阶段并重新生成无反例反馈的第二阶段。

```powershell
.\Ablation-No-RC\run.ps1 --k 10 --start 0 --limit 25
.\Ablation-No-CE\run.ps1 --k 10 --start 0 --limit 25
.\Ablation-No-RV\run.ps1 --k 10 --start 0 --limit 25

.\Ablation-No-RC\analyze.ps1 prepare-review --k 10
.\Ablation-No-CE\analyze.ps1 prepare-review --k 10
.\Ablation-No-RV\analyze.ps1 prepare-review --k 10
```

## 5. 汇总 RQ1–RQ4

```powershell
python .\RACEVoteCore\compare_experiments.py
```

`Baseline/RACEVote-Comparison` 生成论文对应的结果 ：`rq1_detection_effectiveness`、`rq2_component_ablation`、`rq3_variant_quality` 和 `rq4_useful_buggy_variants`。

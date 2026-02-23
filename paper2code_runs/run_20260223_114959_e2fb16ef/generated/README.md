# ERL 最小可执行复现（基于论文 2602.13949v1）

本目录给出一个**可运行、可测试、可验证**的 ERL（Experiential Reinforcement Learning）最小复现实现，核心覆盖：

1. 首次尝试（first attempt）
2. 反思生成（reflection）
3. 二次尝试（second attempt）
4. 策略更新（policy update）
5. 内化蒸馏（distillation/internalization）
6. 跨回合反思记忆（memory）

> 说明：为保证在当前项目环境中可快速验证，本实现采用玩具稀疏奖励环境，并使用解耦模块复现算法机制，而非复刻原论文的大规模分布式训练系统。

## 目录

```text
generated/
├── requirements.txt
├── run_demo.py
├── src/erl/
│   ├── __init__.py
│   ├── config.py
│   ├── environment.py
│   ├── evaluation.py
│   ├── memory.py
│   ├── policy.py
│   └── trainers.py
└── tests/
    ├── test_erl_flow.py
    └── test_erl_vs_rlvr.py
```

## 快速运行

```bash
python3 run_demo.py
python3 -m unittest discover -s tests -p "test_*.py"
```

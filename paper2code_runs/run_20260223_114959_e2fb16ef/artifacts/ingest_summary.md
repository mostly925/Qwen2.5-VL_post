# 论文摄取记录

- 论文地址: https://arxiv.org/abs/2602.13949v1
- 论文标题: Experiential Reinforcement Learning
- 摄取时间: 2026-02-23

## 已保存原始文件

1. `artifacts/paper_abs.html`：arXiv 摘要页原始 HTML
2. `artifacts/paper_full.html`：arXiv HTML 全文原始内容
3. `artifacts/paper.pdf`：论文 PDF 原文

## 关键信息（用于后续分割与实现）

- 核心方法为 ERL（Experience → Reflection → Consolidation）循环。
- 关键算法见 Algorithm 1 / Appendix Algorithm 2。
- 关键目标函数包含 `L_policy` 和 `L_distill`。
- 核心模块包括：第一轮尝试、反思生成、第二轮尝试、记忆更新、策略更新、内化蒸馏。

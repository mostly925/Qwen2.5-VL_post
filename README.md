# Qwen_post

一个面向 **Qwen2.5-VL-7B** 的多模态训练与推理工程，支持从监督微调到偏好优化、再到奖励驱动优化的完整流程。

## 项目定位

本项目围绕 Qwen2.5-VL 的训练稳定性、推理可用性与工程落地展开，核心覆盖：

- **SFT**（监督微调）训练流程
- **DPO**（直接偏好优化）训练流程
- **GRPO**（组相对策略优化）训练流程
- LoRA 权重提取、合并与最终可推理模型导出
- vLLM/OpenAI 兼容 API 推理验证

## 核心能力

- DeepSpeed 分布式训练（ZeRO-2/ZeRO-3）
- LoRA 高效微调（PEFT）
- BF16 混合精度训练与推理
- 多模态数据（图像 + 文本）处理
- 训练过程的 checkpoint、评估与日志能力

## 目录结构

```text
.
├── run_qwen_sft.py          # SFT 训练入口
├── run_qwen_dpo.py          # DPO 训练入口
├── run_qwen_grpo.py         # GRPO 训练入口
├── convert_lora_final.py    # LoRA 提取与合并导出
├── inference.py             # 最终模型推理示例
├── test_vllm_api.py         # vLLM API 兼容测试
├── trainer/                 # 训练框架核心实现
├── configs/                 # 模型/训练配置
└── data/                    # 示例数据与样例图片
```

## 已沉淀的问题修复经验

项目内文档总结了若干关键问题与解决路径：

- `DPO死锁问题修复.md`：小数据集下 DPO 多卡训练末尾卡死排查与修复
- `GRPO训练Inference Tensor问题修复总结.md`：ZeRO-3 + LoRA 组合下 inference tensor 错误修复
- `训练模型导出与推理问题修复.md`：LoRA 导出、格式转换与推理稳定性问题修复

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 选择训练模式

```bash
python run_qwen_sft.py
python run_qwen_dpo.py
python run_qwen_grpo.py
```

### 3) 合并 LoRA 并导出最终模型

```bash
python convert_lora_final.py
```

### 4) 推理验证

```bash
python inference.py
python test_vllm_api.py
```

## 说明

- 仓库默认不提交本地基础模型目录 `Qwen2.5-VL-7B-Instruct/`，请按需从官方来源下载。
- 训练脚本中的路径以本地环境为例，实际使用时请改为你的真实路径。

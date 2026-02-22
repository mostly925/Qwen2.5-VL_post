# GRPO训练Inference Tensor问题修复总结

## 问题根源

在使用 **DeepSpeed ZeRO-3 + LoRA + Gradient Checkpointing** 组合时，会出现 inference tensor 错误：

```
RuntimeError: Inference tensors cannot be saved for backward. To work around you can make a clone to get a normal tensor and use it in autograd.
```

### 技术原因

1. **ZeRO-3** 将模型参数分区到不同GPU，按需获取参数时会创建 **inference tensor**
2. **Inference tensor** 是只读的，不能保存梯度用于反向传播
3. **LoRA** 需要通过这些参数计算梯度，导致冲突
4. **Gradient Checkpointing** 在重新计算前向传播时会加剧这个问题

## 最终解决方案：切换到 ZeRO-2

### 核心修改

#### 1. 修改 DeepSpeed 配置 (run_qwen_grpo.py)

**之前 (ZeRO-3):**

```
ds_config = DsConfig(

    zero_config=DsZero3Config(

        offload_optimizer=DsOffloadConfig(device='cpu'),

        offload_param=DsOffloadConfig(device='cpu'),  # 这导致 inference tensor!

    ),

    gradient_clipping=1.0,

)
```

**现在 (ZeRO-2):**

```
ds_config = DsConfig(

    zero_config=DsZero2Config(

        offload_optimizer=DsOffloadConfig(device='cpu'),

        # ZeRO-2 不支持 offload_param，只分区优化器状态

    ),

    gradient_clipping=1.0,

    # 禁用 activation_checkpointing

)
```

#### 2. 添加必要的 import

```
from trainer.train_configs import (

    ...

    DsZero2Config,  # 新增

    DsZero3Config,  # 保留以备后用

    ...

)
```

#### 3. 显式禁用 Gradient Checkpointing

```
# 显式禁用梯度检查点（与 ZeRO + LoRA 不兼容）

if hasattr(trainer.train_model, 'gradient_checkpointing_disable'):

    trainer.train_model.gradient_checkpointing_disable()

    

if hasattr(trainer.train_model, 'module'):

    if hasattr(trainer.train_model.module, 'gradient_checkpointing_disable'):

        trainer.train_model.module.gradient_checkpointing_disable()
```

### 保留的优化

以下修改被保留，因为它们是通用优化且无副作用：

1. **环境变量配置**

   ```
   os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
   
   os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
   
   os.environ["PARALLEL_TYPE"] = "ds"
   ```

2. **显式禁用 Gradient Checkpointing** - 防止意外启用

### 移除的修改

1. **RMSNorm Monkey Patch** - ZeRO-2 不会产生 RMSNorm 的 inference tensor 问题，不再需要

## ZeRO-2 vs ZeRO-3 对比

| 特性            | ZeRO-2              | ZeRO-3                  |
| --------------- | ------------------- | ----------------------- |
| **优化器状态**  | ✅ 分区              | ✅ 分区                  |
| **梯度**        | ✅ 分区              | ✅ 分区                  |
| **模型参数**    | ❌ 完整副本          | ✅ 分区                  |
| **LoRA 兼容性** | ✅ 完全兼容          | ❌ Inference tensor 问题 |
| **显存节省**    | ~60%                | ~80%                    |
| **适用场景**    | 中等规模模型 + LoRA | 大规模模型，全量微调    |

## 为什么 ZeRO-2 可以工作

1. **模型参数完整保留** - 每个GPU都有完整的模型参数副本
2. **参数是正常tensor** - 不是 inference tensor，可以正常计算梯度
3. **LoRA可以正常工作** - LoRA层可以通过参数计算梯度
4. **仍有显存优化** - 通过分区优化器状态和梯度，节省约60%显存

## 训练配置总结

当前配置适用于 **2x 4090 GPU** 训练 **Qwen2.5-VL-7B**：

```
train_config = TrainConfig(

    n_epochs=3,

    batch_size=1,              # 每GPU一个prompt

    model_name_or_path=model_path,

    max_seq_len=2048,

    

    # LoRA 配置

    use_lora=True,

    lora_rank=8,

    lora_alpha=32,

    lora_dropout=0.05,

    

    # ZeRO-2 配置

    ds_config=DsConfig(

        zero_config=DsZero2Config(

            offload_optimizer=DsOffloadConfig(device='cpu'),

        ),

        gradient_clipping=1.0,

    ),

    

    # GRPO 配置

    grpo_config=GRPOConfig(

        group_size=2,             # 每个prompt生成2个回答

        gen_max_new_tokens=256,

        ...

    ),

)
```

## 预期效果

✅ 不再出现 "Inference tensors cannot be saved for backward" 错误 ✅ GRPO 训练可以正常运行 ✅ 显存使用在2x4090可接受范围内 ✅ LoRA 梯度计算正常

## 如果仍然 OOM

如果切换到 ZeRO-2 后出现显存不足，可以调整：

1. **减少 group_size**: `2` → `1`
2. **减少 max_seq_len**: `2048` → `1536` 或 `1024`
3. **减少 gen_max_new_tokens**: `256` → `128`
4. **减少 LoRA rank**: `8` → `4`

## grpo_trainer.py 检查结果

✅ 



grpo_trainer.py

 

代码结构正常，无需修改 ✅ 已正确使用

 

```
torch.no_grad()
```

 

进行生成 ✅ 已正确处理 eval/train 模式切换 ✅ Dummy pass 已包含

 

```
use_cache=False
```
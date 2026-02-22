import os
import re
from typing import List, Optional
import torch

# ==================== 核心修复：Monkey Patch (以防万一) ====================
# 必须在加载任何 HuggingFace 模型之前应用此补丁
from transformers.models.qwen2 import modeling_qwen2

def fixed_qwen2_rmsnorm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    
    # [修正]: 必须 clone 权重本身，这才是解决 inference tensor 的关键
    return self.weight.clone() * hidden_states.to(input_dtype)

# 应用补丁
modeling_qwen2.Qwen2RMSNorm.forward = fixed_qwen2_rmsnorm_forward
print("✅ 已应用 Qwen2RMSNorm Monkey Patch (Weight Clone Version)")
# ========================================================================

# 环境变量配置
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# 限制只保留最近的 1 个 Checkpoint
os.environ['CKPT_MAX_TO_KEEP'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PARALLEL_TYPE"] = "ds"

from trainer.grpo_trainer import GRPOTrainer
from trainer.train_configs import (
    TrainConfig,
    OptimConfig,
    DsConfig,
    DsZero2Config,  # 使用 ZeRO-2
    DsZero3Config,
    DsOffloadConfig,
    DataLoaderConfig,
    GRPOConfig,
    DsActivationCheckpointingConfig
)
from trainer.tools import FileDataset, TrainerTools


# ================= 数据集定义 =================
class JsonlFileDataset(FileDataset):
    """Jsonl格式的文件数据集"""
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx) -> str:
        return self.file_paths[idx]


# ================= 奖励函数定义 (保持不变) =================
def extract_answer(text: str) -> str:
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def get_last_number(text: str) -> float:
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            pass
    return None

def compute_accuracy_reward(generated: str, target: str) -> float:
    gen_answer = extract_answer(generated)
    target_answer = extract_answer(target)
    if gen_answer.lower() == target_answer.lower():
        return 1.0
    if target_answer.lower() in gen_answer.lower():
        return 0.7
    gen_num = get_last_number(gen_answer)
    target_num = get_last_number(target_answer)
    if gen_num is not None and target_num is not None:
        diff = abs(gen_num - target_num)
        if diff == 0:
            return 1.0
        elif diff < 0.1 * abs(target_num):
            return 0.8
        elif diff < 0.5 * abs(target_num):
            return 0.5
        else:
            return 0.2
    return 0.0

def compute_format_reward(text: str) -> float:
    reward = 0.0
    if '<think>' in text.lower() and '</think>' in text.lower():
        reward += 0.3
    if '<answer>' in text.lower() and '</answer>' in text.lower():
        reward += 0.3
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_len = len(think_match.group(1).strip())
        if think_len > 50:
            reward += 0.2
        elif think_len > 20:
            reward += 0.1
    return reward

def reward_func(prompts: List[torch.Tensor], completion_ids: torch.Tensor, answers: List[str]) -> List[float]:
    tools = TrainerTools()
    tokenizer = tools.tokenizer
    rewards = []
    for i in range(len(completion_ids)):
        generated_text = tokenizer.decode(completion_ids[i], skip_special_tokens=True)
        target_text = answers[i] if isinstance(answers[i], str) else str(answers[i])
        accuracy_reward = compute_accuracy_reward(generated_text, target_text)
        format_reward = compute_format_reward(generated_text)
        total_reward = 0.7 * accuracy_reward + 0.3 * format_reward
        rewards.append(total_reward)
    return rewards


if __name__ == '__main__':
    # 你的模型路径
    model_path = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"
    os.environ['TOKEN_DIR'] = model_path
    data_path = '/root/autodl-tmp/data/grpo_dataset.jsonl'
    
    file_dataset = JsonlFileDataset([data_path])
    
    # ================= 关键配置修改 =================
    
    # 1. DeepSpeed 配置: 使用 ZeRO-2，彻底关闭 Activation Checkpointing
    ds_config = DsConfig(
        zero_config=DsZero2Config(
            offload_optimizer=DsOffloadConfig(device='cpu'), 
        ),
        gradient_clipping=1.0,
        # [重点] 显式设为 None，彻底关闭梯度检查点
        # A800 80GB 显存极大，跑 7B 模型不需要开这个，开了反而报错还变慢
        activation_checkpointing=None 
    )

    # 2. GRPO 配置
    grpo_config = GRPOConfig(
        grpo_steps=1,
        group_size=4,  # [优化] A800显存大，可以把 group_size 设为 4 或 8，采样更多样本，训练更稳定
        mixup_alpha=0.9,
        loss_beta=0.01, # 建议给一点点 KL 惩罚，防止模型输出崩坏
        loss_clip_eps=3e-4,
        loss_clip_eps_high=4e-4,
        loss_delta=None,
        loss_importance_sampling_level='seq',
        loss_type='grpo',
        gen_max_new_tokens=256,
        gen_temperature=0.7,
        gen_k=None,
        gen_p=0.95,
        gen_suppress_tokens=None
    )

    # 3. 训练配置
    train_config = TrainConfig(
        n_epochs=3,
        batch_size=1,  
        model_name_or_path=model_path,
        file_dataset=file_dataset,
        max_seq_len=2048,
        eval_batch_interval=100, 

        # LoRA 配置
        use_lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

        ds_config=ds_config,
        grpo_config=grpo_config,
        optim_config=OptimConfig(initial_lr=5e-7),
        gradient_accumulation_steps=4, # GRPO 中这不起作用，由 grpo_steps 控制

        data_loader_config=DataLoaderConfig(
            data_loader_num_workers=0, # n进程处理数据，数据量 < 1000 条设为 0 意味着主进程直接加载数据，不创建子进程，防止死锁，数据量很大（几十万/百万）通常设置为 min(4 * GPU数量, CPU核心数)
            data_loader_shuffle=True,
            data_loader_drop_last=False,
            data_loader_pin_memory=True
        )
    )

    trainer = GRPOTrainer(
        train_config=train_config,
        reward_func=reward_func,
        eval_prompts=[]
    )

    # [双重保险] 显式禁用梯度检查点
    if hasattr(trainer.train_model, 'gradient_checkpointing_disable'):
        trainer.train_model.gradient_checkpointing_disable()
        print("✅ 已显式禁用 Trainer 模型的梯度检查点")
    
    # 针对 DeepSpeed 包装后的模型
    if hasattr(trainer.train_model, 'module') and hasattr(trainer.train_model.module, 'gradient_checkpointing_disable'):
        trainer.train_model.module.gradient_checkpointing_disable()
        print("✅ 已显式禁用 DeepSpeed Module 的梯度检查点")
        
    # 针对 Base Model (Qwen)
    if hasattr(trainer.train_model, 'model') and hasattr(trainer.train_model.model, 'gradient_checkpointing_disable'):
        trainer.train_model.model.gradient_checkpointing_disable()
        print("✅ 已显式禁用 Base Model 的梯度检查点")

    print(f"🚀 开始在 A800 上进行高速 GRPO 训练 (ZeRO-2, No-GC)...")
    trainer.train()
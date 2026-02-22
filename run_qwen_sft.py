import os
from trainer.sft_trainer import SFTTrainer
from trainer.train_configs import (
    TrainConfig, 
    OptimConfig, 
    DsConfig, 
    DsZero3Config,
    DsOffloadConfig,
    DataLoaderConfig
)
from trainer.tools import FileDataset
os.environ["PARALLEL_TYPE"] = "ds"
# ================= 数据集定义 =================
class JsonlFileDataset(FileDataset):
    """Jsonl格式的文件数据集"""
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx) -> str:
        return self.file_paths[idx]

if __name__ == '__main__':
    # 你的模型路径 (可以是本地路径或 HF ID)
    model_path = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct" 
    os.environ['TOKEN_DIR'] = model_path
    data_path = '/root/autodl-tmp/data/sft_dataset.jsonl'
    
    # 数据集配置
    file_dataset = JsonlFileDataset([data_path])
    
    # DeepSpeed 配置 (ZeRO-3 以支持单卡/多卡微调大模型)
    ds_config = DsConfig(
        zero_config=DsZero3Config(
            offload_optimizer=DsOffloadConfig(device='cpu'),
            offload_param=DsOffloadConfig(device='cpu'),
        ),
        gradient_clipping=1.0
    )

    # 训练配置
    train_config = TrainConfig(
        n_epochs=3,
        batch_size=1, # 根据显存调整
        model_name_or_path=model_path, # [关键] 传入路径
        file_dataset=file_dataset,
        max_seq_len=2048,

        # LoRA 配置 (如果显存不够跑全量，开启这个)
        use_lora=True, # 设为 True 开启 LoRA
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],# 目标模块

        ds_config=ds_config,
        optim_config=OptimConfig(initial_lr=1e-5),
        gradient_accumulation_steps=8,
        
        # 注入自定义的 collate_fn
        data_loader_config=DataLoaderConfig(
            data_loader_num_workers=4,
            data_loader_shuffle=True,
            data_loader_drop_last=False,
            data_loader_pin_memory=True
        )
    )

    # Trainer
    trainer = SFTTrainer(
        train_config=train_config,
        eval_prompts=[]  # 关闭评估提示词，只关注 Loss 曲线，减少显存消耗
    )
    
    print("开始训练...")
    trainer.train()
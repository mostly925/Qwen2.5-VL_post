import os
from trainer.dpo_trainer import DPOTrainer
from trainer.train_configs import (
    TrainConfig, 
    OptimConfig, 
    DsConfig, 
    DsZero2Config,
    DsZero3Config,
    DsOffloadConfig,
    DataLoaderConfig,
    DPOConfig
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
    # 模型路径配置
    model_path = "Qwen2.5-VL-7B-Instruct"  # 可以是本地路径或 HF ID
    os.environ['TOKEN_DIR'] = model_path
    
    # DPO数据集路径
    data_path = '/root/autodl-tmp/data/dpo_dataset.jsonl'
    
    # 数据集配置
    file_dataset = JsonlFileDataset([data_path])
    
    # DeepSpeed 配置
    # 选项1: ZeRO-2 (推荐，显存较小时使用)
    ds_config = DsConfig(
        zero_config=DsZero2Config(
            offload_optimizer=DsOffloadConfig(device='cpu'),
        ),
        gradient_clipping=1.0
    )
    
    # 选项2: ZeRO-3 (显存足够时可选，取消注释使用)
    # ds_config = DsConfig(
    #     zero_config=DsZero3Config(
    #         offload_optimizer=DsOffloadConfig(device='cpu'),
    #         offload_param=DsOffloadConfig(device='cpu'),
    #     ),
    #     gradient_clipping=1.0
    # )

    # DPO 配置
    dpo_config = DPOConfig(
        ref_model_checkpoint={},  # 参考模型checkpoint，留空则使用初始模型作为参考
        loss_beta=0.1,            # DPO loss 中的 beta 参数，控制 KL 散度惩罚强度，小则自由，大则接近参考模型
        loss_label_smoothing=0.0, # 标签平滑，一般设为0，设置小值 (如 0.1) 防止过拟合
        loss_ipo=False,           # 是否使用 IPO loss
        nll_loss_coef=None        # 负对数似然损失系数，None表示不使用，设置为小值 (如 0.1) 可防止模型"忘记"如何正常生成文本
    )

    # 训练配置
    train_config = TrainConfig(
        n_epochs=3,
        batch_size=1,  # 每个GPU的批次大小，DPO需要同时前向chosen和rejected
        model_name_or_path=model_path,
        file_dataset=file_dataset,
        max_seq_len=2048,

        # LoRA 配置 (推荐开启以节省显存)
        use_lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

        # DeepSpeed配置
        ds_config=ds_config,
        
        # DPO配置
        dpo_config=dpo_config,
        
        # 优化器配置 (DPO通常使用较小的学习率)
        optim_config=OptimConfig(initial_lr=5e-6),
        
        # 梯度累积
        gradient_accumulation_steps=4,
        
        # DataLoader配置
        data_loader_config=DataLoaderConfig(
            data_loader_num_workers=0,  # 数据量 < 1000 条设为 0，防止多进程死锁
            data_loader_shuffle=True,
            data_loader_drop_last=False,
            data_loader_pin_memory=True
        )
    )

    # 创建DPO Trainer
    trainer = DPOTrainer(
        train_config=train_config,
        eval_prompts=[]  # 关闭评估提示词，只关注 Loss 曲线
    )
    
    print("=" * 60)
    print("开始 DPO 训练...")
    print(f"模型: {model_path}")
    print(f"数据集: {data_path}")
    print(f"批次大小: {train_config.batch_size}")
    print(f"梯度累积步数: {train_config.gradient_accumulation_steps}")
    print(f"有效批次大小: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"学习率: {train_config.optim_config.initial_lr}")
    print(f"DPO Beta: {dpo_config.loss_beta}")
    print(f"训练轮数: {train_config.n_epochs}")
    print(f"LoRA: {'启用' if train_config.use_lora else '禁用'}")
    if train_config.use_lora:
        print(f"  - Rank: {train_config.lora_rank}")
        print(f"  - Alpha: {train_config.lora_alpha}")
    print("=" * 60)
    
    trainer.train()

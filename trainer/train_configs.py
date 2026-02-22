from typing import Optional, Union, Callable, List, Mapping, Any, Tuple
from dataclasses import dataclass, field

import torch
from .tools import FileDataset


@dataclass(kw_only=True)
class DsOffloadConfig:
    """DeepSpeed Offload 配置（用于将参数或优化器状态Offload到 CPU/NVMe）"""
    device: str = 'cpu'       # 卸载的目标设备，默认为 'cpu'
    pin_memory: bool = True   # 是否使用锁页内存以加快 CPU-GPU 传输速度


@dataclass(kw_only=True)
class DsActivationCheckpointingConfig:
    """DeepSpeed 激活值检查点配置，用于节省显存
    """
    
    partition_activations: bool = True      # 是否在模型并行GPU之间对激活值进行分区:开启时，“检查点”数据也切分存储到不同的 GPU 上。当某个 GPU 需要使用某块数据进行重计算时，再通过通信从其他 GPU 那里拿过来。
    cpu_checkpointing: bool = False         # 是否将激活值检查点Offload到 CPU
    contiguous_memory_optimization: bool = True # 是否开启连续内存优化，减少内存碎片
    number_checkpoints: Optional[int] = None # 手动指定检查点的数量
    synchronize_checkpoint_boundary: bool = False # 是否在检查点边界处同步：如果设为 True，在每次进入或退出一个 Checkpoint 模块（通常是一个 Transformer Layer）时，所有的 GPU 必须互相等待，直到所有 GPU 都到达这个点
    profile: bool = False                   # 是否开启性能分析：记录日志


@dataclass(kw_only=True)
class DsZeROConfig:
    """DeepSpeed ZeRO基础配置
    ZeRO Stage 3：
    AllGather
        前向传播时收集各GPU参数
    ReduceScatter
        反向传播，梯度同步完成后，各GPU立刻释放显存，只保存自己负责更新的那部分参数的梯度
    AllReduce (传统 DDP)：
        反向传播，梯度同步完成后，每张卡拥有完整的全局梯度
        """
    stage: int                                          # ZeRO 阶段 (0, 1, 2, 3)
    allgather_partitions: Optional[bool] = True         # 是否在 AllGather 通信时对显存连续性和拷贝效率优化
    allgather_bucket_size: Optional[int] = 5e8          # AllGather 通信发包的大小
    overlap_comm: Optional[bool] = True                 # 是否让通信和计算重叠以提高速度
    reduce_scatter: Optional[bool] = True               # 是否使用 ReduceScatter 替代 AllReduce
    reduce_bucket_size: Optional[Union[str, int]] = 5e8 # 反向传播时，每一层计算完，都会产生这一层的梯度，该参数决定梯度达到多大显存才进行一次ReduceScatter，默认500MB
    contiguous_gradients: Optional[bool] = True         # 是否在生成梯度时保证内存连续


@dataclass(kw_only=True)
class DsZero0Config(DsZeROConfig):
    """ZeRO Stage 0 配置（相当于标准 DDP，不进行分片）"""
    stage: int = field(default=0, init=False)


@dataclass(kw_only=True)
class DsZero1Config(DsZeROConfig):
    """ZeRO Stage 1 配置（仅对优化器状态进行分片）"""
    stage: int = field(default=1, init=False)


@dataclass(kw_only=True)
class DsZero2Config(DsZeROConfig):
    """ZeRO Stage 2 配置（对优化器状态和梯度进行分片）"""
    stage: int = field(default=2, init=False)
    offload_optimizer: Optional[DsOffloadConfig] = None # 优化器状态卸载配置
    offload_param: Optional[DsOffloadConfig] = None     # 参数offload配置（Stage 2 不常用此项，多用于 Stage 3）


@dataclass(kw_only=True)
class DsZero3Config(DsZeROConfig):
    """ZeRO Stage 3 配置（对优化器状态、梯度和模型参数都进行分片）"""
    stage: int = field(default=3, init=False)
    sub_group_size: Optional[int] = 1e9                 # 参数通信时分组大小
    stage3_prefetch_bucket_size: Optional[Union[str, int]] = 'auto' # Stage 3 预取缓冲区大小。掩盖通信延迟：DeepSpeed 会预测接下来需要计算哪层参数，并在当前层还在计算时，提前开始从其他 GPU 下载下一层的参数。
    stage3_param_persistence_threshold: Optional[Union[str, int]] = 'auto' # 参数驻留阈值（小参数不切分，也不释放）
    stage3_max_live_parameters: Optional[int] = 1e9     # GPU 中保留的最大活跃参数量：限制预取参数，防止显存溢出
    stage3_max_reuse_distance: Optional[int] = 1e9      # 智能的缓存策略：如果参数在当前用完后，马上又要被用到（小于阈值，距离很近），就不释放它，直接留着给下次用
    stage3_gather_16bit_weights_on_model_save: Optional[bool] = True # 保存模型时是否收集完整的 16bit 权重
    offload_optimizer: Optional[DsOffloadConfig] = None # 优化器offload配置（Stage 3 常用）
    offload_param: Optional[DsOffloadConfig] = None     # offload_param配置：模型参数平时存在 CPU 里，计算前上传到 GPU，算完立刻删掉


@dataclass(kw_only=True)
class DsFp16Config:
    """FP16 混合精度训练配置
    在训练中，梯度往往非常小，FP16（半精度浮点数）表示范围很小，梯度小于 FP16 能表示的最小值，结果就会变成 0，即下溢，梯度变成 0，模型就学不到东西
    梯度缩放：在算梯度之前，先把 Loss 乘以一个2**initial_scale_power，这样计算出的梯度也会变大，等更新参数时，再把这个倍数除回去。
    """
    enabled: Union[str, bool] = 'auto' # 是否开启 FP16
    loss_scale: int = 0                # 初始 Loss Scale，0 表示动态调整
    loss_scale_window: int = 1000      # 动态 Loss Scale 调整窗口：如果连续 1000 个 Batch都没有发生上溢出，DeepSpeed 就会尝试把 loss_scale 增大一倍
    initial_scale_power: int = 16      # 初始 Scale 的幂次 (2^16)
    hysteresis: int = 2                # 冷却期：降低 Scale 之后，需要等待多少个 Step 才能重新开始 loss_scale_window 计数
    min_loss_scale: int = 1            # 最小缩放倍数：如果 Scale 小于 1（比如 0.5），那是在缩小梯度，会加剧下溢出问题
    fp16_opt_level: Optional[str] = 'O2' # APEX 优化等级
                                            # 如 O1: PyTorch 官方 AMP 的默认模式。部分层用 FP16，部分敏感层（如 Softmax）用 FP32。
                                            #    O2: DeepSpeed 的典型模式。
                                                    # 模型的主权重（Model Weights）在计算时被转换为 FP16。
                                                    # 优化器中保留一份 FP32 的“主权重备份”（Master Weights）用于更新。
                                                    # 输入数据会被立刻转为 FP16。


@dataclass(kw_only=True)
class DsBf16Config:
    """BF16 (Brain Float 16) 混合精度训练配置"""
    enabled: bool = True # 是否开启 BF16（通常在 A100/H100 等支持 BF16 的硬件上使用）


@dataclass(kw_only=True)
class DsConfig:
    """DeepSpeed 总配置类"""
    zero_config: Optional[DsZeROConfig] = field(default_factory=DsZero3Config) # ZeRO 配置，默认 Stage 3
    fp16_config: Optional[DsFp16Config] = field(default_factory=DsFp16Config)  # FP16 配置
    bf16_config: Optional[DsBf16Config] = field(default_factory=DsBf16Config)  # BF16 配置
    gradient_clipping: Optional[float] = 1.0                                   # 梯度裁剪阈值
    activation_checkpointing: Optional[DsActivationCheckpointingConfig] = None # 激活值检查点配置


@dataclass(kw_only=True)
class DataLoaderConfig:
    """
    data loader配置项
    """
    data_loader_pin_memory: bool = False # 是否使用锁页内存加速数据传输
    data_loader_num_workers: int = 0     # 数据加载的子进程数量
    data_loader_shuffle: bool = False    # 是否打乱数据
    data_loader_drop_last: bool = True   # 是否丢弃最后一个不完整的 Batch


@dataclass(kw_only=True)
class OptimConfig:
    """优化器配置"""
    optim_type: str = 'adam' # 优化器类型，如 'adam' 或 'lion'
    enable_lr_scheduler: bool = False    # 是否启用学习率调度器
    initial_lr: float                    # 初始学习率
    weight_decay: Optional[float] = None # 权重衰减系数
    betas: Optional[Tuple[float, float]] = None # Adam 优化器的 beta 参数 (beta1, beta2)
    warmup_iters: Optional[int] = None   # 预热步数
    max_lr: Optional[float] = None       # 最大学习率（用于 Cosine 等调度器）
    min_lr: Optional[float] = None       # 最低学习率
    cosine_annealing_period: Optional[int] = None # 余弦退火周期
    cosine_annealing_period_mul: int = 0 # 余弦退火周期的倍增因子（用于 SGDR）


@dataclass(kw_only=True)
class LossConfig:
    """通用损失函数配置"""
    critical_tokens: Optional[List[int]] = None # 关键 Token 的 ID 列表
    critical_alpha: float = 1.0                 # 关键 Token 的 Loss 权重
    aux_loss_coef: Optional[float] = 0.001      # 辅助 Loss 系数（如 MoE 的负载均衡 Loss）


@dataclass(kw_only=True)
class DPOConfig:
    """DPO训练配置"""
    ref_model_checkpoint: Mapping[str, Any] # 参考模型（Reference Model）的检查点路径/配置
    loss_beta: float                        # DPO Loss 中的 KL 散度约束系数 Beta
    loss_label_smoothing: float = 0.0       # 标签平滑系数
    loss_ipo: bool = False                  # 是否使用 IPO Loss
    nll_loss_coef: Optional[float] = None   # 负对数似然 Loss 的系数（如果混合训练）


@dataclass(kw_only=True)
class PPOConfig:
    """PPO强化学习训练配置"""
    ppo_epochs: int                         # 每次采集数据后 PPO 更新的轮数
    ppo_batch_size: int                     # PPO 更新时的 Batch Size
    ref_model_checkpoint: Mapping[str, Any] # 参考模型检查点（计算 KL 散度）
    value_model_checkpoint: Optional[Mapping[str, Any]] = None # 价值模型（Critic）检查点
    gamma: float = 1.0                      # 奖励折扣因子 (Discount Factor)
    lam: float = 0.95                       # GAE 的 Lambda 参数
    clip_eps: float = 0.1                   # PPO 的裁剪范围 epsilon
    vf_coef: float = 0.5                    # 价值函数损失（Value Loss）的系数
    kl_beta: float = 0.02                   # KL 散度惩罚系数
    kl_estimator: str = 'k1'                # KL 散度估计器类型 ('k1' 或 'k3')
    whiten_rewards: bool = False            # 是否对奖励进行归一化
    gen_max_new_tokens: int                 # 生成采样时的最大新 Token 数
    gen_temperature: Optional[float] = None # 采样温度
    gen_k: Optional[int] = None             # Top-K 采样
    gen_p: Optional[float] = None           # Top-P (Nucleus) 采样
    gen_suppress_tokens: Optional[list[int]] = None # 生成时需要抑制的 Token 列表


@dataclass(kw_only=True)
class GRPOConfig:
    """GRPO (Group Relative Policy Optimization) 训练配置"""
    grpo_steps: int = 1                     # GRPO 更新步数
    group_size: int = 12                    # 每组采样的样本数量（用于计算组内相对优势）
    mixup_alpha: float = 1.0                # Mixup 数据增强系数
    loss_beta: float = 0.0                  # KL 散度系数（GRPO 中有时设为 0 或 0.04）
    loss_clip_eps: float = 3e-4             # Loss 裁剪阈值
    loss_clip_eps_high: Optional[float] = 4e-4 # Loss 裁剪阈值上限
    loss_delta: Optional[float] = None      # Loss 计算中的 Delta 参数
    loss_importance_sampling_level: str = 'seq' # 重要性采样级别：'token' 级或 'seq' (序列) 级
    loss_type: str = 'grpo'                 # Loss 类型：'grpo', 'bnpo' 或 'dr_grpo'
    gen_max_new_tokens: int                 # 采样生成的最大长度
    gen_temperature: Optional[float] = None # 采样温度
    gen_k: Optional[int] = None             # Top-K 采样
    gen_p: Optional[float] = None           # Top-P 采样
    gen_suppress_tokens: Optional[list[int]] = None # 抑制 Token 列表


@dataclass(kw_only=True)
class KDConfig:
    """
    知识蒸馏模式配置项
    """
    # 教师模型 Logits 提供函数：输入 (input_ids, attention_mask)，输出 logits
    teacher_logits_provider: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    kd_coef: float = 0.4 # 蒸馏 Loss 在总 Loss 中的权重：loss = kd_coef * kd_loss + (1 - kd_coef) * lm_loss


@dataclass(kw_only=True)
class EvalConfig:
    """评估阶段的生成配置"""
    max_new_tokens: int = 1024   # 评估生成时的最大新 Token 数
    temperature: float = 1.0     # 采样温度
    top_p: float = 0.95          # Top-P 采样阈值
    top_k: Optional[float] = None # Top-K 采样阈值


@dataclass(kw_only=True)
class TrainConfig:
    """
    训练参数配置项
    """
    n_epochs: int                               # 训练总轮数
    batch_size: int                             # 单次训练的批次大小
    model_name_or_path: str                     # 预训练模型路径 (如 "Qwen/Qwen2.5-VL-7B-Instruct")

    file_dataset: FileDataset                   # 训练数据集对象
    max_seq_len: int                            # 训练时的最大序列长度
    is_vlm: bool = True                        # 是否为多模态任务
    # LoRA
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    data_loader_config: DataLoaderConfig = field(default_factory=DataLoaderConfig) # DataLoader 配置
    loss_config: LossConfig = field(default_factory=LossConfig)   # 损失函数通用配置
    optim_config: OptimConfig = field(default_factory=OptimConfig) # 优化器配置
    ds_config: DsConfig = field(default_factory=DsConfig) # DeepSpeed 配置

    # 以下为不同训练模式的配置，同一时间通常只启用一种
    kd_config: Optional[KDConfig] = None          # 知识蒸馏配置，为None时不使用知识蒸馏
    dpo_config: Optional[DPOConfig] = None        # DPO 配置
    ppo_config: Optional[PPOConfig] = None        # PPO 配置
    grpo_config: Optional[GRPOConfig] = None      # GRPO 配置

    mask_prompt: bool = True               # 是否在计算 Loss 时掩盖 Prompt 部分（只训练回答部分）
    gradient_accumulation_steps: int = 0   # 梯度累积步数（0 表示不累积）grpo训练时不生效该配置！
    eval_batch_interval: int = 100         # 每隔多少个 Batch 进行一次评估
    eval_config: EvalConfig = field(default_factory=EvalConfig) # 评估配置
    
    wandb_config: Optional['WandbConfig'] = None # WandB 配置


@dataclass(kw_only=True)
class WandbConfig:
    """WandB 配置"""
    enabled: bool = True
    project: str = "llm-train-sft"
    name: Optional[str] = None
    group: Optional[str] = None
    entity: Optional[str] = None

@dataclass(kw_only=True)
class VLMConfig:
    """
    VLM 模型特定配置兼容类。
    在使用 HF AutoModel 时，通常不需要手动配置模型架构，
    保留此类是为了兼容旧代码的引用或存放特定 VLM 超参。
    """
    vision_tower: Optional[str] = None
    image_token_id: Optional[int] = None
from typing import Optional, Tuple, List, Dict, Any
import copy

import torch
from torch.utils.data import Dataset
# 导入PyTorch分布式训练模块
import torch.distributed as dist

from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoModelForVision2Seq, 
    Qwen2_5_VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, TaskType

# 从当前包导入并行处理模块
from .parallel import DsParallel
# 从当前包导入训练工具类
from .tools import TrainerTools
# 从当前包导入损失函数类
from .loss import LMLoss, KDLoss
# 从当前包导入评估任务提交函数
from .eval import submit_gen_task
# 从当前包导入模型解包工具，用于生成任务
from .partition_utils import unwrap_model_for_generation

# 从当前包导入训练配置类
from .train_configs import (
    TrainConfig,
    VLMConfig,
    DsZero2Config,
    DsZero3Config
)

# 从当前包导入学习率调度器类
from .scheduler import (
    LRScheduler,
    WarmupCosineAnnealingLRScheduler,
    NoneLRScheduler
)

# 从当前包导入检查点相关函数
from .checkpoint import (
    load_checkpoint,
    save_checkpoint,
    load_steps,
    save_steps,
)

# 从当前包导入工具函数
from .utils import (
    set_seed,  # 设置随机种子
    autocast,  # 自动混合精度上下文管理器
    create_doc_boundary_mask,  # 创建文档边界掩码
    generate_position_ids,  # 生成位置ID
    pretrain_collate_fn,  # 预训练数据整理函数
)

from .log import log

class Trainer:
    # 初始化方法，接收训练配置和评估提示
    def __init__(
            self,
            *,
            train_config: TrainConfig,  # 训练配置对象
            eval_prompts: List[str],  # 评估用的提示文本列表
            eval_image_tags: Optional[List[str]] = None  # 评估用的图片标签列表（可选）
    ):
        set_seed()

        # 是否打包序列（仅预训练阶段需要），格式如[[1,1,eos,2,2,eos]]
        self.packed_sequences = False # 对于 VLM 通常建议关闭打包，避免图像位置混乱

        # 保存训练配置到实例属性
        self.train_config = train_config
        self.eval_prompts = eval_prompts
        self.eval_image_tags = eval_image_tags
        # 评估索引初始化为-1，用于循环取评估样本
        self.eval_idx = -1
        # 上一次的全局步数，用于恢复训练
        self.last_global_steps = 0

        # 转换训练参数，获取并行配置、数据加载器配置、采样器配置
        self.parallel_kwargs, self.data_loader_kwargs, self.sampler_kwargs = self._convert_train_args()
        # 初始化梯度缩放器（用于混合精度训练），是否启用由TrainerTools的use_amp决定
        self.scalar = torch.GradScaler(enabled=TrainerTools().use_amp)

        # 学习率
        # 单卡：跑完 10000 条数据，需要走 10000/B 步
        # N卡：跑完 10000 条数据，只需要走10000/（B*N） 步
        # 那么跑完同样的 Epoch（轮数），参数更新的总距离就只有单卡时的 1/N，模型可能还没收敛训练就结束了，为了在更少的步数内走完同样的优化路程，每一步放大学习率 *N
        initial_lr = train_config.optim_config.initial_lr

        # 初始化训练模型和优化器
        self.train_model, self.optimizer = self._init_train_model_and_optim(initial_lr)
        # 初始化学习率调度器
        self.lr_scheduler = self._init_lr_scheduler(initial_lr, self.optimizer)
        # 初始化损失函数（主损失，KD损失）
        self.criterion, self.kd_loss = self._init_loss()

        # 加载检查点
        load_checkpoint(
            self.train_model,
            optimizer=self.optimizer,
            device=TrainerTools().parallel.device
        )

        # 加载训练步数
        steps_dict = load_steps()
        # 恢复步数到检查点
        self._apply_restore_ckpt(steps_dict)

        # 初始化 Processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.train_config.model_name_or_path,
                trust_remote_code=True
            )
            log("Processor 加载成功：自动识别文本/图像预处理")
        except Exception as e:
            log(f"⚠ Processor 加载失败: {e}")
            self.processor = None

    # 根据训练配置创建新模型
    def _new_model(self, train_config: TrainConfig):
        model_path = train_config.model_name_or_path
        log(f"正在加载模型（自动兼容 VLM / LLM）: {model_path}")

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # ---- Step 1: 自动加载配置，判断是否为 VLM ----
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        is_vlm = hasattr(config, "vision_config") or "vision" in config.to_dict()

        # ---- Step 2: 加载模型 ----
        if is_vlm:
            log("检测到 Vision-Language 模型（VLM），使用 AutoModelForCausalLM 自动加载 VLM 类...")
        else:
            log("检测到文本模型（LLM），按 CausalLM 加载...")

        # 关键：所有 Qwen-VL 模型都是 CausalLM + 自定义类，不是 Vision2Seq！
        if is_vlm:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,          # 必须，Qwen2.5-VL 是用它加载自定义类的
                _attn_implementation="flash_attention_2"
            )

        # ---- Step 3: LoRA ----
        if train_config.use_lora:
            log("正在应用 LoRA 微调...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=train_config.lora_rank,
                lora_alpha=train_config.lora_alpha,
                lora_dropout=train_config.lora_dropout,
                target_modules=train_config.lora_target_modules
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # ---- Step 4: Gradient Checkpointing ----
        if train_config.ds_config.activation_checkpointing:
            # 开启梯度检查点
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            
            # ================= [新增关键修复] =================
            # 解决 "element 0 of tensors does not require grad" 问题
            # 当使用 LoRA + Gradient Checkpointing 时，必须开启这个
            # 它会让 Embedding 层的输出 requires_grad=True，连接起计算图
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                # 兜底方案：如果模型没有封装该方法，手动注册 Hook
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        return model
    
    def _get_trainable_params(self, model):
        # 无论是否配置冻结，永远只将 requires_grad=True 的参数传给 DeepSpeed 优化器
        return filter(lambda p: p.requires_grad, model.parameters())

    # 初始化训练模型和优化器
    def _init_train_model_and_optim(self, initial_lr: float):
        # 创建新模型实例
        model = self._new_model(self.train_config)

        # 如果是主进程，打印模型参数信息
        if TrainerTools().parallel.is_main_process:
            # .numel()计算参数量，比如形状为(2, 3)的张量，参数为（2*3）=6
            # 计算总参数数量
            total = sum(p.numel() for p in model.parameters())
            log(f"总参数量: {total}")

            # 计算可训练参数数量
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log(f"总参数: {total/1e9:.2f}B, 可训练: {trainable/1e9:.4f}B")

            # 计算模型总大小（按float32计算）
            total_size_bytes = total * 4
            total_size_mb = total_size_bytes / (1024 * 1024)
            log(f"模型总大小: {total_size_mb:.2f} MB")

        # 借助分布式训练框架（如 DeepSpeed）对模型和优化器进行 “包装”，自动实现多GPU并行计算
        model, optim = TrainerTools().parallel.process(
            model=model,  # 原始模型
            optimizer=self._config_optim(model, initial_lr),  # 配置好的优化器
            kwargs=self.parallel_kwargs  # 并行配置参数
        )

        # 返回处理后的模型和优化器
        return model, optim

    # 配置优化器
    def _config_optim(self, model, initial_lr):
        # 初始化优化器为None
        optimizer = None
        # 判断是否使用Lion优化器
        use_lion_optim = self.train_config.optim_config.optim_type == 'lion'

        # 如果使用DeepSpeed并行且有并行配置
        if isinstance(TrainerTools().parallel, DsParallel) and self.parallel_kwargs:
            import deepspeed
            # 检查配置中是否开启了 offload_optimizer 且目标设备是 cpu
            # 如果都满足，DeepSpeed的ZeRO技术将优化器状态从GPU移动到CPU中，节省显存
            if ('zero_optimization' in self.parallel_kwargs
                    and 'offload_optimizer' in self.parallel_kwargs['zero_optimization']
                    and self.parallel_kwargs['zero_optimization']['offload_optimizer']['device'] == 'cpu'):
                if self.train_config.optim_config.optim_type == 'lion':# 如果配置Lion优化器
                    optimizer = deepspeed.ops.lion.DeepSpeedCPULion
                else: # 使用Adam优化器
                    optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam
            # 如果否，使用“GPU 优化器”
            else:
                if self.train_config.optim_config.optim_type == 'lion':
                    optimizer = deepspeed.ops.lion.FusedLion
                else:
                    optimizer = deepspeed.ops.adam.FusedAdam

        # 非DeepSpeed场景
        if not optimizer:
            # 如果配置Lion优化器
            if self.train_config.optim_config.optim_type == 'lion':
                import lion_pytorch
                # 使用lion_pytorch的Lion优化器
                optimizer = lion_pytorch.Lion
            else:
                # 使用PyTorch原生的AdamW优化器
                optimizer = torch.optim.AdamW

        # 获取优化器的betas参数（动量参数）
        betas = self.train_config.optim_config.betas
        # 获取权重衰减参数
        weight_decay = self.train_config.optim_config.weight_decay

        # 如果未配置betas，使用默认值
        if betas is None:
            if use_lion_optim:
                betas = (0.95, 0.98)  # Lion默认betas
            else:
                betas = (0.9, 0.999)  # AdamW默认betas

        # 如果未配置weight_decay，使用默认值
        if weight_decay is None:
            if use_lion_optim:
                weight_decay = 0.015  # Lion默认权重衰减
            else:
                weight_decay = 0.01   # AdamW默认权重衰减

        # 初始化并返回优化器
        return optimizer(
            self._get_trainable_params(model),  # 可训练参数
            lr=initial_lr,  # 初始学习率
            betas=betas,    # 动量参数
            weight_decay=weight_decay  # 权重衰减
        )

    # 初始化学习率调度器
    def _init_lr_scheduler(self, initial_lr: float, optimizer) -> LRScheduler:
        # 如果启用学习率调度器
        if self.train_config.optim_config.enable_lr_scheduler:
            # 获取调度器配置参数
            warmup_iters = self.train_config.optim_config.warmup_iters  # 预热步数
            min_lr = self.train_config.optim_config.min_lr              # 最小学习率
            max_lr = self.train_config.optim_config.max_lr              # 最大学习率
            cosine_annealing_period = self.train_config.optim_config.cosine_annealing_period  # 余弦退火周期
            cosine_annealing_period_mul = self.train_config.optim_config.cosine_annealing_period_mul  # 周期倍增系数

            # 预热余弦退火调度器
            return WarmupCosineAnnealingLRScheduler(
                optimizer=optimizer,
                warmup_iters=warmup_iters,
                initial_lr=initial_lr,
                min_lr=min_lr,
                max_lr=max_lr,
                cosine_annealing_period=cosine_annealing_period,
                cosine_annealing_period_mul=cosine_annealing_period_mul,
                need_log=TrainerTools().parallel.is_main_process  # 是否需要打印日志
            )

        # 不启用调度器时，返回空调度器
        return NoneLRScheduler(initial_lr)

    # 初始化损失函数
    def _init_loss(self):
        # 损失：关键 Token 加权
        # 关键token列表
        # EOS：如果模型学不会这个，它生成内容时就会停不下来
        # 对话分隔符：比如 <user>, <assistant>, <system>
        # 思维链标记：比如 DeepSeek R1 中的 <think> 和 </think>
        critical_tokens: Optional[List[int]] = None
        # 设置默认值：关键token的损失权重
        critical_alpha: float = 1.0
        if self.train_config.loss_config.critical_tokens:
            critical_tokens = self.train_config.loss_config.critical_tokens
            critical_alpha = self.train_config.loss_config.critical_alpha

        # 初始化主损失函数（语言模型损失）
        criterion = LMLoss(
            critical_tokens=critical_tokens,  # 关键token
            critical_alpha=critical_alpha,    # 关键token权重
            vocab_size=TrainerTools().tokenizer.vocab_size  # 词表大小
        )

        # 如果配置了KD（知识蒸馏），初始化KD损失；否则为None
        kd_loss = KDLoss() if self.train_config.kd_config else None

        # 返回主损失和KD损失
        return criterion, kd_loss

    # 恢复训练进度相关的全局步数、学习率调度器状态，保证训练的连续性
    def _apply_restore_ckpt(self, steps_dict):
        if steps_dict:
            # 恢复全局步数
            self.last_global_steps = steps_dict['global_steps']
            # 处理步数为0的情况
            if not self.last_global_steps:
                self.last_global_steps = 0

            # 恢复学习率调度器状态
            self.lr_scheduler.restore_ckpt_dict(steps_dict)


    # 转换训练参数为并行配置、数据加载器配置、采样器配置
    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        # 默认并行配置：None
        parallel_kwargs: Optional[Dict[str, Any]] = None
        # 如果使用DeepSpeed并行且有DS配置
        if isinstance(TrainerTools().parallel, DsParallel) and self.train_config.ds_config:
            # 基础并行配置
            parallel_kwargs = {
                'gradient_accumulation_steps': max(1, self.train_config.gradient_accumulation_steps),  # 使用配置的梯度累积步数
                'gradient_clipping': self.train_config.ds_config.gradient_clipping,  # 防止梯度爆炸，设置梯度裁剪阈值
                'train_micro_batch_size_per_gpu': self.train_config.batch_size  # 单张显卡在一次前向/反向传播中处理的数据量
            }

            # 如果配置了Zero优化
            if self.train_config.ds_config.zero_config:
                zero_config = self.train_config.ds_config.zero_config
                # ZeRO 阶段，决定显存节省的力度：
                # Stage 0: 不使用 ZeRO
                # Stage 1: 切分优化器状态
                # Stage 2: 切分优化器状态 + 梯度
                # Stage 3: 切分优化器状态 + 梯度 + 模型参数
                zero_optimization: Dict[str, Any] = {'stage': zero_config.stage}

                # Zero优化可选参数配置
                # AllGather:ZeRO 把模型参数切分了，GPU 0 只维护第一层，GPU 1 只维护第二层.....
                # 前向传播时，GPU 0 把第一层发给所有人，GPU 1 把第二层发给所有人……结束后，所有 GPU 暂时都拥有了完整的模型
                if zero_config.allgather_partitions is not None:
                    zero_optimization['allgather_partitions'] = zero_config.allgather_partitions
                # Bucket Size (缓冲区大小)：算出的梯度先存起来，等攒够了再与其他GPU通信
                if zero_config.allgather_bucket_size is not None:
                    zero_optimization['allgather_bucket_size'] = zero_config.allgather_bucket_size
                # 让计算和通信并行执行。比如在计算第 N 层的同时，传输第 N+1 层的数据
                if zero_config.overlap_comm is not None:
                    zero_optimization['overlap_comm'] = zero_config.overlap_comm
                # ReduceScatter：
                # 把所有梯度加起来，求平均梯度，但是不把完整结果给任何人，而是切开分发，每个卡只存储它负责更新的那一小部分梯度
                if zero_config.reduce_scatter is not None:
                    zero_optimization['reduce_scatter'] = zero_config.reduce_scatter
                if zero_config.reduce_bucket_size is not None:
                    zero_optimization['reduce_bucket_size'] = zero_config.reduce_bucket_size
                # 在显存中开辟一块连续缓冲区来存储梯度，减少显存碎片
                if zero_config.contiguous_gradients is not None:
                    zero_optimization['contiguous_gradients'] = zero_config.contiguous_gradients

                # 如果是Zero2或Zero3配置
                if isinstance(zero_config, DsZero2Config) or isinstance(zero_config, DsZero3Config):
                    # 将优化器状态移到 CPU 中
                    # pin_memory锁页内存：加速CPU——GPU数据传输
                    if zero_config.offload_optimizer is not None:
                        zero_optimization['offload_optimizer'] = {
                            "device": zero_config.offload_optimizer.device,
                            "pin_memory": zero_config.offload_optimizer.pin_memory
                        }
                    # 将模型参数移到 CPU 中（仅 Stage 3 支持）
                    if zero_config.offload_param is not None:
                        zero_optimization['offload_param'] = {
                            "device": zero_config.offload_param.device,
                            "pin_memory": zero_config.offload_param.pin_memory
                        }

                # Zero3特有配置
                if isinstance(zero_config, DsZero3Config):
                    # 把 GPU 分成小组，有 1024 张 GPU，但是分成8组，不再是把参数切成 1024 份，而是只切成 8 份。单卡显存节省的程度不如全局切分那么极致，但换来了更高的通信稳定性和速度
                    if zero_config.sub_group_size is not None:
                        zero_optimization['sub_group_size'] = zero_config.sub_group_size
                    # ZeRO-3 预取参数的缓冲区大小：DeepSpeed 会提前从其他卡拉取下一层计算所需的参数
                    if zero_config.stage3_prefetch_bucket_size is not None:
                        zero_optimization['stage3_prefetch_bucket_size'] = zero_config.stage3_prefetch_bucket_size
                    # 如果是很小的参数（小于param_persistence_threshold阈值）就不切分了，直接留在GPU上，减少通信延迟
                    if zero_config.stage3_param_persistence_threshold is not None:
                        zero_optimization['stage3_param_persistence_threshold'] = zero_config.stage3_param_persistence_threshold
                    # 计算过程中，显存中允许同时存在的最大完整参数量。防止为了计算构建过多完整层导致显存溢出
                    if zero_config.stage3_max_live_parameters is not None:
                        zero_optimization['stage3_max_live_parameters'] = zero_config.stage3_max_live_parameters
                    # 如果某个参数计算完后，很久之后（超过此距离）才会被再次用到，就立即释放它以节省显存；否则保留在缓存中避免重复传输。
                    if zero_config.stage3_max_reuse_distance is not None:
                        zero_optimization['stage3_max_reuse_distance'] = zero_config.stage3_max_reuse_distance
                    # 是否自动把分散在各卡的参数收集起来保存为一个完整的 fp16 模型文件，如果为 True，保存出的模型可以直接用 torch.load 加载；如果为 False，保存的是切分后的权重
                    if zero_config.stage3_gather_16bit_weights_on_model_save is not None:
                        zero_optimization['stage3_gather_16bit_weights_on_model_save'] = zero_config.stage3_gather_16bit_weights_on_model_save

                # 将Zero配置加入并行参数
                parallel_kwargs['zero_optimization'] = zero_optimization

            # BF16配置：指数位和 FP32 一样多
            if (self.train_config.ds_config.bf16_config is not None
                    and self.train_config.ds_config.bf16_config.enabled):
                bf16_config = self.train_config.ds_config.bf16_config
                bf16 = {'enabled': bf16_config.enabled}
                parallel_kwargs['bf16'] = bf16
            # FP16配置（兼容20系等旧版显卡）
            elif self.train_config.ds_config.fp16_config:
                fb16_config = self.train_config.ds_config.fp16_config
                # fb16对于比10^−10还小的值直接视为0，因此需要放大系数
                # 算出 Loss 后，立刻乘以放大系数，用这个放大的 Loss 算梯度，梯度也就跟着放大了，FP16 就能存下非零的数值了
                # 在更新权重之前，把梯度除以这个系数，还原回真实的大小
                fp16 = {
                    'enabled': fb16_config.enabled,
                    'loss_scale': fb16_config.loss_scale,# 放大系数  通常设为0，表示启用“动态”模式，DeepSpeed会自动调整这个值
                    'loss_scale_window': fb16_config.loss_scale_window,# 窗口期，如果多少步没有发生溢出DeepSpeed就会尝试把放大系数乘以 2
                    'initial_scale_power': fb16_config.initial_scale_power,# 初始放大倍数的指数
                    'hysteresis': fb16_config.hysteresis,# 放大倍数太大了，导致梯度变成无穷大 Inf，DeepSpeed会立刻减小放大系数。这个参数决定了减小后要等待多久才允许再次尝试增大放大系数。
                    'min_loss_scale': fb16_config.min_loss_scale# 放大倍数的底线
                }
                # 混合精度的优化等级
                # DeepSpeed 默认推荐02：模型权重大部分都被转换为 FP16，只有优化器更新时，会维护一份 FP32 的“主权重”来保证累积精度。
                if fb16_config.fp16_opt_level is not None:
                    fp16['fp16_opt_level'] = fb16_config.fp16_opt_level

                parallel_kwargs['fp16'] = fp16

            # 激活检查点配置（节省显存）
            if self.train_config.ds_config.activation_checkpointing:
                activation_checkpointing_config = self.train_config.ds_config.activation_checkpointing
                # # partition_activations：分布式训练时，将激活值分片存储到不同 GPU，而非每个 GPU 存储完整的激活值
                    # cpu_checkpointing：将激活值存储到CPU 内存
                    # contiguous_memory_optimization：确保激活值存储在连续的内存块中，减少内存碎片，提升内存访问效率
                    # synchronize_checkpoint_boundary：强制各 GPU 在检查点的边界处同步操作（比如同时开始/结束计算），保证多 GPU 间的计算一致性，避免因 GPU 进度差异导致的错误（如激活值计算顺序混乱）
                    # profile：开启性能分析：记录激活检查点的时间开销（如重新计算激活值的耗时）、显存节省量等数据
                activation_checkpointing: Dict[str, Any] = {
                    'partition_activations': activation_checkpointing_config.partition_activations,
                    'cpu_checkpointing': activation_checkpointing_config.cpu_checkpointing,
                    'contiguous_memory_optimization': activation_checkpointing_config.contiguous_memory_optimization,
                    'synchronize_checkpoint_boundary': activation_checkpointing_config.synchronize_checkpoint_boundary,
                    'profile': activation_checkpointing_config.profile
                }

                if activation_checkpointing_config.number_checkpoints is not None:
                    activation_checkpointing['number_checkpoints'] = activation_checkpointing_config.number_checkpoints

                parallel_kwargs['activation_checkpointing'] = activation_checkpointing

        # WandB 配置注入 DeepSpeed
        if self.train_config.wandb_config and self.train_config.wandb_config.enabled:
            wandb_config = {
                "enabled": True,
                "project": self.train_config.wandb_config.project,
            }
            # DeepSpeed 只支持 project 和 entity，不支持 name 和 group
            # 如果需要自定义 run name，可以通过环境变量 WANDB_RUN_ID 或手动初始化 wandb
            if self.train_config.wandb_config.entity:
                wandb_config["entity"] = self.train_config.wandb_config.entity
            
            if parallel_kwargs is None:
                parallel_kwargs = {}
            parallel_kwargs['wandb'] = wandb_config
            
            # 通过环境变量设置 run name (如果用户指定了)
            if self.train_config.wandb_config.name:
                import os
                os.environ['WANDB_NAME'] = self.train_config.wandb_config.name
            if self.train_config.wandb_config.group:
                import os
                os.environ['WANDB_RUN_GROUP'] = self.train_config.wandb_config.group

        # 数据加载器配置：采样器选索引、数据集取样本、整理成批次
        # 用sampler（采样器）获取样本索引
        # 用num_workers（多线程）加速从数据集取样本
        # 用collate_fn把零散样本整理成批次（比如拼接张量、处理 padding等）
        # 用pin_memory加速 “内存→GPU 显存” 的数据传输
        dataloader_args = self.train_config.data_loader_config
        data_loader_kwargs = {
            "batch_size": self.train_config.batch_size,  # 批次大小
            "pin_memory": dataloader_args.data_loader_pin_memory,  # 是否使用pin_memory
            "collate_fn": pretrain_collate_fn,  # 数据整理函数：默认，填充
            "num_workers": dataloader_args.data_loader_num_workers,  # 线程数
            "shuffle": dataloader_args.data_loader_shuffle,  # 是否打乱数据
            "drop_last": dataloader_args.data_loader_drop_last,  # 是否丢弃最后不完整批次
        }
        # 采样器配置：生成 “样本索引”
        sampler_kwargs = {
            "shuffle": dataloader_args.data_loader_shuffle,  # 是否打乱
            "drop_last": dataloader_args.data_loader_drop_last,  # 是否丢弃最后批次
        }

        # 返回三个配置字典
        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    # 初始化参考模型参数/教师模型
    def _init_ref_model_args(self) -> dict:
        # 深拷贝并行配置（避免修改原配置）
        parallel_kwargs = copy.deepcopy(self.parallel_kwargs) if self.parallel_kwargs else None

        # 如果使用DeepSpeed
        if parallel_kwargs and isinstance(TrainerTools().parallel, DsParallel):
            # 参考模型是冻结的，它只前向传播来产生 logits 或 value，永远不会运行反向传播
            parallel_kwargs.pop('activation_checkpointing', None)  # 移除激活检查点
            parallel_kwargs.pop('gradient_clipping', None)         # 移除梯度裁剪

            # 强制将参考模型的 ZeRO 优化阶段设为 Stage 0
            # ZeRO Stage 1/2是专门用来优化优化器状态和梯度的显存占用的，参考模型没有优化器状态，也没有梯度
            # 虽然 Stage 3 可以切分模型参数，但在同一个训练脚本中同时运行两个 DeepSpeed 引擎（一个训练用，一个推理用）极其容易卡住
            parallel_kwargs["zero_optimization"] = {"stage": 0}

        # 返回处理后的并行配置
        return parallel_kwargs

    # 创建数据集实例
    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        max_seq_len = self.train_config.max_seq_len
        raise NotImplementedError("请在子类中重写此方法（SFTTrainer/DPOTrainer/GRPOTrainer）")

    # 计算损失（主损失+KD损失）
    def _calc_loss(self, inputs, attention_mask, logits, labels):
        # HF 模型通常在 forward 中计算 loss 并返回
        # 如果模型输出了 loss，直接使用它 (这是最准确的，因为它处理了内部的 shift logits)
        if isinstance(logits, dict) and 'loss' in logits:
             return logits['loss']
        # 或者 logits 是 output 对象
        elif hasattr(logits, 'loss') and logits.loss is not None:
             return logits.loss
        
        # 如果模型只返回了 logits，我们手动计算
        # 注意：inputs/logits/labels 的 shape 必须要对齐
        # VLM 的 labels 处理比较复杂，通常建议让模型自己算 loss
        
        vocab_size = logits.size(-1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return loss

    # 反向传播损失
    def _backward_loss(self, loss):
        # 如果使用DeepSpeed，调用DeepSpeed的backward
        if isinstance(TrainerTools().parallel, DsParallel):
            self.train_model.backward(loss)
        else:
            # 否则使用梯度缩放器的backward（混合精度）
            self.scalar.scale(loss).backward()

    # 应用梯度裁剪
    def _apply_grad_clipping(self):
        # DeepSpeed已集成梯度裁剪，这里构造非DeepSpeed场景的手动裁剪：
        if not isinstance(TrainerTools().parallel, DsParallel) and self.lr_scheduler.can_clip_grad():
            # 反向传播计算出的梯度值往往很小，小到半精度存不下，被当成 0，前向传播后，把损失值放大 N 倍，再基于放大的损失做反向传播
            # 反向传播后、参数更新前，把梯度再缩小回来
            self.scalar.unscale_(self.optimizer)
            # 裁剪（最大范数1.0）：如果梯度的 L2 范数超过 1.0，就把梯度整体乘以阈值/当前范数，让范数刚好等于 1.0
            torch.nn.utils.clip_grad_norm_(self._get_trainable_params(self.train_model), 1.0)

    # 执行优化步骤（更新参数+学习率调度）
    def _apply_step(self):
        # 更新学习率
        self.lr_scheduler.step()
        # 如果使用DeepSpeed，调用DeepSpeed的step
        if isinstance(TrainerTools().parallel, DsParallel):
            self.train_model.step()
        else:
            # 否则使用梯度缩放器的step更新参数
            self.scalar.step(self.optimizer)
            # 更新梯度缩放器的scale
            self.scalar.update()
            # 清空梯度（设为None更省显存）
            self.optimizer.zero_grad(set_to_none=True)

        # 同步所有进程（分布式训练）
        TrainerTools().parallel.synchronize()

    # 获取评估数据（循环取评估样本）
    def _get_eval_data(self) -> Tuple[Optional[str], Optional[str]]:
        # 如果没有评估提示，返回None
        if len(self.eval_prompts) == 0:
            return None, None

        # 评估索引自增
        self.eval_idx += 1
        # 超过列表长度则重置：以便可以循环
        if self.eval_idx == len(self.eval_prompts):
            self.eval_idx = 0

        # 如果没有评估图片标签，返回提示和None
        if not self.eval_image_tags:
            return self.eval_prompts[self.eval_idx], None

        # 返回提示和对应的图片标签
        return self.eval_prompts[self.eval_idx], self.eval_image_tags[self.eval_idx]

    # 日志记录函数
    def _log(self, keys: Dict[str, any], values: Dict[str, any]):
        """
        日志格式：keys_key1: keys_value1, keys_key2: keys_value2 -> values_key1: values_value1, values_key2: values_value2
        """
        # 仅主进程记录日志
        if TrainerTools().parallel.is_main_process:
            # 拼接keys部分
            log_tags = ', '.join([f'{k}: {v}' for k, v in keys.items()])
            # 拼接values部分
            log_values = ', '.join([f'{k}: {v}' for k, v in values.items()])
            # 组合日志消息
            log_msg = f'{log_tags} -> {log_values}'

            # 打印到控制台
            log(log_msg)
            # 写入日志文件
            log(f"{log_msg}\n", 'log.txt')

            # WandB 日志
            if self.train_config.wandb_config and self.train_config.wandb_config.enabled:
                import wandb
                if wandb.run:
                    wandb.log(values)

    # 异常处理函数
    def _on_exception(
            self,
            e: Exception,
            epoch: int,
            batch: int
    ):
        # 获取异常发生的文件
        exception_file = e.__traceback__.tb_frame.f_globals["__file__"]
        # 获取异常发生的行号
        exception_line = e.__traceback__.tb_lineno
        # 构造异常信息
        log_msg = f"epoch: {epoch}, batch: {batch} -> {e} at {exception_file} line {exception_line}\n"
        # 写入异常日志文件
        log(log_msg, 'exception.txt')

        # 重新抛出异常
        raise e

    # 获取模型的数据类型
    def _get_model_dtype(self):
        # 如果使用DeepSpeed
        if isinstance(TrainerTools().parallel, DsParallel):
            import deepspeed
            # 断言模型是DeepSpeed引擎
            assert isinstance(self.train_model, deepspeed.DeepSpeedEngine)
            # 返回模型的数据类型
            return self.train_model.get_data_types()[0]
        else:
            # 否则根据GPU返回bf16或fp16
            return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # 评估函数
    def _eval(self, tag: str):
        # 如果没有评估提示词，直接退出
        if not self.eval_prompts:
            return
        # 将被分布式框架包装的模型，还原为原生的模型实例
        with unwrap_model_for_generation(self.train_model) as eval_model:
            # 仅主进程执行评估
            if TrainerTools().parallel.is_main_process:
                # 获取评估数据
                eval_prompt, eval_image_tag = self._get_eval_data()

                # 如果有eval_prompt
                if eval_prompt:
                    #  设为eval模式
                    eval_model = self._check_eval_model(eval_model)
                    eval_model.eval()

                    # 提交生成任务
                    submit_gen_task(
                        eval_model,
                        self.train_config,
                        tag=tag,
                        prompt=eval_prompt,
                        image_path=eval_image_tag,
                        processor=self.processor
                    )

                    # 恢复训练模式
                    eval_model.train()

        # 等待所有进程完成评估
        TrainerTools().parallel.wait('eval')

    # 检查评估模型（钩子函数，可重写）
    def _check_eval_model(self, eval_model):
        return eval_model

    # 批次结束：执行评估
    def _on_batch_end(self, tag: str):
        self._eval(f'sign:batch/{tag}')

    # Epoch结束：执行评估
    def _on_epoch_end(self, tag: str):
        self._eval(f'sign:epoch/{tag}')

    # 打印日志：文件开始处理
    def _on_file_start(
            self,
            epoch: int,
            file_name: str
    ):
        # 仅主进程打印
        if TrainerTools().parallel.is_main_process:
            log(f"====epoch: {epoch}, start train {file_name}====\n", 'log.txt')

    # 计算平均损失（分布式场景下）
    def _avg_loss(self, losses: List[float], gradient_accumulation_steps, batches_accumulated) -> List[float]:
        avg_losses = []
        # 遍历每个损失值
        for loss in losses:
            # loss累积过程=每一批次的loss/梯度累积步数
            #  原始 Loss 的总和=梯度累积步数*loss
            # 计算平均损失=原始 Loss 的总和/实际的批次数量
            avg_loss = torch.tensor(
                loss * gradient_accumulation_steps / batches_accumulated,
                device=TrainerTools().parallel.device)

            # 如果是分布式训练，进行all_reduce求平均
            if TrainerTools().parallel.parallel_train:
                # 收集所有卡的avg_loss
                # dist.ReduceOp.AVG:求平均值
                # 将结果写回每张卡的 avg_loss 变量中
                dist.all_reduce(avg_loss, dist.ReduceOp.AVG)

            # 添加到平均损失列表
            avg_losses.append(avg_loss.detach().item())

        # 返回平均损失列表
        return avg_losses

    # 训练主函数
    def train(self):
        # 获取梯度累积步数
        gradient_accumulation_steps = self.train_config.gradient_accumulation_steps
        # 初始化全局步数
        global_steps = 0
        # 开关：是否跳过训练（用于恢复训练时跳过已训练步数）
        skipping_train = False

        # 损失累积变量（主任务 Loss + 辅助 Loss）
        loss_with_aux_accumulation = 0.0
        # 模型的 CrossEntropy Loss（交叉熵损失）（不含辅助损失）
        loss_without_aux_accumulation = 0.0
        # MoE Loss（负载均衡损失）
        aux_loss_accumulation = 0.0
        # 累积的批次数量
        batches_accumulated = 0

        # 遍历每个epoch
        for epoch in range(self.train_config.n_epochs):
            # 模型设为训练模式
            self.train_model.train()
            # 训练文件数量
            file_count = len(self.train_config.file_dataset)

            # 遍历每个训练文件
            for file_idx in range(file_count):
                # 创建数据集和获取文件路径
                dataset, file_path = self._create_dataset(file_idx)
                # 创建数据加载器（处理分布式采样）
                train_data_loader = TrainerTools().parallel.process_dataloader(
                    dataset=dataset,
                    data_loader_kwargs=self.data_loader_kwargs,
                    sampler_kwargs=self.sampler_kwargs
                )

                # 初始化上一次保存检查点的批次
                last_ckpt_batch = 0
                # 当前文件的批次总数
                batch_count_per_file = len(train_data_loader)

                # 分布式训练：epoch开始处理
                TrainerTools().parallel.on_epoch_start(epoch)
                # 打印日志：文件开始处理
                self._on_file_start(epoch, file_path)

                # 遍历每个批次
                for batch, batch_data in enumerate(train_data_loader):
                    global_steps += 1
                    # 如果当前步数小于已训练步数，跳过
                    if global_steps < self.last_global_steps:
                        skipping_train = True
                        continue

                    # 判断是否需要更新梯度（梯度累积）
                    if skipping_train:
                        need_update_grad = False
                    elif gradient_accumulation_steps > 1:
                        # 梯度累积：每accumulation_steps批次更新一次，或最后一个批次
                        # 索引从0开始
                        need_update_grad = (batch + 1) % gradient_accumulation_steps == 0 or batch == batch_count_per_file - 1
                    else:
                        # 无梯度累积，每个批次都更新
                        need_update_grad = True

                    # 跳过训练时的同步（解决恢复训练时的卡死问题）
                    if skipping_train:
                        # 协调多 GPU 间的同步：让所有进程（主进程 + 子进程）都到达这个wait调用点，才继续执行后续代码。
                        TrainerTools().parallel.wait('skip train')
                        skipping_train = False

                    # 从批次数据中获取输入和标签
                    input_ids = batch_data['input_ids']
                    labels = batch_data['labels']
                    attention_mask = batch_data.get('attention_mask')

                    try:
                        # 将数据移到目标设备
                        input_ids, labels = input_ids.to(TrainerTools().parallel.device), labels.to(TrainerTools().parallel.device)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(TrainerTools().parallel.device)
                        else:
                            # 掩码（pad位置为False）
                            attention_mask = input_ids != TrainerTools().tokenizer.pad

                        # 如果打包序列，创建文档边界掩码和位置ID
                        if self.packed_sequences:
                            doc_boundary_mask = create_doc_boundary_mask(input_ids, self._get_model_dtype())
                            position_ids = generate_position_ids(input_ids)
                        else:
                            doc_boundary_mask = None
                            position_ids = None

                        # 优先使用 dataset/collate_fn 传过来的 pixel_values
                        if 'pixel_values' in batch_data and batch_data['pixel_values'] is not None:
                            pixel_values = batch_data['pixel_values'].to(TrainerTools().parallel.device)
                        elif hasattr(self, 'pixel_values_provider') and self.pixel_values_provider and 'image_tags' in batch_data and batch_data['image_tags'][0] is not None:
                            image_tags = batch_data['image_tags']
                            pixel_values = self.pixel_values_provider(image_tags).to(TrainerTools().parallel.device)
                        else:
                            pixel_values = None

                        # 判断并行训练模式
                        if TrainerTools().parallel.parallel_train:
                            # 只有当gradient_accumulation_steps时，need_update_grad=True  没达到累计步数时，只需累积梯度，不更新参数，无需同步
                            # 当require_backward_grad_sync = True：反向传播时，各 GPU 会将计算出的梯度进行全局同步（all-reduce），得到所有 GPU 的平均梯度，用于后续参数更新；
                            self.train_model.require_backward_grad_sync = need_update_grad

                        # 根据设备类型自动选择合适的精度模式
                        with autocast(TrainerTools().parallel.device_type):
                            # 前向传播
                             # HuggingFace 模型的前向传播标准接口
                            # 它可以直接接收 input_ids, labels, pixel_values 等
                            
                            # 构建参数字典
                            forward_kwargs = {
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                                "labels": labels, # 传入 labels，模型会自动计算 loss
                                "use_cache": False
                            }
                            
                            # 如果是 VLM，传入 pixel_values
                            if 'pixel_values' in batch_data and batch_data['pixel_values'] is not None:
                                forward_kwargs['pixel_values'] = batch_data['pixel_values'].to(TrainerTools().parallel.device)
                                # Qwen2.5-VL还需要 image_grid_thw
                                if 'image_grid_thw' in batch_data:
                                     forward_kwargs['image_grid_thw'] = batch_data['image_grid_thw'].to(TrainerTools().parallel.device)

                            outputs = self.train_model(**forward_kwargs)
                            
                            # 获取 Loss
                            # 如果 DeepSpeed 包装了模型，outputs可能是 tuple 或 dict
                            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                        # 反向传播
                        self._backward_loss(loss)

                        # 累积损失值
                        loss_with_aux_accumulation += loss.detach().item()
                        loss_without_aux_accumulation += loss.detach().item()

                        # 累积批次计数
                        batches_accumulated += 1

                        # 如果需要更新梯度
                        if need_update_grad:
                            # 梯度裁剪
                            self._apply_grad_clipping()
                            # 执行优化步骤
                            self._apply_step()

                            # 计算平均损失
                            avg_loss = self._avg_loss(
                                losses=[loss_with_aux_accumulation],
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                batches_accumulated=batches_accumulated
                            )[0]

                            # 记录日志
                            self._log(
                                keys={
                                    'epoch': epoch,
                                    'file': f'{file_idx + 1}/{file_count}',
                                    'batch': f'{batch}/{batch_count_per_file}'
                                },
                                values={
                                    'loss': avg_loss
                                }
                            )

                            # 重置损失累积变量
                            loss_with_aux_accumulation = 0.0
                            loss_without_aux_accumulation = 0.0
                            batches_accumulated = 0
                    except Exception as e:
                        # 异常处理
                        self._on_exception(e, epoch, batch)
                    finally:
                        # 当batches_accumulated==gradient_accumulation_steps时，参数更新，梯度清零，此时need_update_grad=true
                        # 此时就可以保存步数和检查点
                        if need_update_grad:
                            # 每gradient_accumulation_steps步，保存步数、学习率
                            save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)

                            # 达到eval_batch_interval，保存检查点并评估
                            if (batch - last_ckpt_batch) >= self.train_config.eval_batch_interval:
                                # 存整个模型权重和优化器状态
                                save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                                last_ckpt_batch = batch
                                # 评估
                                self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')

                        # 清理loss变量（避免显存泄漏）
                        try:
                            del loss
                        except UnboundLocalError: ...
                        # 比如断点续训时候，跳过了train，此时loss变量是未定义的
                        # 静态处理，虽然会捕获异常，但是不中断程序运行

            # Epoch结束处理
            if not skipping_train:
                # 保存步数信息
                save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)
                # 保存检查点
                save_checkpoint(model=self.train_model, optimizer=self.optimizer)

                # 传入当前结束的 epoch 索引，通知底层的并行计算组件（特别是数据采样器），当前的 Epoch 已经结束了，准备进入下一个 Epoch
                # 更新分布式采样器的随机种子，确保下一个 Epoch 的数据会被重新打乱，防止模型过拟合数据的顺序
                TrainerTools().parallel.on_epoch_end(epoch)
                # Epoch结束：执行评估
                self._on_epoch_end(tag=f'epoch:{epoch}')

        # 销毁并行环境
        TrainerTools().parallel.destroy()
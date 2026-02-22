from typing import Tuple, List, Callable, Optional
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# 导入自定义训练器基类
from .trainer import Trainer
from .train_configs import TrainConfig
# 针对RL特定的数据集
from .dataset import RLDataset
from .loss import GRPOLoss
from .tools import TrainerTools
# 导入批量生成工具函数
from .generate_utils import batch_generate
from .log import log
# 导入工具函数（自动混合精度、序列填充、compute_token_losses、禁用dropout等）
from .utils import (
    autocast,
    left_pad_sequence,
    compute_token_losses,
    disable_dropout_in_model
)

# 导入模型参数同步、生成时模型解包工具
from .partition_utils import (
    sync_model_params,
    unwrap_model_for_generation
)

# 导入 checkpoint 保存工具
from .checkpoint import (
    save_checkpoint,
    save_steps,
)

# 定义GRPOTrainer类，继承自基础Trainer类
class GRPOTrainer(Trainer):
    """
        奖励函数：输入提示ID、补全ID、答案ID，返回奖励分数列表
    """
    def __init__(
            self,
            *,
            train_config: TrainConfig,  # 训练配置
            reward_func: Callable[[List[torch.Tensor], torch.Tensor, List[Optional[torch.Tensor]]], List[float]],  # 奖励函数
            eval_prompts: List[str],  # 评估用的提示词列表
            eval_image_tags: Optional[List[str]] = None  # 评估用的图像标签（可选）
    ):
        # 调用父类Trainer的初始化方法
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            eval_image_tags=eval_image_tags
        )

        # 是否使用打包序列（将多个短样本拼接在一起）
        self.packed_sequences = False
        # 奖励函数
        self.reward_func = reward_func
        # 初始化参考模型（ref_model）
        self.ref_model = self._init_ref_model()


    # 创建新模型（重写父类方法）
    def _new_model(self, train_config: TrainConfig):
        # 调用父类方法：传入训练配置，创建模型
        model = super()._new_model(train_config)
        # GRPO训练中通常固定dropout以稳定训练，所以禁用模型中的dropout层
        disable_dropout_in_model(model)
        return model
    
    # 转换训练参数为并行配置、数据加载器配置、采样器配置（重写父类方法）
    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        # 并行参数        数据加载器参数         采样器参数
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()
        # 更新数据加载器的collate_fn（数据拼接函数）为直接返回列表（不填充）
        data_loader_kwargs.update({"collate_fn": lambda x: x})
        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    # 创建数据集（重写父类方法）
    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        # 文件索引——————>数据集文件路径
        file_path = self.train_config.file_dataset[file_idx]
        max_seq_len = self.train_config.max_seq_len
        # 将数据转换成张量并返回
        return RLDataset(file_path, self.processor, max_seq_len), file_path

    # 初始化参考模型（用于计算KL散度）
    def _init_ref_model(self):
        # 如果KL惩罚系数beta为0，则不需要参考模型
        if self.train_config.grpo_config.loss_beta == 0.0:
            return None

        # 创建参考模型（参数初始化为训练配置）
        ref_model = self._new_model(self.train_config)

        # 对参考模型进行并行化处理（多GPU分布）
        ref_model, _ = TrainerTools().parallel.process(
            model=ref_model,
            optimizer=None,  # 冻结：不需要优化器
            kwargs=self._init_ref_model_args(),  # 初始化参考模型的参数
            save_instance=False,  # 参考模型参数冻结，因此无需保存模型实例的状态（如参数、优化器状态等）
        )

        # 评估模式（关闭dropout等训练特有的层）
        ref_model.eval()
        # 冻结参考模型参数
        for param in ref_model.parameters():
            param.requires_grad = False

        return ref_model


    # 损失函数
    def _init_loss(self):
        criterion = GRPOLoss(
            beta=self.train_config.grpo_config.loss_beta,  # KL惩罚系数
            clip_eps_low=self.train_config.grpo_config.loss_clip_eps,  # 下界裁剪系数
            clip_eps_high=self.train_config.grpo_config.loss_clip_eps_high,  # 上界裁剪系数
            delta=self.train_config.grpo_config.loss_delta,  # 优势截断参数
            importance_sampling_level=self.train_config.grpo_config.loss_importance_sampling_level,  # 重要性采样级别
            loss_type=self.train_config.grpo_config.loss_type,  # 损失类型（如"kl"或"js"）
            gen_max_new_tokens=self.train_config.grpo_config.gen_max_new_tokens  # 生成时的最大新token数
        )

        # 返回损失函数和None（GRPO不需要额外的辅助损失计算器）
        return criterion, None


    # 计算模型生成序列的log probability
    def _compute_log_probs(
            self,
            model,
            input_ids,
            attention_mask,
            **kwargs  # [新增] 接收 pixel_values 等额外参数
    ):
        # 模型前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs # [新增] 透传 pixel_values 和 image_grid_thw
        )
        
        logits = outputs['logits'][:, :-1, :]
        input_ids = input_ids[:, 1:]

        return compute_token_losses(logits, input_ids), outputs.get('aux_loss')

    # 计算组内相对优势
    def _compute_group_relative_advantages(self, rewards):       
        # 每组内的样本数
        group_size = self.train_config.grpo_config.group_size

        # 将奖励按组重塑 rewards[batch*group_size] -> [batch, group_size]
        rewards_by_group = rewards.view(-1, group_size)

        # 计算每组奖励的均值和标准差
        # group_means shape: [batch]，group_stds shape: [batch]
        group_means = rewards_by_group.mean(dim=1)
        group_stds = rewards_by_group.std(dim=1)

        # 广播：使用PyTorch的repeat_interleave将均值和标准差扩展到与原始rewards形状一致 [batch] -> [batch*group_size]
        expanded_means = group_means.repeat_interleave(group_size)
        expanded_stds = group_stds.repeat_interleave(group_size)

        # 标准化奖励得到优势（(奖励 - 组均值) / (组标准差 + 小epsilon)）
        # advantages shape: [batch*group_size]
        advantages = (rewards - expanded_means) / (expanded_stds + 1e-4)

        # 将seq_len维度增加至token级操作，因为前面对数概率是基于token计算的
        # [batch*group_size] -> [batch*group_size, 1]
        return advantages.unsqueeze(1)

    # 基于提示词生成响应
    def _generate_completions(self, model, prompts, group_size: int, pixel_values=None, image_grid_thw=None):
        pad_token_id = TrainerTools().tokenizer.pad
        device = TrainerTools().parallel.device

        # 1. 对提示词序列进行左填充 (Tensor)
        prompt_ids = left_pad_sequence(prompts, padding_value=pad_token_id)
        prompt_ids = prompt_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        # 2. 广播提示词 (Batch Size -> Batch Size * Group Size)
        # 例如: [A, B] -> [A, A, A, A, B, B, B, B] (当 group_size=4)
        prompt_ids = prompt_ids.repeat_interleave(group_size, 0)
        prompt_masks = prompt_ids != pad_token_id
        
        # 3. [关键修正] 广播视觉特征
        # 既然 Prompt 重复了，图片也必须重复对应的次数，才能和 input_ids 一一对应
        expanded_pixel_values = None
        expanded_image_grid_thw = None

        if pixel_values is not None:
            # 假设 pixel_values 维度是 [N, C, H, W] 或 [Total_Images, ...]
            # 我们需要让它变成 [N*G, ...]
            expanded_pixel_values = pixel_values.repeat_interleave(group_size, dim=0)
            
        if image_grid_thw is not None:
            expanded_image_grid_thw = image_grid_thw.repeat_interleave(group_size, dim=0)
        
        # 4. 批量生成
        outputs, _ = batch_generate(
            model=model,
            tokens=prompt_ids,
            attention_mask=prompt_masks,
            max_new_tokens=self.train_config.grpo_config.gen_max_new_tokens,
            temperature=self.train_config.grpo_config.gen_temperature,
            k=self.train_config.grpo_config.gen_k,
            p=self.train_config.grpo_config.gen_p,
            device=device,
            suppress_tokens=self.train_config.grpo_config.gen_suppress_tokens,
            # 传入扩展后的视觉参数
            pixel_values=expanded_pixel_values,
            image_grid_thw=expanded_image_grid_thw 
        )

        # 5. 提取生成结果
        completion_ids = outputs[:, prompt_len:]
        completion_masks = (completion_ids != pad_token_id).int()

        return prompt_ids, prompt_masks, completion_ids, completion_masks

    # 采样阶段
    def _generate_rollout_data(self, generate_model, batch_data: List[dict]):
        # ==================== 1. 准备数据 ====================
        prompts = [item["input_ids"] for item in batch_data]
        answers = [item["answer"] for item in batch_data]

        pixel_values = []
        image_grid_thw = []
        for item in batch_data:
            if "pixel_values" in item: pixel_values.append(item["pixel_values"])
            if "image_grid_thw" in item: image_grid_thw.append(item["image_grid_thw"])

        batch_pixel_values = None
        batch_image_grid_thw = None
        if len(pixel_values) > 0:
            batch_pixel_values = torch.cat(pixel_values, dim=0).to(generate_model.device)
        if len(image_grid_thw) > 0:
            batch_image_grid_thw = torch.cat(image_grid_thw, dim=0).to(generate_model.device)

        group_size = self.train_config.grpo_config.group_size

        # ==================== 2. 环境清理与切换 ====================
        # 切换到 eval 模式
        was_training = generate_model.training
        generate_model.eval()
        
        # 强制清理显存
        import gc
        torch.cuda.empty_cache()
        gc.collect()

        # ==================== 3. 生成 (Inference Mode) ====================
        with torch.no_grad():
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_completions(
                generate_model,
                prompts,
                group_size,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_image_grid_thw
            )

            # 拼接
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

            # 广播视觉特征
            expanded_pixel_values = None
            expanded_image_grid_thw = None
            if batch_pixel_values is not None:
                expanded_pixel_values = batch_pixel_values.repeat_interleave(group_size, dim=0)
            if batch_image_grid_thw is not None:
                expanded_image_grid_thw = batch_image_grid_thw.repeat_interleave(group_size, dim=0)

            # 计算概率
            old_log_probs, _ = self._compute_log_probs(
                generate_model, 
                input_ids, 
                attention_mask,
                pixel_values=expanded_pixel_values,
                image_grid_thw=expanded_image_grid_thw
            )

            if self.ref_model:
                ref_log_probs, _ = self._compute_log_probs(
                    self.ref_model, 
                    input_ids, 
                    attention_mask,
                    pixel_values=expanded_pixel_values,
                    image_grid_thw=expanded_image_grid_thw
                )
            else:
                ref_log_probs = None


        # ==================== 4. 清理缓存并恢复训练模式 ====================
        # 清理 RoPE 缓存（通用优化，防止缓存冲突）
        if was_training:
            try:
                def clear_rotary_cache(module):
                    if hasattr(module, 'cos_cached'): module.cos_cached = None
                    if hasattr(module, 'sin_cached'): module.sin_cached = None
                    for child in module.children():
                        clear_rotary_cache(child)
                clear_rotary_cache(generate_model)
            except:
                pass
            
            # 恢复训练模式
            generate_model.train()

        # ==================== 5. 返回数据 (深度 Detach) ====================
        safe_pixel_values = expanded_pixel_values.clone().detach() if expanded_pixel_values is not None else None
        safe_image_grid_thw = expanded_image_grid_thw.clone().detach() if expanded_image_grid_thw is not None else None
        safe_ref_log_probs = ref_log_probs.clone().detach() if ref_log_probs is not None else None

        repeated_prompts = [p for p in prompts for _ in range(group_size)]
        repeated_answers = [a for a in answers for _ in range(group_size)]

        return {
            'input_ids': input_ids.clone().detach(),
            'attention_mask': attention_mask.clone().detach(),
            'completion_mask': completion_mask.clone().detach(),
            'old_log_probs': old_log_probs.clone().detach(),
            'ref_log_probs': safe_ref_log_probs,
            'completion_ids': completion_ids.clone().detach(),
            'pixel_values': safe_pixel_values,
            'image_grid_thw': safe_image_grid_thw,
            'repeated_prompts': repeated_prompts,
            'repeated_answers': repeated_answers,
        }

    # 最大化GRPO目标函数（计算损失并返回）
    def _maximize_grpo_objective(self, rollout_data):
        device = TrainerTools().parallel.device

        # 从rollout数据中解包变量
        input_ids = rollout_data['input_ids']
        attention_mask = rollout_data['attention_mask']
        completion_mask = rollout_data['completion_mask']
        old_log_probs = rollout_data['old_log_probs']
        ref_log_probs = rollout_data['ref_log_probs']
        completion_ids = rollout_data['completion_ids']
        repeated_prompts = rollout_data['repeated_prompts']
        repeated_answers = rollout_data['repeated_answers']
        pixel_values = rollout_data.get('pixel_values')
        image_grid_thw = rollout_data.get('image_grid_thw')

        # 计算提示词长度（完整输入长度 - 回答长度）
        prompt_len = input_ids.shape[1] - completion_ids.shape[1]

        # 调用奖励函数：计算每个回答的奖励(与repeated_answers参考答案对比)
        # rewards: [batch*group_size]
        rewards = torch.tensor(
            self.reward_func(repeated_prompts, completion_ids, repeated_answers),
            dtype=torch.float32,
            device=device
        )

        # 计算组内相对优势 advantages[batch*group_size, 1]
        advantages = self._compute_group_relative_advantages(rewards)

        # 计算当前训练模型的对数概率（新策略概率）和辅助损失
        log_probs, aux_loss = self._compute_log_probs(
            self.train_model, 
            input_ids, 
            attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )

        # 构造掩码结构：“提示词的位置”（屏蔽，不计算损失）+ “补全序列的位置”（有效，计算损失）
        pad_len = prompt_len - 1  # 提示词部分长度（log_probs是token级的对数概率，长度为“输入序列总长度-1”）
        if pad_len > 0:
            # 左填充pad_len个0（对齐logits提示词部分）
            # pad=(左侧填充长度, 右侧填充长度)
            padded_completion_mask = F.pad(completion_mask, pad=(pad_len, 0), mode='constant', value=0)
        else:
            padded_completion_mask = completion_mask

        # 断言：掩码与log_probs形状一致（确保维度匹配）
        assert padded_completion_mask.shape == log_probs.shape, \
            f"padded_completion_mask与log_probs维度不匹配! padded_completion_mask: {padded_completion_mask.shape}, log_probs: {log_probs.shape}"

        # 调用GRPO损失函数计算损失
        loss = self.criterion(
            log_probs=log_probs,  # 新策略对数概率
            old_log_probs=old_log_probs,  # 旧策略对数概率
            ref_log_probs=ref_log_probs,  # 参考模型对数概率
            completion_mask=padded_completion_mask,  # 对齐后的掩码（用于筛选有效token）
            advantages=advantages  # 组内相对优势
        )

        return loss, aux_loss, rewards

    # 训练主函数
    def train(self):
        # 初始化全局步数
        global_steps = 0
        # 用于断点续训时跳过已训练的步骤
        skipping_train = False
        # 辅助损失系数
        aux_loss_coef = self.train_config.loss_config.aux_loss_coef

        # 遍历训练轮次（epoch）
        for epoch in range(self.train_config.n_epochs):
            # 获取训练文件总数
            file_count = len(self.train_config.file_dataset)

            # 遍历每个训练文件（按文件分批训练）
            for file_idx in range(file_count):
                # 创建当前文件对应的数据集和文件路径
                dataset, file_path = self._create_dataset(file_idx)

                # 创建数据加载器（支持并行处理）
                train_data_loader = TrainerTools().parallel.process_dataloader(
                    dataset=dataset,
                    data_loader_kwargs=self.data_loader_kwargs,
                    sampler_kwargs=self.sampler_kwargs
                )

                # 上一次保存checkpoint的批次索引
                last_ckpt_batch = 0
                # 当前文件的总批次数
                batch_count_per_file = len(train_data_loader)

                # 通知并行处理器开始当前epoch
                TrainerTools().parallel.on_epoch_start(epoch)
                # 通知开始处理当前文件
                self._on_file_start(epoch, file_path)

                # 遍历数据加载器中的批次
                for batch, batch_data in enumerate(train_data_loader):
                    # 全局步数+1
                    global_steps += 1
                    # 如果当前步数小于已训练的步数（断点续训），跳过当前批次
                    if global_steps < self.last_global_steps:
                        skipping_train = True
                        continue

                    # 上面逻辑if如果不执行，说明当前步数大于已训练步数，恢复训练（等待其他进程同步）
                    if skipping_train:
                        TrainerTools().parallel.wait('skip train')# 不同进程的 “跳过已训练批次” 速度可能不一致（比如进程 A 先跳完，进程 B 还在跳），直到所有进程都完成了 “跳过已训练批次” 的操作，再一起进入正式训练
                        skipping_train = False

                    # 主进程打印生成开始日志
                    if TrainerTools().parallel.is_main_process:
                        log(f'开始生成，批次： {batch}/{batch_count_per_file}')

                    # 生成rollout数据：解包模型（去除并行包装）用于生成
                    with unwrap_model_for_generation(self.train_model) as generate_model:
                        rollout_data = self._generate_rollout_data(generate_model, batch_data)
                    # 生成结束

                    # 清空CUDA缓存，节省内存
                    torch.cuda.empty_cache()

                    try:
                        # 主进程打印训练开始日志
                        if TrainerTools().parallel.is_main_process:
                            log(f'开始训练，批次： {batch}/{batch_count_per_file}')

                        # 执行多步GRPO更新（按配置中的grpo_steps）
                        for grpo_step in range(self.train_config.grpo_config.grpo_steps):
                            # 启用自动混合精度训练（节省内存，加速训练）
                            with autocast(TrainerTools().parallel.device_type):
                                # 计算GRPO损失、辅助损失和奖励
                                loss, aux_loss, rewards = self._maximize_grpo_objective(rollout_data)
                                # 如果辅助损失系数不为0且存在辅助损失，加权辅助损失
                                if aux_loss_coef and aux_loss is not None:
                                    aux_loss = aux_loss_coef * aux_loss
                                else:
                                    aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

                            # 总损失 = GRPO损失 + 加权辅助损失
                            total_loss = loss + aux_loss
                            # 反向传播计算梯度
                            self._backward_loss(total_loss)
                            # 梯度裁剪（防止梯度爆炸）
                            self._apply_grad_clipping()
                            # 更新模型参数
                            self._apply_step()

                            # 记录损失值（ detach() 脱离计算图，无梯度信息不会影响原计算图的梯度传播）
                            loss_with_aux_accumulation = total_loss.detach().item()
                            loss_without_aux_accumulation = loss.detach().item()
                            aux_loss_accumulation = aux_loss.detach().item()

                            # 计算平均损失
                            # gradient_accumulation_steps（梯度累积步数）效果近似直接用 batch size=batch_size*gradient_accumulation_steps
                            # batches_accumulated（当前已经累积的批次数量）
                            avg_loss, avg_loss_without_aux, avg_aux_loss = self._avg_loss(
                                losses=[
                                    loss_with_aux_accumulation,
                                    loss_without_aux_accumulation,
                                    aux_loss_accumulation
                                ],
                                gradient_accumulation_steps=1,
                                batches_accumulated=1
                            )

                            # 记录日志（包含epoch、文件索引、批次、GRPO步骤等信息）
                            self._log(
                                keys={
                                    'epoch': epoch,
                                    'file': f'{file_idx + 1}/{file_count}',
                                    'batch': f'{batch}/{batch_count_per_file}',
                                    'grpo_step': grpo_step
                                },
                                values={
                                    'loss(with aux)': avg_loss,
                                    'loss(without aux)': avg_loss_without_aux,
                                    'aux_loss': avg_aux_loss,
                                    'rewards': (rewards.sum() / rewards.size(0)).item(),  # 平均奖励
                                }
                            )
                    except Exception as e:
                        # 处理训练中的异常
                        self._on_exception(e, epoch, batch)
                    finally:
                        # 保存当前步数和学习率调度器状态
                        save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)

                        # 是否达到了 “保存该检查点” 的时机（如果当前批次与上次保存批次的差 >= 训练配置中预设的 “检查点保存间隔”）
                        if (batch - last_ckpt_batch) >= self.train_config.eval_batch_interval:
                            save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                            last_ckpt_batch = batch
                            # 通知批次结束
                            self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')

                        # 清理loss变量（防止内存泄漏）
                        try:
                            del loss
                        except UnboundLocalError: pass   # 若loss未定义，触发变量未定义异常，防止程序崩溃，但继续执行

            # 当前是否是正常训练，如果是则本轮epoch结束
            if not skipping_train:
                # 保存当前步数和学习率调度器
                save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)
                # 保存checkpoint
                # save_checkpoint(model=self.train_model, optimizer=self.optimizer)   # 是否每个epoch结束保存checkpoint，适合大数据集

                # 通知并行处理器本轮epoch结束
                TrainerTools().parallel.on_epoch_end(epoch)
                # 通知epoch结束
                self._on_epoch_end(tag=f'epoch:{epoch}')
        
        save_checkpoint(model=self.train_model, optimizer=self.optimizer)# 在所有Epoch都结束时才保存checkpoint，适合小数据集
        # 销毁并行处理器
        TrainerTools().parallel.destroy()
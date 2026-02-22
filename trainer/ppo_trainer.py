from typing import Tuple, List, Union, Callable, Optional
import torch
from torch.utils.data import Dataset
import torch.nn as nn

from transformers import AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration

from .trainer import Trainer
from .train_configs import TrainConfig
from .dataset import RLDataset
from .loss import PPOLoss
from .tools import TrainerTools
from .generate_utils import batch_generate
from .utils import (
    autocast,  # 自动混合精度上下文管理器
    left_pad_sequence,  # 序列左填充函数
    compute_token_losses,  # 计算每个token的log probability
    masked_whiten,  # 带掩码的标准化
    disable_dropout_in_model  # 禁用模型中的dropout
)
from .partition_utils import unwrap_model_for_generation  # 解包模型以进行生成
from .log import log
from .checkpoint import (
    save_checkpoint,  # 保存检查点
    save_steps,  # 保存训练步数
)


class ValueModel(nn.Module):
    """
    价值模型（Critic），用于评估状态的价值
    """
    def __init__(self, base_model: Union[AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration]):
        super().__init__()
        # 基础模型（通常是LLM或VLM）
        self.base_model = base_model
        # 价值头：将隐藏层状态映射到标量价值
        self.value_head = nn.Linear(base_model.config.hidden_size, 1, bias=True)
        # 初始化价值头权重：正态分布
        self.value_head.weight.data.normal_(mean=0.0, std=0.01)
        # 初始化价值头偏置：零初始化
        self.value_head.bias.data.zero_()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # 基础模型前向传播
        outputs = self.base_model(*args, **kwargs)
        # 获取最后一层的隐藏状态
        # [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs['hidden_states']
        # 通过价值头计算价值
        # [batch_size, seq_len, 1]
        values = self.value_head(last_hidden_state)
        # 移除最后一个维度
        # [batch_size, seq_len]
        return values.squeeze(-1)


class PolicyAndValueModelWrapper(nn.Module):
    """
    策略模型和价值模型的包装器，方便统一管理
    """
    def __init__(self, policy_model: nn.Module, value_model: nn.Module):
        super().__init__()
        # 策略模型（Actor）
        self.policy_model = policy_model
        # 价值模型（Critic）
        self.value_model = value_model

    def forward(self, *args, **kwargs):
        # 同时返回策略模型和价值模型的输出
        return self.policy_model(*args, **kwargs), self.value_model(*args, **kwargs)


class PPOTrainer(Trainer):
    """
    PPO训练器
    reward_func(prompt_ids, complete_ids, answer_ids) -> scores
    """

    def __init__(
            self,
            *,
            train_config: TrainConfig,  # 训练配置
            reward_func: Callable[[List[torch.Tensor], torch.Tensor, List[Optional[torch.Tensor]]], List[float]],  # 奖励函数
            eval_prompts: List[str],  # 评估用的提示词
            eval_image_tags: Optional[List[str]] = None  # [可选]评估图像标签
    ):
        # 调用父类初始化
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            eval_image_tags=eval_image_tags
        )
        # PPO不使用序列打包
        self.packed_sequences = False
        # 保存奖励函数
        self.reward_func = reward_func

        # 初始化参考模型
        self.ref_model = self._init_ref_model()

    def _init_train_model_and_optim(self, initial_lr: float):
        """
        初始化训练模型（Actor和Critic）及优化器
        """
        # 创建策略模型（Actor）
        policy_model = self._new_model(self.train_config)
        # 创建价值模型（Critic），基于一个新的基础模型
        value_model = ValueModel(self._new_model(self.train_config))
        # 将两者包装在一起
        train_model = PolicyAndValueModelWrapper(policy_model, value_model)

        # 如果指定了初始化状态字典（通常用于从SFT模型继续训练）
        if self.train_config.init_state_dict:
            # 加载策略模型权重
            policy_model.load_state_dict(self.train_config.init_state_dict)
            # 加载价值模型的基础模型权重
            value_model.base_model.load_state_dict(self.train_config.init_state_dict)
            # 清空配置中的状态字典以释放内存
            self.train_config.init_state_dict = None

        # 如果指定了价值模型的检查点（用于恢复训练价值模型）
        if self.train_config.ppo_config.value_model_checkpoint:
            # 加载价值模型权重
            value_model.load_state_dict(self.train_config.ppo_config.value_model_checkpoint)
            # 清空配置中的检查点数据
            self.train_config.ppo_config.value_model_checkpoint = {}

        # 在主进程中打印模型参数信息
        if TrainerTools().parallel.is_main_process:
            for name, model in zip(['policy', 'value'], [policy_model, value_model]):
                # 计算总参数量
                total_params = sum(p.numel() for p in model.parameters())
                log(f"Total number of {name} model parameters: {total_params:,}")

                # 计算可训练参数量
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                log(f"Trainable number of {name} model parameters: {trainable_params:,}")

                # 计算模型大小（MB）
                total_size_bytes = total_params * 4
                total_size_mb = total_size_bytes / (1024 * 1024)
                log(f"Total size of {name} model model: {total_size_mb:.2f} MB")

        # 使用并行工具处理模型和优化器
        model, optim = TrainerTools().parallel.process(
            model=train_model,
            optimizer=self._config_optim(train_model, initial_lr),  # 配置优化器
            kwargs=self.parallel_kwargs  # 并行参数
        )

        return model, optim

    def _init_ref_model(self):
        """
        初始化参考模型，用于计算KL散度
        """
        ref_model = self._new_model(self.train_config)

        # 如果指定了参考模型检查点
        if self.train_config.ppo_config.ref_model_checkpoint:
            # 加载权重
            ref_model.load_state_dict(self.train_config.ppo_config.ref_model_checkpoint)
            # 清空配置，防止内存占用
            self.train_config.ppo_config.ref_model_checkpoint = {}

        # 使用并行工具处理参考模型（不需要优化器）
        ref_model, _ = TrainerTools().parallel.process(
            model=ref_model,
            optimizer=None,
            kwargs=self._init_ref_model_args(),
            save_instance=False  # 参考模型是冻结的，不参与训练，其状态不会改变，因此不需要TrainerTools进行保存或管理
        )

        # 设置为评估模式
        ref_model.eval()
        # 冻结所有参数
        for param in ref_model.parameters():
            param.requires_grad = False

        return ref_model

    def _new_model(self, train_config: TrainConfig):
        """
        创建新模型实例（重写父类方法）
        """
        # 调用父类方法创建模型
        model = super()._new_model(train_config)
        # 禁用模型中的dropout（dropout的随机性会与 PPO 训练对策略稳定性、输出一致性的需求冲突）
        disable_dropout_in_model(model)
        return model

    def _init_loss(self):
        """
        初始化损失函数
        """
        ppo_config = self.train_config.ppo_config
        # 创建PPO损失函数
        criterion = PPOLoss(
            clip_eps=ppo_config.clip_eps,  # PPO裁剪参数epsilon
            vf_coef=ppo_config.vf_coef  # 价值函数损失系数
        )
        # 返回损失函数（PPO不需要KD loss，所以第二个返回值为None）
        return criterion, None

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        """
        转换训练参数（重写父类方法）
        """
        # 获取父类的参数
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()
        # 更新collate_fn，PPO的数据整理比较简单，直接返回list即可，后续在_generate_rollout_data中处理
        data_loader_kwargs.update({"collate_fn": lambda x: x})
        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        """
        创建数据集
        """
        file_path = self.train_config.file_dataset[file_idx]
        max_seq_len = self.train_config.max_seq_len
        # 使用RLDataset
        return RLDataset(file_path, self.processor, max_seq_len), file_path

    def _calc_loss(self, inputs, attention_mask, logits, labels):
        """
        计算损失（占位，PPO使用自定义的_ppo_learning_phase）
        """
        ...

    def _check_eval_model(self, eval_model):
        """
        返回策略模型用于评估
        """
        return eval_model.policy_model

    def _compute_advantages_and_returns(
            self,
            rewards: torch.Tensor,
            values: torch.Tensor,
            last_values: torch.Tensor,
            completion_mask: torch.Tensor,
            dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用GAE计算优势（Advantages）和回报（Returns）
        """
        # 获取折扣因子gamma和GAE参数lambda
        gamma, lam = self.train_config.ppo_config.gamma, self.train_config.ppo_config.lam
        # 存储反向计算的优势值列表
        advantages_reversed = []
        # 初始化上一时刻的GAE值（对于最后一步为0）
        last_gae_lam = 0
        # 获取序列长度
        seq_len = rewards.size(1)

        # 有效token
        values = values * completion_mask
        # 逆序遍历时间步：从后向前递归计算（因为当前的优势取决于未来的收益）
        for t in reversed(range(seq_len)):
            # 如果是最后一步
            if t == seq_len - 1:
                # 如果 dones=True（生成结束了，比如遇到了 EOS），那么未来的价值是 0
                # 如果 dones=False（生成被截断了，达到最大长度但没说完），我们用 last_values 来作为未来价值的近似
                next_values = torch.where(dones, 0.0, last_values)
            # 普通情况：直接取 values 张量里下一个时间步的值
            else:
                next_values = values[:, t + 1]

            # 计算TD误差 delta = r + gamma * V(s') - V(s)
            delta = rewards[:, t] + gamma * next_values - values[:, t]
            # 计算GAE: A_t = delta_t + (gamma * lambda) * A_{t+1}
            # Padding 的部分（无效 Token），mask 为 0，这会切断优势的传播
            last_gae_lam = delta + gamma * lam * last_gae_lam * completion_mask[:, t]
            advantages_reversed.append(last_gae_lam)

        # 列表里的顺序是[AT,AT−1,...,A 0]  我们需要用 [::-1] 把它变回正常的时间顺序 [A0,...,AT]
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # 计算回报 Returns = Advantages + Values
        returns = advantages + values

        # 应用掩码并返回
        return advantages * completion_mask, returns * completion_mask

    def _generate_rollout_data(self, batch_data: List[dict]) -> dict:
        """
        生成Rollout数据（采样阶段）
        """
        ppo_config = self.train_config.ppo_config
        device = TrainerTools().parallel.device
        pad_token_id = TrainerTools().tokenizer.pad
        eos_token_id = TrainerTools().tokenizer.end

        # 提取prompt和answer
        prompts = [item["prompt"] for item in batch_data]
        answers = [item["answer"] for item in batch_data]

        # 对prompt进行左填充
        prompt_ids = left_pad_sequence(prompts, padding_value=pad_token_id).to(device)
        # 生成prompt掩码（padding部分为0，有效token为1）
        prompt_masks = (prompt_ids != pad_token_id)
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            # 解包模型以进行生成（处理并行包装）
            with unwrap_model_for_generation(self.train_model) as unwrapped_model:
                # 批量生成回复
                full_ids, logitss = batch_generate(
                    model=unwrapped_model.policy_model,  # 使用策略模型生成
                    tokens=prompt_ids,
                    attention_mask=prompt_masks,
                    max_new_tokens=ppo_config.gen_max_new_tokens,
                    temperature=ppo_config.gen_temperature,
                    k=ppo_config.gen_k,
                    p=ppo_config.gen_p,
                    suppress_tokens=ppo_config.gen_suppress_tokens,
                    device=device
                )
                # 完整的attention mask
                full_attention_mask = (full_ids != pad_token_id)
                # 回复部分
                completion_ids = full_ids[:, prompt_len:]
                

                # 价值模型：
                # 输入：prompt+回复
                # 输出：value
                with autocast(TrainerTools().parallel.device_type):
                    value_output = unwrapped_model.value_model(full_ids, attention_mask=full_attention_mask)

            # 对数概率：旧策略的回复
            # logitss 只包含新生成的 token 的 logits （batch_generate函数不输出prompt部分的logits）
            old_log_probs = compute_token_losses(logitss.float(), completion_ids)

            # 参考模型：
            # 输入：prompt+回复
            # 输出：logits
            with unwrap_model_for_generation(self.ref_model) as unwrapped_ref_model:
                ref_outputs = unwrapped_ref_model(full_ids, attention_mask=full_attention_mask)
                ref_logits_full = ref_outputs['logits']

            # 参考模型：回复部分的logits
            # 自回归模型有移位
            ref_logits_completion = ref_logits_full[:, prompt_len - 1: -1]
            
            # 对数概率：参考模型的回复与策略模型的回复
            ref_log_probs_completion = compute_token_losses(ref_logits_completion.float(), completion_ids)

            # 判断是否结束（遇到EOS）
            dones = torch.any(completion_ids == eos_token_id, dim=1)
            # 初始化奖励张量
            rewards = torch.zeros_like(completion_ids, dtype=torch.float32, device=device)
            # 回复掩码
            completion_mask = (completion_ids != pad_token_id)

            # 计算KL散度奖励
            if ppo_config.kl_beta > 0.0:
                # 1. 计算对数概率之差
                logr = ref_log_probs_completion - old_log_probs
                # 2. 计算KL散度近似值
                # k1 估计器：log(pi/ref) = log(pi) - log(ref)
                # k2/k3 估计器：更精确的近似
                kl = -logr if ppo_config.kl_estimator == "k1" else (logr.exp() - 1) - logr
                # 3. 计算KL惩罚
                # “负奖励”：如果 KL 越大（偏离越远），扣分越多
                kl_rewards = -ppo_config.kl_beta * kl
                # 累加到奖励中：惩罚加在生成的每一个有效 Token 上
                rewards += kl_rewards * completion_mask

            # 计算环境奖励（使用自定义奖励函数）
            env_rewards_tensor = torch.tensor(
                self.reward_func(prompts, completion_ids, answers),
                dtype=torch.float32,
                device=device
            )

            # 将环境奖励加到每个序列的最后一个有效token上
            last_token_indices = completion_mask.sum(dim=1) - 1
            valid_indices_mask = last_token_indices >= 0

            if valid_indices_mask.any():
                valid_batch_indices = torch.arange(prompt_ids.size(0), device=device)[valid_indices_mask]
                valid_last_token_indices = last_token_indices[valid_indices_mask]
                valid_env_rewards = env_rewards_tensor[valid_indices_mask]
                rewards[valid_batch_indices, valid_last_token_indices] += valid_env_rewards

        # 返回Rollout数据字典
        return {
            'prompt_ids': prompt_ids.detach(),
            'completion_ids': completion_ids.detach(),
            'old_log_probs': old_log_probs.detach(),
            'values': value_output.detach(),
            'rewards': rewards.detach(),
            'env_rewards': env_rewards_tensor.detach(),
            'dones': dones.detach(),
        }

    def _ppo_learning_phase(self, rollout_data: dict):
        """
        PPO学习阶段
        """
        ppo_config = self.train_config.ppo_config

        # 从旧策略中提取各项
        prompt_ids: torch.Tensor = rollout_data['prompt_ids']
        completion_ids: torch.Tensor = rollout_data['completion_ids']
        old_log_probs: torch.Tensor = rollout_data['old_log_probs']
        old_values: torch.Tensor = rollout_data['values']
        rewards: torch.Tensor = rollout_data['rewards']
        dones: torch.Tensor = rollout_data['dones']

        # 获取维度信息
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        
        # 提取用于GAE计算的价值（对应生成部分的每个token）
        values_for_gae = old_values[:, prompt_len - 1: -1]
        # 最后一个时间步的价值
        last_values = old_values[:, -1]
        # 确保维度一致   即  回复的句子长度
        assert values_for_gae.shape[1] == completion_ids.shape[1]

        # 回复部分的掩码
        completion_mask: torch.Tensor = (completion_ids != TrainerTools().tokenizer.pad)

        # 如果配置了奖励标准化：只做缩放（除以标准差），而不减去均值，忽略 Padding，只统计有效的 Token
        if ppo_config.whiten_rewards:
            rewards = masked_whiten(rewards, completion_mask, shift_mean=False)

        # 计算优势函数和回报
        advantages, returns = self._compute_advantages_and_returns(
            rewards, values_for_gae, last_values, completion_mask, dones
        )

        # 对优势函数进行标准化
        # 如果 A_t 很大，梯度就会很大，导致策略更新过猛，破坏原有的策略。
        # 如果 A_t 很小，更新就会非常缓慢。 标准化将 A_t 强制拉回到均值为 0、方差为 1 的分布，确保了每次更新的幅度是可控且一致的
        advantages_whitened = masked_whiten(advantages, completion_mask, shift_mean=True)

        # 拼接完整的输入IDs
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        # 生成完整的attention mask
        attention_mask = (input_ids != TrainerTools().tokenizer.pad)

        # 初始化统计信息字典
        ppo_stats = {
            "loss_with_aux": 0, "loss_without_aux": 0, "aux_loss": 0,
            "actor_loss": 0, "value_loss": 0, "approx_kl": 0, "clip_frac": 0
        }

        # Mini-batch_size
        ppo_batch_size = ppo_config.ppo_batch_size
        n_updates = 0
        # Epoch循环
        for ppo_epoch in range(ppo_config.ppo_epochs):
            # 生成随机索引以打乱数据
            indices = torch.randperm(batch_size, device=TrainerTools().parallel.device)

            # Mini-batch循环
            # range(start, stop, step)
            for i in range(0, batch_size, ppo_batch_size):
                # 获取当前mini-batch的索引
                mini_batch_indices = indices[i:i + ppo_batch_size]

                # 根据索引提取mini-batch旧策略数据
                mb_input_ids = input_ids[mini_batch_indices]
                mb_attention_mask = attention_mask[mini_batch_indices]
                mb_completion_ids = completion_ids[mini_batch_indices]
                mb_completion_mask = completion_mask[mini_batch_indices]
                mb_old_log_probs = old_log_probs[mini_batch_indices]
                mb_values = values_for_gae[mini_batch_indices]
                mb_returns = returns[mini_batch_indices]
                mb_advantages = advantages_whitened[mini_batch_indices]

                # 使用自动混合精度
                with autocast(TrainerTools().parallel.device_type):
                    # 前向传播
                    policy_output, value_output = self.train_model(mb_input_ids, attention_mask=mb_attention_mask)

                    # 确保数据类型一致
                    target_dtype = policy_output['logits'].dtype
                    mb_old_log_probs = mb_old_log_probs.to(target_dtype)
                    mb_values = mb_values.to(target_dtype)
                    mb_returns = mb_returns.to(target_dtype)
                    mb_advantages = mb_advantages.to(target_dtype)
                    
                    # 自回归移动
                    # logits：回复部分
                    logits_completion = policy_output['logits'][:, prompt_len - 1: -1]
                    # 价值：回复部分
                    current_values = value_output[:, prompt_len - 1: -1]
                    # 对数概率：当前策略vs旧策略
                    current_log_probs = compute_token_losses(logits_completion, mb_completion_ids)
                    

                    # 计算PPO损失
                    loss, actor_loss, value_loss, approx_kl, clip_frac = self.criterion(
                        log_probs=current_log_probs,  # 当前对数概率
                        old_log_probs=mb_old_log_probs,  # 旧对数概率
                        values=current_values,  # 当前价值
                        old_values=mb_values,  # 旧价值
                        returns=mb_returns,  # 目标回报
                        advantages=mb_advantages,  # 优势函数
                        mask=mb_completion_mask  # 掩码
                    )

                    # 计算辅助损失
                    aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                    if policy_output.get('aux_loss') and self.train_config.loss_config.aux_loss_coef:
                        aux_loss = self.train_config.loss_config.aux_loss_coef * policy_output['aux_loss']

                # 总损失
                total_loss = loss + aux_loss
                # 反向传播
                self._backward_loss(total_loss)
                # 梯度裁剪
                self._apply_grad_clipping()
                # 参数更新
                self._apply_step()
                n_updates += 1

                # 累积统计信息
                ppo_stats["loss_with_aux"] += total_loss.detach().item()
                ppo_stats["loss_without_aux"] += loss.detach().item()
                ppo_stats["aux_loss"] += aux_loss.detach().item()
                ppo_stats["actor_loss"] += actor_loss.detach().item()
                ppo_stats["value_loss"] += value_loss.detach().item()
                ppo_stats["approx_kl"] += approx_kl.detach().item()
                ppo_stats["clip_frac"] += clip_frac.detach().item()

        # 计算平均统计信息
        if n_updates > 0:
            for key in ppo_stats:
                ppo_stats[key] /= n_updates

        return ppo_stats

    def train(self):
        """
        PPO训练主循环
        """
        global_steps = 0
        skipping_train = False

        # Epoch循环
        for epoch in range(self.train_config.n_epochs):
            file_count = len(self.train_config.file_dataset)
            # 文件循环
            for file_idx in range(file_count):
                # 创建数据集和数据加载器
                dataset, file_path = self._create_dataset(file_idx)
                train_data_loader = TrainerTools().parallel.process_dataloader(
                    dataset=dataset,
                    data_loader_kwargs=self.data_loader_kwargs,
                    sampler_kwargs=self.sampler_kwargs
                )

                # 初始化上一次保存检查点的批次索引
                last_ckpt_batch = 0
                # 当前文件的批次总数
                batch_count_per_file = len(train_data_loader)

                # 分布式同步和日志
                TrainerTools().parallel.on_epoch_start(epoch)
                self._on_file_start(epoch, file_path)

                # Batch循环
                for batch, batch_data in enumerate(train_data_loader):
                    global_steps += 1
                    # 恢复训练跳过逻辑
                    if global_steps < self.last_global_steps:
                        skipping_train = True
                        continue

                    if skipping_train:
                        TrainerTools().parallel.wait('skip train')
                        skipping_train = False

                    # 1. Rollout阶段：生成数据
                    rollout_data = self._generate_rollout_data(batch_data)

                    try:
                        # 2. Learning阶段：PPO更新
                        ppo_stats = self._ppo_learning_phase(rollout_data)

                        # 记录日志
                        self._log(
                            keys={
                                'epoch': epoch,
                                'file': f'{file_idx + 1}/{file_count}',
                                'batch': f'{batch}/{batch_count_per_file}'
                            },
                            values={
                                'loss(with aux)': ppo_stats['loss_with_aux'],
                                'loss(without aux)': ppo_stats['loss_without_aux'],
                                'aux_loss': ppo_stats['aux_loss'],
                                'actor_loss': ppo_stats['actor_loss'],
                                'value_loss': ppo_stats['value_loss'],
                                'approx_kl': ppo_stats['approx_kl'],
                                'clip_frac': ppo_stats['clip_frac'],
                                'rewards': rollout_data['env_rewards'].mean().item()
                            }
                        )
                    except Exception as e:
                        # 异常处理
                        self._on_exception(e, epoch, batch)
                    finally:
                        # batch结束保存训练步数和学习率调度器状
                        save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)

                        # 有条件保存检查点（每隔 eval_batch_interval 个批次保存一次）
                        if (batch - last_ckpt_batch) >= self.train_config.eval_batch_interval:
                            save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                            last_ckpt_batch = batch
                            self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')

                        # 清理显存
                        torch.cuda.empty_cache()

            # Epoch结束保存和评估
            if not skipping_train:
                # epoch结束保存训练步数和学习率调度器状态
                save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)

                # 无条件保存检查点（epoch 结束时总是保存）
                save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                
                # 通知所有进程epoch结束
                TrainerTools().parallel.on_epoch_end(epoch)
                
                self._on_epoch_end(tag=f'epoch:{epoch}')

        # 销毁分布式环境
        TrainerTools().parallel.destroy()
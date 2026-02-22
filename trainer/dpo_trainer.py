from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset

from .trainer import Trainer
from .train_configs import TrainConfig
from .dataset import DPODataset
from .loss import DPOLoss
from .tools import TrainerTools
from .utils import (
    autocast,  # 自动混合精度上下文管理器
    get_dpo_collate_fn,  # 获取DPO数据整理函数
    fill_loss_mask,  # 填充损失掩码
    compute_token_losses,  # 计算每个token的log probability
    disable_dropout_in_model  # 禁用模型中的dropout
)

from .checkpoint import (
    save_checkpoint,  # 保存的检查点
    save_steps,  # 保存训练步数
)

# 继承父类并重写一些方法


class DPOTrainer(Trainer):
    def __init__(
            self,
            *,
            train_config: TrainConfig,  # 训练配置对象
            eval_prompts: List[str],  # 评估用的提示词列表
            eval_image_tags: Optional[List[str]] = None  # [可选]评估图像标签列表
    ):
        # 调用父类Trainer的初始化方法
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            eval_image_tags=eval_image_tags
        )
        # 设置是否使用序列打包，DPO场景默认关闭
        # 序列打包是什么？
        # - 预训练时为了提高GPU利用率，会将多个文档拼接到一个序列中
        # - 格式如：[doc1_token1, doc1_token2, <eos>, doc2_token1, doc2_token2, <eos>]
        # - 使用doc_boundary_mask防止不同文档之间的attention交叉
        #
        # 为什么DPO必须关闭？
        # 1. DPO需要成对比较：每个样本包含chosen和rejected两个独立的完整回复
        #    - chosen: [prompt, good_response]
        #    - rejected: [prompt, bad_response]
        # 2. 如果打包序列，会破坏chosen和rejected的对应关系
        # 3. DPO需要分别计算chosen和rejected的对数概率，然后做差值比较
        self.packed_sequences = False
        # 初始化参考模型（用于计算KL散度的冻结模型）
        self.ref_model = self._init_ref_model()

    def _init_ref_model(self):
        # 创建新的模型实例作为参考模型
        ref_model = self._new_model(self.train_config)

        # 如果配置中提供了参考模型的检查点
        if self.train_config.dpo_config.ref_model_checkpoint:
            # 加载参考模型的权重
            ref_model.load_state_dict(
                self.train_config.dpo_config.ref_model_checkpoint)
            # 清空检查点数据，避免占用内存
            self.train_config.dpo_config.ref_model_checkpoint = {}

        # 使用并行框架处理参考模型
        ref_model, _ = TrainerTools().parallel.process(
            model=ref_model,  # 参考模型
            optimizer=None,  # 参考模型不需要优化器（不训练）
            kwargs=self._init_ref_model_args(),  # 参考模型的并行配置参数
            save_instance=False,  # 参考模型是冻结的，不参与训练，其状态不会改变，因此不需要TrainerTools进行保存或管理
        )

        # 将参考模型设为评估模式（关闭dropout等）
        ref_model.eval()
        # 冻结参考模型的所有参数，禁止梯度计算
        for param in ref_model.parameters():
            param.requires_grad = False

        # 返回初始化好的参考模型
        return ref_model

    # 创建新模型（重写父类方法）
    def _new_model(self, train_config: TrainConfig):
        # 调用父类方法创建基础模型
        model = super()._new_model(train_config)
        # 禁用模型中的所有dropout层（DPO训练中为了稳定性通常禁用dropout）
        disable_dropout_in_model(model)
        # 返回处理后的模型
        return model

    # 初始化损失函数（重写父类方法）
    def _init_loss(self):
        # 创建DPO损失函数实例
        criterion = DPOLoss(
            beta=self.train_config.dpo_config.loss_beta,  # DPO超参，控制奖励的强度
            label_smoothing=self.train_config.dpo_config.loss_label_smoothing,  # 标签平滑系数
            ipo=self.train_config.dpo_config.loss_ipo  # 是否使用IPO
        )

        # 返回损失函数，None表示不使用KD损失
        return criterion, None

    # 转换训练参数（重写父类方法）
    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        # 获取DPO专用的数据整理函数
        dpo_collate_fn = get_dpo_collate_fn(self.train_config.mask_prompt)
        # 调用父类方法获取基础配置
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()
        # DPO训练需要特殊的数据处理方式，例如将chosen和rejected响应配对，并生成相应的attention_mask和labels。
        # 因此，我们用DPO专用的数据整理函数 `dpo_collate_fn` 替换掉父类（Trainer）中存在的默认 `collate_fn`。
        # 这样可以确保数据加载器在每个批次中正确地准备DPO训练所需的数据格式。
        data_loader_kwargs.update({"collate_fn": dpo_collate_fn})
        #          并行参数        数据加载器参数         采样器参数
        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    # 创建数据集（重写父类方法）
    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        return DPODataset(file_path, self.processor, self.train_config.max_seq_len), file_path

    # 计算损失（占位符，DPO使用自定义的损失计算流程）
    def _calc_loss(self, inputs, attention_mask, logits, labels):
        """DPO使用自定义的损失计算流程,此方法不需要实现"""
        pass

    # 计算对数概率
    def _logprobs(self, logits, labels, attention_mask):
        """
        计算批次序列的平均对数概率
        logits (torch.Tensor): 模型输出的logits，形状为 (B, T, V)
                               B=批次大小，T=序列长度，V=词表大小
        labels (torch.Tensor): 真实标签，形状为 (B, T)
        attention_mask (torch.Tensor): 掩码张量，形状为 (B, T)
                                1表示有效token，0表示padding
        返回:
        torch.Tensor: 形状为 (B,)，表示批次中每个序列的平均对数概率
        """
        loss_masks = attention_mask.clone().bool()
        # 将prompt部分的loss_mask设为False（不计算loss）
        loss_masks = fill_loss_mask(loss_masks, labels)
        # 错位计算logits和labels
        # 去掉logits的最后一个时间步
        logits = logits[:, :-1, :]
        # 去掉labels的第一个时间步
        labels = labels[:, 1:].clone()
        # 相应地调整loss_masks
        loss_masks = loss_masks[:, 1:]

        # 将占位符-100替换为0    -100在词表中没有（从0开始）
        labels[labels == -100] = 0

        # 计算每个token的log probability（包括原本是-100的位置）
        per_token_logps = compute_token_losses(logits, labels)

        # 应用掩码，只保留需要计算损失的部分
        # - padding位置: loss_masks=False → 乘以0 → 不计入损失
        # - prompt位置: loss_masks=False → 乘以0 → 不计入损失
        # - 原-100位置: loss_masks=False → 乘以0 → 不计入损失
        # - response位置: loss_masks=True → 乘以1 → 计入损失
        # 每个序列的对数概率总和
        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        # 每个序列的对数概率平均值
        logprobs_means = (per_token_logps *
                          loss_masks).sum(-1) / loss_masks.sum(-1)

        # 返回对数概率总和和平均值
        return logprobs_sums, logprobs_means  # 形状为 (B,)

    # DPO训练主函数
    def train(self):
        # 梯度累积步数
        gradient_accumulation_steps = self.train_config.gradient_accumulation_steps
        # 全局步数计数器
        global_steps = 0
        # 跳过训练标志（用于恢复训练时跳过已训练步数）
        skipping_train = False

        # 初始化:总损失累积变量
        loss_with_aux_accumulation = 0.0
        # 初始化：不含辅助损失的DPO损失累积变量
        loss_without_aux_accumulation = 0.0
        # 初始化：辅助损失累积变量
        aux_loss_accumulation = 0.0
        # 初始化：负对数似然损失：防止模型在优化偏好时“忘掉”如何正常说话（只针对 chosen 样本）
        nll_loss_accumulation = 0.0
        # 累积的批次数量计数器
        batches_accumulated = 0

        # 辅助损失系数
        aux_loss_coef = self.train_config.loss_config.aux_loss_coef
        # 负对数似然损失损失系数
        nll_loss_coef = self.train_config.dpo_config.nll_loss_coef

        # 遍历每个训练epoch
        for epoch in range(self.train_config.n_epochs):
            # 训练模式
            self.train_model.train()
            # 训练文件数量
            file_count = len(self.train_config.file_dataset)

            # 遍历每个训练文件
            for file_idx in range(file_count):
                # 创建数据集和获取文件路径
                dataset, file_path = self._create_dataset(file_idx)
                # 创建数据加载器（处理分布式采样）
                train_data_loader = TrainerTools().parallel.process_dataloader(
                    dataset=dataset,  # 数据集实例
                    data_loader_kwargs=self.data_loader_kwargs,  # 数据加载器参数
                    sampler_kwargs=self.sampler_kwargs  # 采样器参数
                )

                # 初始化上一次保存检查点的批次索引
                last_ckpt_batch = 0
                # 当前文件的批次总数
                batch_count_per_file = len(train_data_loader)
                # 分布式训练：通知所有进程epoch开始
                TrainerTools().parallel.on_epoch_start(epoch)
                # 打印日志：文件开始处理
                self._on_file_start(epoch, file_path)

                # 遍历每个批次数据
                for batch, batch_data in enumerate(train_data_loader):
                    global_steps += 1  # 全局步数自增
                    # 如果当前步数小于已训练步数（恢复训练时），跳过
                    if global_steps < self.last_global_steps:
                        skipping_train = True
                        continue

                    # 判断是否需要更新梯度（梯度累积逻辑）
                    if skipping_train:
                        # 如果是恢复训练后的“第一步”（下面if skipping_train会改成skipping_train = False），我们通常希望只跑数据流程，而不进行参数更新（因为这一步在保存 Checkpoint 时通常已经更新过了）
                        need_update_grad = False
                    elif gradient_accumulation_steps > 1:
                        # 梯度累积：每accumulation_steps/或最后一个批次，更新一次
                        need_update_grad = (
                            batch + 1) % gradient_accumulation_steps == 0 or batch == batch_count_per_file - 1
                    else:
                        # 无梯度累积，每个批次都更新
                        need_update_grad = True

                    # 上面if global_steps < self.last_global_steps如果不执行，说明当前步数大于已训练步数，恢复训练（等待其他进程同步）
                    if skipping_train:
                        # 协调所有进程，确保同步
                        TrainerTools().parallel.wait('skip train')
                        # 清除跳过训练标志
                        skipping_train = False

                    try:
                        # 从批次数据中提取chosen（被选择的）样本的输入和标签，并移到GPU
                        chosen_inputs: torch.Tensor = batch_data['chosen_inputs'].to(
                            TrainerTools().parallel.device)
                        chosen_labels: torch.Tensor = batch_data['chosen_labels'].to(
                            TrainerTools().parallel.device)

                        # 从批次数据中提取rejected（被拒绝的）样本的输入和标签，并移到GPU
                        rejected_inputs: torch.Tensor = batch_data['rejected_inputs'].to(
                            TrainerTools().parallel.device)
                        rejected_labels: torch.Tensor = batch_data['rejected_labels'].to(
                            TrainerTools().parallel.device)

                        # 创建chosen样本的attention mask（pad token位置为False）
                        chosen_attention_masks: torch.Tensor = chosen_inputs != TrainerTools().tokenizer.pad
                        # 创建rejected样本的attention mask（pad token位置为False）
                        rejected_attention_masks: torch.Tensor = rejected_inputs != TrainerTools().tokenizer.pad

                        # 在batch维度拼接chosen和rejected样本
                        # 拼接后的顺序：[chosen_1, chosen_2, ..., rejected_1, rejected_2, ...]
                        concat_inputs = torch.concat(
                            [chosen_inputs, rejected_inputs], dim=0)
                        concat_labels = torch.concat(
                            [chosen_labels, rejected_labels], dim=0)
                        concat_attention_masks = torch.concat(
                            [chosen_attention_masks, rejected_attention_masks], dim=0)

                        # 如果是分布式训练
                        if TrainerTools().parallel.parallel_train:
                            # 获取是否需要在反向传播时同步梯度
                            self.train_model.require_backward_grad_sync = need_update_grad

                        # 使用自动混合精度
                        with autocast(TrainerTools().parallel.device_type):
                            # 策略模型（正在训练的模型）前向传播
                            policy_outputs = self.train_model(
                                concat_inputs, attention_mask=concat_attention_masks)
                            # 计算策略模型的对数概率（总和与平均值）
                            policy_logprobs_sums, policy_logprobs_means = self._logprobs(
                                policy_outputs['logits'], concat_labels, concat_attention_masks)

                            # 不计算梯度（参考模型是冻结的）
                            with torch.no_grad():
                                # 参考模型前向传播
                                ref_outputs = self.ref_model(
                                    concat_inputs, attention_mask=concat_attention_masks)
                                # 计算参考模型的对数概率（只需要总和）
                                ref_logprobs_sums, _ = self._logprobs(
                                    ref_outputs['logits'], concat_labels, concat_attention_masks)

                            # 从拼接的结果中分离出chosen和rejected的对数概率
                            # 策略模型对chosen样本的对数概率
                            policy_chosen_logps = policy_logprobs_sums[:chosen_inputs.shape[0]]
                            # 策略模型对rejected样本的对数概率
                            policy_rejected_logps = policy_logprobs_sums[chosen_inputs.shape[0]:]

                            # 参考模型对chosen样本的对数概率
                            ref_chosen_logps = ref_logprobs_sums[:chosen_inputs.shape[0]]
                            # 参考模型对rejected样本的对数概率
                            ref_rejected_logps = ref_logprobs_sums[chosen_inputs.shape[0]:]

                            # 负对数似然损失：防止模型在优化偏好时“忘掉”如何正常说话（只针对 chosen 样本）
                            nll_loss = - \
                                policy_logprobs_means[:chosen_inputs.shape[0]].mean(
                                )

                            # 计算DPO损失
                            loss = self.criterion(
                                policy_chosen_logps,  # 策略模型对chosen样本的对数概率
                                policy_rejected_logps,  # 策略模型对rejected样本的对数概率
                                ref_chosen_logps,  # 参考模型对chosen样本的对数概率
                                ref_rejected_logps  # 参考模型对rejected样本的对数概率
                            )

                            # 如果配置了辅助损失系数且模型输出了辅助损失
                            if aux_loss_coef and policy_outputs.get('aux_loss'):
                                # 加权
                                aux_loss = aux_loss_coef * \
                                    policy_outputs.get('aux_loss')
                            else:
                                # 否则辅助损失为0
                                aux_loss = torch.tensor(
                                    0.0, device=loss.device, dtype=loss.dtype)

                            # 如果配置了NLL损失系数且NLL损失存在
                            # 如果你的数据集非常小，或者你在训练过程中发现模型开始输出乱码、语法错误，或者遗忘了之前的知识，可以尝试开启它。
                            if nll_loss_coef and nll_loss:
                                # 加权
                                nll_loss = nll_loss_coef * nll_loss
                            # (标准的 DPO 算法通常不加 NLL Loss)
                            else:
                                # 否则NLL损失为0
                                nll_loss = torch.tensor(
                                    0.0, device=loss.device, dtype=loss.dtype)

                        # 如果启用了梯度累积（accumulation_steps > 1）
                        if gradient_accumulation_steps > 1:
                            # 将损失除以累积步数
                            loss = loss / gradient_accumulation_steps
                            aux_loss = aux_loss / gradient_accumulation_steps
                            nll_loss = nll_loss / gradient_accumulation_steps

                        # 计算总损失（DPO损失 + 辅助损失 + NLL损失）
                        total_loss = loss + aux_loss + nll_loss
                        # 执行反向传播
                        self._backward_loss(total_loss)

                        # 累积各项损失值（用于日志记录）
                        loss_with_aux_accumulation += total_loss.detach().item()
                        loss_without_aux_accumulation += loss.detach().item()
                        aux_loss_accumulation += aux_loss.detach().item()
                        nll_loss_accumulation += nll_loss.detach().item()

                        batches_accumulated += 1 # 累积批次计数器自增

                        # 如果需要更新梯度
                        if need_update_grad:
                            # 应用梯度裁剪（防止梯度爆炸）
                            self._apply_grad_clipping()
                            # 更新模型参数
                            self._apply_step()

                            # 计算平均损失（分布式场景下会进行all_reduce）
                            avg_loss, avg_loss_without_aux, avg_aux_loss, avg_nll_loss = self._avg_loss(
                                losses=[
                                    loss_with_aux_accumulation,  # 包含所有损失的总损失
                                    loss_without_aux_accumulation,  # 仅DPO损失
                                    aux_loss_accumulation,  # 仅辅助损失
                                    nll_loss_accumulation,  # 仅NLL损失
                                ],
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                batches_accumulated=batches_accumulated
                            )

                            # 记录训练日志
                            self._log(
                                keys={
                                    'epoch': epoch,  # 当前epoch
                                    # 当前文件/总文件数
                                    'file': f'{file_idx + 1}/{file_count}',
                                    # 当前batch/每文件batch总数
                                    'batch': f'{batch}/{batch_count_per_file}',
                                },
                                values={
                                    'loss(with aux and nll)': avg_loss,  # 总损失
                                    'loss(without aux and nll)': avg_loss_without_aux,# DPO损失
                                    'aux_loss': avg_aux_loss,  # 辅助损失
                                    'nll_loss': avg_nll_loss  # NLL损失
                                }
                            )

                            # 重置损失累积变量（准备下一个梯度累积周期）
                            loss_with_aux_accumulation = 0.0
                            loss_without_aux_accumulation = 0.0
                            aux_loss_accumulation = 0.0
                            nll_loss_accumulation = 0.0
                            batches_accumulated = 0
                    except Exception as e:
                        # 捕获异常并记录（包含epoch和batch信息）
                        self._on_exception(e, epoch, batch)
                    finally:
                        # 无论是否发生异常，都执行清理和保存操作
                        if need_update_grad:
                            # batch结束保存训练步数和学习率调度器状
                            save_steps(global_steps=global_steps,
                                       lr_scheduler=self.lr_scheduler)

                            # 有条件保存检查点（每隔 eval_batch_interval 个批次保存一次）
                            if (batch - last_ckpt_batch) >= self.train_config.eval_batch_interval:
                                save_checkpoint(
                                    model=self.train_model, optimizer=self.optimizer)

                                # 更新last_ckpt_batch的批次索引
                                last_ckpt_batch = batch

                                self._on_batch_end(
                                    tag=f'epoch:{epoch}/batch:{batch}')

                        try:
                            # 尝试删除loss变量以释放显存
                            del loss
                        except UnboundLocalError:
                            ...  # 如果loss未定义则忽略

            # epoch结束
            if not skipping_train:
                # epoch结束保存训练步数和学习率调度器状态
                save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)
                # 无条件保存检查点（epoch 结束时总是保存）
                # save_checkpoint(model=self.train_model, optimizer=self.optimizer)  # 是否每个epoch结束保存checkpoint，适合大数据集

                # 通知所有进程epoch结束
                TrainerTools().parallel.on_epoch_end(epoch)

                self._on_epoch_end(tag=f'epoch:{epoch}')

        # 所有epoch训练完成后保存checkpoint（适合小数据集）
        save_checkpoint(model=self.train_model, optimizer=self.optimizer)
        # 训练完成，销毁分布式训练环境
        TrainerTools().parallel.destroy()

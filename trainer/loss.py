from typing import List, Optional
import torch
from torch import nn
import torch.nn.functional as F

class LMLoss(nn.Module):
    """
    语言模型损失
    """
    def __init__(
            self,
            ignore_index: int = -100,  # 指定在计算损失时忽略的标签索引（通常用于padding）
            *,
            critical_tokens: Optional[List[int]] = None,  # 关键token的ID列表，可选
            critical_alpha: float = 1.0,  # 关键token的权重系数
            vocab_size: int = 0  # 词表大小，用于初始化权重张量
    ):
        super().__init__()
        self.ignore_index = ignore_index  # 在计算损失时忽略的标签索引（通常用于padding）
        self.critical_tokens = critical_tokens  # 关键token列表
        self.critical_alpha = critical_alpha  # 关键token权重

        # 如果指定了关键token且词表大小大于0，则初始化自定义权重
        if critical_tokens and vocab_size > 0:
            self.register_buffer('weights', torch.ones(vocab_size))  # 注册一个不参与梯度更新但跟模型关联的buffer，初始化为全1
            # 为关键token设置权重
            self.weights[self.critical_tokens] = critical_alpha  # 将关键token位置的权重设置为alpha值


    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits shape (batch, seq_len, vocab_size) -> 模型的原始输出
        # labels shape (batch, seq_len) -> 真实标签
        
        # 语言模型预测下一个词，所以logits取前N-1个，labels取后N-1个（错位对齐）
        shift_logits = logits[..., :-1, :].contiguous()  # 截取除了最后一个时间步的logits，并确保内存连续
        shift_labels = labels[..., 1:].contiguous()  # 截取除了第一个时间步的labels，并确保内存连续

        logits = shift_logits.reshape(-1, logits.shape[-1])  # 将logits展平为 (batch * (seq_len-1), vocab_size)
        targets = shift_labels.reshape(-1)  # 将标签展平为 (batch * (seq_len-1))

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(
            logits,  # 预测值
            targets,  # 真实值
            ignore_index=self.ignore_index,  # 忽略padding部分的损失
            weight=self.weights.to(logits.device, dtype=logits.dtype) if self.critical_tokens else None  # 如果有定义，传入类别权重
        )
        return ce_loss  # 返回计算出的损失值


class KDLoss(nn.Module):
    """
    知识蒸馏损失
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index  # 在计算损失时忽略的标签索引（通常用于padding）

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 计算教师模型的概率分布（Softmax）
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)  # [batch, seq_len, vocab_size]
        
        # 检查学生模型的logits是否存在无穷大（防止数值不稳定）
        # inf_mask=True的位置表示 “无效位置”，False表示有效位置
        inf_mask = torch.isinf(logits)  # 生成无穷大掩码

        # 计算学生模型的对数概率
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)  # [batch, seq_len, vocab_size]
        
        # 计算教师概率与学生对数概率的乘积（用于计算KL散度或交叉熵的一部分）
        # 先乘积，后填充
        # logits是inf即inf_mask=True的位置会被填充为0，避免NaN
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)

        # 在词表维度求和，得到每个token位置的损失值，并展平
        x = torch.sum(prod_probs, dim=-1).view(-1)  # shape: [batch * seq_len]
        
        # 创建有效token的掩码（非padding部分为1，padding部分为0）
        mask = (labels != self.ignore_index).int()  # shape: [batch, seq_len]

        # 计算加权平均损失：
        # 分子：有效token位置的损失之和（注意前面是log_softmax，通常KD损失需要取负号来最小化）
        # 分母：有效token的总数
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss  # 返回蒸馏损失


class DPOLoss(nn.Module):
    """
    DPO Loss (直接偏好优化损失)
    """
    def __init__(
            self,
            beta: float,  # KL散度惩罚系数，控制偏离参考模型的程度
            label_smoothing: float = 0.0,  # 标签平滑系数
            ipo: bool = False  # 是否使用IPO (Identity Preference Optimization) 损失
    ):
        super().__init__()
        self.beta = beta  # 保存beta参数
        self.label_smoothing = label_smoothing  # 标签平滑
        self.ipo = ipo  # 保存IPO标志位

    def forward(
            self,
            policy_chosen_logps: torch.Tensor,  # 策略模型对"被选中"回答的log概率
            policy_reject_logps: torch.Tensor,  # 策略模型对"被拒绝"回答的log概率
            ref_chosen_logps: torch.Tensor,     # 参考模型对"被选中"回答的log概率
            ref_reject_logps: torch.Tensor      # 参考模型对"被拒绝"回答的log概率
    ) -> torch.Tensor:
        # 计算策略模型对 Chosen 和 Rejected 的 log 概率差
        pi_logratios = policy_chosen_logps - policy_reject_logps
        # 计算参考模型对 Chosen 和 Rejected 的 log 概率差
        ref_logratios = ref_chosen_logps - ref_reject_logps
        
        # DPO的核心逻辑：logits是策略模型相对于参考模型的优势比率
        logits = pi_logratios - ref_logratios

        if self.ipo:
            # 如果使用IPO损失
            # 公式参考: https://arxiv.org/pdf/2310.12036v2.pdf Eq. 17
            losses = (logits - 1 / (2 * self.beta)) ** 2  # 均方误差形式
        else:
            # 默认使用DPO损失
            # 公式参考: https://ericmitchell.ai/cdpo.pdf Eq. 3
            # 如果 label_smoothing=0，则还原为原始DPO公式 (Sigmoid Cross Entropy)
            losses = (
                    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)  # 正样本部分
                    - F.logsigmoid(-self.beta * logits) * self.label_smoothing      # 负样本部分（标签平滑引入）
            )

        loss = losses.mean()  # 对batch内所有样本求平均

        return loss  # 返回DPO损失值


class PPOLoss(nn.Module):
    """
    PPO损失函数
    统一计算 Actor 和 Value 的损失
    """

    def __init__(
            self,
            clip_eps: float,  # PPO裁剪范围的epsilon值 (例如 0.2)
            vf_coef: float,   # 价值函数损失的系数 (例如 0.1 或 0.5)
    ):
        super().__init__()
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef

    def forward(
            self,
            log_probs: torch.Tensor,     # 当前策略的log probabilities, 形状: [batch_size, seq_len]
            old_log_probs: torch.Tensor, # 生成rollout时的旧策略的log probabilities, 形状: [batch_size, seq_len]
            values: torch.Tensor,        # 当前评论家模型输出的价值, 形状: [batch_size, seq_len]
            old_values: torch.Tensor,    # 生成rollout时的旧价值, 形状: [batch_size, seq_len]
            returns: torch.Tensor,       # GAE计算出的回报, 形状: [batch_size, seq_len]
            advantages: torch.Tensor,    # GAE计算出的优势函数, 形状: [batch_size, seq_len]
            mask: torch.Tensor           # 掩码，只计算生成部分的损失 (Prompt部分通常不计算), 形状: [batch_size, seq_len]
    ):
        """
        计算PPO的总损失、Actor损失和Value损失。
        """
        # --- Value Loss (价值损失) 计算 ---
        # 裁剪价值函数的预测值，限制其偏离旧价值函数的程度
        values_clipped = old_values + torch.clamp(values - old_values, -self.clip_eps, self.clip_eps)
        
        # 计算未裁剪的均方误差损失
        vf_loss_unclipped = F.mse_loss(values, returns, reduction='none')
        # 计算裁剪后的均方误差损失
        vf_loss_clipped = F.mse_loss(values_clipped, returns, reduction='none')
        
        # 取两者中的最大值（这是一种防御性策略，防止Value Function更新过快）
        value_loss = torch.max(vf_loss_unclipped, vf_loss_clipped)
        
        # 应用掩码并计算平均值
        value_loss = 0.5 * (value_loss * mask).sum() / mask.sum().clamp(min=1.0) # 0.5是MSE公式惯例
        value_loss = value_loss * self.vf_coef  # 乘以价值损失系数

        # --- Actor Loss (策略损失) 计算 ---
        # 计算新旧策略的概率比 r_t = exp(log_prob_new - log_prob_old)
        # ratio 形状: [batch_size, seq_len]
        ratio = torch.exp(log_probs - old_log_probs)

        # PPO裁剪替代目标（Clipped Surrogate Objective）
        # surr1 (未裁剪部分): ratio * A_t
        surr1 = ratio * advantages
        
        # surr2 (裁剪部分): clamp(ratio, 1-eps, 1+eps) * A_t
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

        # 取两者中较小的一个（min），并加负号（因为我们要最大化这个目标，所以代码中实现为最小化其负值）
        # 我们只关心生成部分（由mask标记）的损失
        actor_loss = -torch.sum(torch.min(surr1, surr2) * mask) / torch.sum(mask).clamp(min=1.0)

        # 总损失 = Actor损失 + Value损失
        total_loss = actor_loss + value_loss

        # --- 统计指标计算 (不参与梯度反向传播) ---
        with torch.no_grad():
            # 计算近似KL散度 (Approximate KL Divergence) 用于监控
            # 公式: sum((ratio - 1) - log(ratio))
            logratios = log_probs - old_log_probs
            approx_kl = torch.sum(((torch.exp(logratios) - 1) - logratios) * mask) / mask.sum().clamp(min=1.0)

            # 计算被裁剪的比例 (Clip Fraction)，用于监控训练稳定性
            clipped = ratio.gt(1.0 + self.clip_eps) | ratio.lt(1.0 - self.clip_eps) # 判断是否超出范围
            clip_frac = torch.sum(clipped.float() * mask) / mask.sum().clamp(min=1.0) # 计算比例

        return total_loss, actor_loss, value_loss, approx_kl, clip_frac  # 返回各项损失和指标


class GRPOLoss(nn.Module):
    """
    GRPO (Group Relative Policy Optimization) 损失函数
    以及变体 GSPO 等。
    """
    def __init__(
            self,
            beta: float,  # KL惩罚系数
            clip_eps_low: float,  # 裁剪下界 epsilon
            clip_eps_high: Optional[float] = None,  # 裁剪上界 epsilon (可选)
            delta: Optional[float] = None,  # 另一种截断方式的参数 (可选)
            importance_sampling_level: str = 'token',  # 重要性采样级别 ('token' 或 'seq')
            loss_type: str = 'grpo',  # 损失类型 ('grpo', 'bnpo', 'dr_grpo')
            gen_max_new_tokens: Optional[float] = None  # 生成的最大token数 (用于dr_grpo归一化)
    ):
        super().__init__()

        self.beta = beta  # 保存beta
        self.clip_eps_low = clip_eps_low  # 保存下界
        self.clip_eps_high = clip_eps_high if clip_eps_high else clip_eps_low  # 如果未指定上界，则与下界相同
        self.delta = delta  # 保存delta
        self.importance_sampling_level = importance_sampling_level  # 保存采样级别
        self.loss_type = loss_type  # 保存损失类型
        self.gen_max_new_tokens = gen_max_new_tokens  # 保存最大token数

    def forward(
            self,
            log_probs: torch.Tensor,        # 当前策略 log probs
            old_log_probs: torch.Tensor,    # 旧策略 log probs
            ref_log_probs: torch.Tensor,    # 参考模型 log probs
            completion_mask: torch.Tensor,  # 回答部分的掩码
            advantages: torch.Tensor        # 优势函数值
    ) -> torch.Tensor:

        # 计算 KL散度（很简单的数学变换）
        if self.beta != 0.0:
            per_token_kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
        else:
            per_token_kl = None  # 如果beta为0，不计算KL

        # 计算新旧策略的 log ratio
        log_ratio = log_probs - old_log_probs
        
        # 计算重要性采样权重
        if self.importance_sampling_level == "seq":
            # GSPO模式：序列级别的权重
            # 计算整个序列的 log ratio 之和，并归一化
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)  # 扩展维度以便广播
        else:
            # GRPO 模式：Token级别的权重，直接使用 log_ratio
            log_importance_weights = log_ratio

        # 计算重要性系数 rho = exp(log_weights)
        coef_1 = torch.exp(log_importance_weights)
        # 对系数进行裁剪，限制在 [1-eps_low, 1+eps_high] 之间
        coef_2 = torch.clamp(coef_1, 1 - self.clip_eps_low, 1 + self.clip_eps_high)

        # 最大不能超过阈值：self.delta
        if self.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.delta)

        # 计算两种损失项 (类似PPO)
        per_token_loss1 = coef_1 * advantages  # 未裁剪项
        per_token_loss2 = coef_2 * advantages  # 裁剪项
        
        # 取最小值并取负
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # 如果启用KL惩罚，将其加到损失中
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # 根据不同的损失类型聚合最终损失
        if self.loss_type == "bnpo":
            # BNPO模式：对所有token求和后除以总有效token数
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            # DR-GRPO模式：归一化分母包含最大生成长度，通常用于DeepSeekR1等场景
            assert self.gen_max_new_tokens is not None  # 确保已设置max_tokens
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.gen_max_new_tokens)
        else:
            # 默认GRPO模式：先在序列维度平均，再在batch维度平均
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        return loss  # 返回最终损失
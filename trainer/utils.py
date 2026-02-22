import random
# 从contextlib导入nullcontext，用于创建空的上下文管理器（无操作）
from contextlib import nullcontext
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .tools import TrainerTools
import numpy as np
from typing import Union, List, Optional


def set_seed(seed=42):
    """设置全局随机种子以保证实验可复现"""
    # 设置Python内置random模块的随机种子
    random.seed(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置PyTorch CPU的随机种子
    torch.manual_seed(seed)
    # 设置PyTorch GPU的随机种子（单GPU）
    torch.cuda.manual_seed(seed)
    # 设置PyTorch所有GPU的随机种子（多GPU）
    torch.cuda.manual_seed_all(seed)


def autocast(device_type):
    # 检查是否启用混合精度训练
    # amp（Automatic Mixed Precision）（自动混合精度）
    if TrainerTools().use_amp:
        # GPU可用且支持bfloat16时使用，否则用float16
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        # 创建PyTorch的autocast自动将模型的权重、激活等张量从高精度（如 float32）转换为低精度（如 float16/bfloat16）以加速计算
        return torch.autocast(
            device_type=device_type,  # 指定设备类型（如"cuda"或"cpu"）
            dtype=dtype,              # 指定混合精度使用的数据类型
            enabled=True,             # 启用autocast
            cache_enabled=None        # 缓存机制:True启用，False禁用，None默认策略：在 CUDA 设备上默认启用缓存（因 GPU 类型转换开销更高），在 CPU 上默认禁用（CPU 类型转换开销较低，缓存收益有限
        )
    else:
        # 不启用AMP时，返回空上下文管理器（无任何操作）
        return nullcontext()


def create_doc_boundary_mask(
        input_ids: torch.Tensor,
        dtype: torch.dtype
) -> torch.Tensor:
    """
    根据文档结束符 (eot) 的位置，创建一个 attention mask 来阻止跨文档的注意力。
    Args:
        input_ids (torch.Tensor): 输入的 token ID 张量，形状为 (bsz, seq_len)。
        dtype (torch.dtype): 数据类型。

    Returns:
        torch.Tensor: 符合 attention 机制要求的 mask 张量，
                      形状为 (bsz, 1, seq_len, seq_len)。
                      值为 -inf 的位置表示被屏蔽，值为 0 的位置表示允许注意力。
    """
    # 获取输入张量的batch size（批次大小）和sequence length（序列长度）
    bsz, seq_len = input_ids.shape
    # is_eot是布尔张量，形状为(bsz, seq_len)，标记每个位置是否是文档结束符
    is_eot = (input_ids == TrainerTools().tokenizer.end)

    # 使用cumsum（累加和）来创建递增的文档ID。一个token所属的文档ID，
    # 取决于它前面有多少个eot。
    # 示例:
    # input_ids:        [[1, 2, 3, eot, 4, 5, eot]]
    # is_eot:           [F, F, F, T, F, F, T] -> [0, 0, 0, 1, 0, 0, 1]
    # doc_ids_ending:   [0, 0, 0, 1, 1, 1, 2]（cumsum的结果）
    # doc_ids:          [0, 0, 0, 0, 1, 1, 1]（向右移位后的结果）
    # 这个结果正确地将文档0分配给了前四个token，将文档1分配给了后三个token。
    doc_ids_ending = torch.cumsum(is_eot, dim=-1)
    # 对累加结果后截1位，左填充数=1，右填充数=0，得到每个token实际所属的文档ID
    doc_ids = F.pad(doc_ids_ending[:, :-1], pad=(1, 0), value=0)
    # 确保每个 token 只能关注同一文档内的 token
    # 因果 mask 阻止关注后面的 token，文档边界 mask 只负责阻止后面文档关注前面文档
    # 将doc_ids增加维度，形状变为(bsz, seq_len, 1)，用于query的文档ID比较
    query_doc_ids = doc_ids.unsqueeze(2)
    # 将doc_ids增加维度，形状变为(bsz, 1, seq_len)，用于key的文档ID比较
    key_doc_ids = doc_ids.unsqueeze(1)
    # 当query的文档 ID 大于key的文档 ID = 当前 query 属于 “后面的文档”，而 key 属于 “前面的文档”。
    # 利用PyTorch的广播机制，`query_doc_ids > key_doc_ids`会创建一个形状为(bsz, seq_len, seq_len)的布尔张量。
    # 当query的文档ID大于key的文档ID时，值为True，这正是我们需要屏蔽的位置。
    boundary_mask = query_doc_ids > key_doc_ids

    # 创建全0张量，形状为(bsz, seq_len, seq_len)，设备和数据类型与输入一致
    final_mask = torch.zeros(
        (bsz, seq_len, seq_len), device=input_ids.device, dtype=dtype
    )
    # 将boundary_mask中为True的位置填充为负无穷（表示屏蔽）
    # 值为 0 的位置表示允许注意力
    final_mask.masked_fill_(boundary_mask, torch.finfo(dtype).min)
    
    # 增加一个维度以匹配多头注意力的输入要求（bsz, num_heads, seq_len, seq_len）
    return final_mask.unsqueeze(1)


def generate_position_ids(input_ids: torch.Tensor):
    """
    为打包序列生成position_ids张量。
    Args:
      input_ids (torch.Tensor): 输入的token ID张量 (batch_size, sequence_length)。
    Returns:
      torch.Tensor: 生成的position_ids张量。
    """
    # 获取输入张量的形状：batch_size（批次大小）和sequence_length（序列长度）
    batch_size, seq_length = input_ids.shape

    # 创建一个与输入形状相同、全为0的张量来存储position_ids
    # 第一个token的位置永远是0，所以这个初始化是正确的
    position_ids = torch.zeros_like(input_ids, dtype=torch.long)

    # 从索引 1 (第二个token) 开始遍历到最后
    for t in range(1, seq_length):
        # 检查 前一个 token (t-1) 是不是结束符，为批次中的每个序列生成一个布尔值
        is_reset_token = (input_ids[:, t - 1] == TrainerTools().tokenizer.end)
        # 仅仅是把 t-1 位置的值读出来，赋值给临时变量 prev_position_ids
        prev_position_ids = position_ids[:, t - 1]

        # 核心逻辑：
        # 如果前一个是结束符 -> 当前位置重置为 0
        # 如果前一个不是结束符 -> 当前位置 = 前一个位置 + 1
        # position_ids: [0, 1, 2, 0, 1]
        position_ids[:, t] = torch.where(is_reset_token, 0, prev_position_ids + 1)

    return position_ids


def repeat_image_tok(
        tokens: torch.Tensor,
        tokens_per_image: int
) -> torch.Tensor:
    """将图像token重复指定次数（如<image>→<image><image><image>）"""
    # 获取tokenizer中的图像tokenID
    image_tok = TrainerTools().tokenizer.image
    # 创建掩码，标记tokens中哪些位置是图像token
    mask = (tokens == image_tok)
    # 如果没有图像token，直接返回原tokens
    if not mask.any():
        return tokens

    # 所有值为 True（或非0）的元素的索引=图片token的索引
    idxs = torch.nonzero(mask, as_tuple=False)# as_tuple=False 不返回元组，而是返回张量
    # 找到第一个<image>的索引
    image_tok_idx = idxs[0, 0].item()
    # 创建重复tokens_per_image次的图像token张量
    repeat_image_toks = torch.tensor([image_tok] * tokens_per_image, dtype=tokens.dtype, device=tokens.device)
    # 拼接张量：[101, 202] + [<image>, <image>, <image>] + [303, 404]
    new_tokens = torch.cat([tokens[:image_tok_idx], repeat_image_toks, tokens[image_tok_idx + 1:]], dim=-1)
    return new_tokens


def batch_repeat_image_tok(
        tokens: torch.Tensor,
        tokens_per_image: int
) -> torch.Tensor:
    """对批次中的每个序列执行图像token重复操作"""
    # 创建空列表存储处理后的tokens
    new_tokens = []

    # 遍历批次中的每个token序列
    for token in tokens:
        # 对单个序列执行图像token重复，并添加到列表
        new_tokens.append(repeat_image_tok(token, tokens_per_image))

    # 将列表中的张量堆叠成批次张量（形状：[batch_size, seq_len]）
    return torch.stack(new_tokens, dim=0)


def pretrain_collate_fn(batch_data):
    """预训练任务的collate函数，用于将数据批次处理为模型输入格式"""
    # [[x,x,x], [y,y,y]] → 对批次中的序列进行填充对齐（使用pad_token）
    inputs = pad_sequence(batch_data, batch_first=True, padding_value=TrainerTools().tokenizer.pad)
    # crossEntropy默认的ignore_index是-100 → 对labels填充-100（不计算这些位置的loss）
    labels = pad_sequence(batch_data, batch_first=True, padding_value=-100)

    return {
        'inputs': inputs,
        'labels': labels
    }


def get_sft_collate_fn(mask_prompt: bool):
    """SFT（监督微调）任务的collate函数，支持是否屏蔽prompt部分的loss"""
    
    def sft_collate_fn(batch_data):
        '''只对模型的回答（Response）计算 Loss，而忽略用户的提问（Prompt/Instruction）
        第1轮 User: [USER]你好 -> Mask (不学)
        第1轮 AI: [BOT]我好[SEP] -> 计算 Loss (学！)
        第2轮 User: [USER]很好 -> Mask (作为历史背景输入，但不计算Loss)
        第2轮 AI: [BOT]不好[SEP] -> 计算 Loss (学！)
        '''
        # 创建空列表存储输入数据和图像标签
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = [] # 收集所有样本的图片
        has_image = False
        # 遍历批次数据
        for item in batch_data:
            # 添加输入序列到列表
            batch_input_ids.append(item['input_ids'])
            # 添加注意力掩码到列表
            if 'attention_mask' in item:
                batch_attention_mask.append(item['attention_mask'])
            # 收集 pixel_values
            pv = item.get('pixel_values')
            if pv is not None:
                has_image = True
                batch_pixel_values.append(pv)

        # 对输入序列进行填充对齐（使用pad_token）
        input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=TrainerTools().tokenizer.pad)
        
        # 对注意力掩码进行填充对齐（使用0）
        attention_mask = None
        if batch_attention_mask:
            attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
        
        # 初始化labels为input_ids的克隆
        labels = input_ids.clone()
        
        # 对labels填充-100（不计算这些位置的loss）
        labels = torch.where(input_ids == TrainerTools().tokenizer.pad, -100, labels)

        # 如果需要屏蔽prompt部分，则处理labels（将prompt部分设为-100）
        if mask_prompt:
            labels = _mask_prompt(labels)

        # 处理 pixel_values 的拼接
        final_pixel_values = None
        if has_image and len(batch_pixel_values) > 0:
            # batch_pixel_values 是一个 list，每个元素是 [num_images, 3, H, W]
            # 我们将它们在第 0 维拼接
            final_pixel_values = torch.cat(batch_pixel_values, dim=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': final_pixel_values # 传入 Trainer
        }

    # 闭包：内部函数能够 “记住并访问” 外部函数的变量、参数等
    # 在 PyTorch 中，DataLoader 的 collate_fn 只能接收一个参数：batch_data，那么mask_prompt=True传不进去
    # my_collate_func = get_sft_collate_fn(mask_prompt=True)
    # 将这个生成的函数传给 DataLoader
    # DataLoader 内部调用时，只需要传 batch_data，不需要管 mask_prompt
    # loader = DataLoader(dataset, collate_fn=my_collate_func)
    return sft_collate_fn


def get_dpo_collate_fn(mask_prompt: bool):
    """DPO（直接偏好优化）任务的collate函数，支持是否屏蔽prompt部分的loss"""
    def dpo_collate_fn(batch_data):
        # batch_data: [{'chosen': chosen, 'rejected': rejected}, {'chosen': chosen, 'rejected': rejected}]
        # 创建空列表存储chosen和rejected的输入、标签
        chosen_inputs = []
        chosen_labels = []
        rejected_inputs = []
        rejected_labels = []

        # 计算批次中chosen和rejected中最大的序列长度
        max_len = 0
        for key in ['chosen', 'rejected']:
            max_len = max(max(len(item[key]) for item in batch_data), max_len)

        # 遍历批次数据，对序列进行填充对齐
        for item in batch_data:
            # 处理chosen序列：填充pad_token到max_len长度
            chosen_sequence = item['chosen']
            chosen_inputs.append(chosen_sequence + [TrainerTools().tokenizer.pad] * (max_len - len(chosen_sequence)))
            # 处理chosen标签：填充-100到max_len长度
            chosen_labels.append(chosen_sequence + [-100] * (max_len - len(chosen_sequence)))

            # 处理rejected序列：填充pad_token到max_len长度
            rejected_sequence = item['rejected']
            rejected_inputs.append(rejected_sequence + [TrainerTools().tokenizer.pad] * (max_len - len(rejected_sequence)))
            # 处理rejected标签：填充-100到max_len长度
            rejected_labels.append(rejected_sequence + [-100] * (max_len - len(rejected_sequence)))

        # 将列表转换为PyTorch长整型张量
        chosen_inputs = torch.tensor(chosen_inputs).long()
        chosen_labels = torch.tensor(chosen_labels).long()
        # 如果需要屏蔽prompt部分，处理chosen标签
        if mask_prompt:
            chosen_labels = _mask_prompt(chosen_labels)

        # 将列表转换为PyTorch长整型张量
        rejected_inputs = torch.tensor(rejected_inputs).long()
        rejected_labels = torch.tensor(rejected_labels).long()
        # 如果需要屏蔽prompt部分，处理rejected标签
        if mask_prompt:
            rejected_labels = _mask_prompt(rejected_labels)

        # 返回DPO任务的输入字典
        return {
            'chosen_inputs': chosen_inputs,
            'chosen_labels': chosen_labels,
            'rejected_inputs': rejected_inputs,
            'rejected_labels': rejected_labels
        }

    # 闭包：内部函数能够 “记住并访问” 外部函数的变量、参数等
    return dpo_collate_fn


def split_batch(data_per_batch: dict) -> list[dict]:
    """
    拆分批次数据
    from: data_per_batch("sequences": [group_size, max_generate_len] ...)
    to:   [dict("sequences": [max_generate_len] ...) ... group_size]
    """
    # 定义需要拆分的键列表
    keys = (
        'sequence_ids',
        'old_log_probs',
        'ref_log_probs',
        'advantages',
        'attention_mask',
        'mask',
    )
    
    # group_size组内样本数
    # data_per_batch['sequence_ids']：[group_size, max_generate_len]
    group_size = data_per_batch['sequence_ids'].size(0)
    # 创建空列表存储单个样本的字典，长度为group_size
    group_data = [{} for _ in range(group_size)]
    
    # 遍历每个键，拆分对应的值到单个样本
    for key in keys:
        # 获取当前键对应的值
        value = data_per_batch[key]
        if value is None:
            # 如果值为None，创建长度为group_size的None列表
            vals = [None] * group_size
        else:
            # 将张量按第一维拆分（批次）为单个样本的张量列表
            vals = torch.unbind(value)

        # 将拆分后的值分配到每个样本的字典中
        for i, v in enumerate(vals):
            group_data[i][key] = v

    return group_data


def join_batch(batch_data: list[dict]) -> dict:
    """
    将单个样本的零散张量（可能长度不同）→ 收集为列表 → 检查有效性 → 填充对齐为相同长度 → 合并为统一形状的批次张量
    from: [dict("sequences": [max_generate_len] ...), ...]
    to:   dict("sequences": max_generate_len, ...)
    """
    # 创建空字典存储合并后的结果
    result = {}
    # 定义需要合并的键列表
    keys = (
        'sequence_ids',
        'old_log_probs',
        'ref_log_probs',
        'advantages',
        'attention_mask',
        'mask',
    )

    # 遍历每个键，合并对应的值为批次张量
    for key in keys:
        # 遍历batch_data（单个样本的字典列表）中的每个item（单个样本的字典），提取每个item中key对应的值（比如sequence_ids/old_log_probs等张量），最终将这些值收集为一个列表vals
        vals = [item[key] for item in batch_data]
        # vals列表中所有元素都不为None（即每个样本的该key都有有效张量值），避免对包含None的列表执行填充
        if all(v is not None for v in vals):
            # 执行零填充对齐（左侧填充）左填充不破坏有效 token 的连续性
            data = _zero_pad_sequences(vals, "left")
        else:
            # 如果有None值，合并后的值为None
            data = None
        # 将合并后的数据存入结果字典
        result[key] = data

    # 返回合并后的批次数据字典
    return result


def fill_loss_mask(loss_masks, labels):
    """
    将loss_mask中prompt部分强制设置为False
    loss_masks: shape  (B, T)
    labels: shape (B, T)
    """
    # 获取tokenizer实例
    tokenizer = TrainerTools().tokenizer
    # 支持多轮会话的mask → 遍历每个批次样本
    for batch, label in enumerate(labels):
        # 初始化prompt起始索引为-1（未开始）
        start_index = -1
        # 遍历当前样本的每个token
        for index, token in enumerate(label):
            # 如果token是system或user标识，标记为prompt起始位置
            if token == tokenizer.system or token == tokenizer.user:
                start_index = index
            # 如果token是end标识且已标记prompt起始位置
            elif token == tokenizer.end and start_index != -1:
                # 将prompt部分的loss_mask设为False（不计算loss）
                loss_masks[batch, start_index:index + 1] = False
                # 重置prompt起始索引
                start_index = -1
    return loss_masks


def left_pad_sequence(
    sequences: Union[torch.Tensor, List[torch.Tensor]],
    padding_value: float,
) -> torch.Tensor:
    """对序列进行左侧填充（left pad），手动实现"""
    # 如果已经是 Tensor (Batch, Seq)，直接返回
    if isinstance(sequences, torch.Tensor):
        return sequences

    # 获取最大长度
    max_len = max(seq.size(0) for seq in sequences)
    padded_seqs = []

    for seq in sequences:
        # 计算需要填充的长度
        pad_len = max_len - seq.size(0)
        # 使用 F.pad 进行填充 (pad_left, pad_right)
        # 注意：seq 需要是 Tensor
        padded_seq = F.pad(seq, (pad_len, 0), value=padding_value)
        padded_seqs.append(padded_seq)

    # 堆叠成 Batch 张量
    return torch.stack(padded_seqs)


def compute_token_losses(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    计算每个token的log probability
    logits：模型的原始输出，形状为 [B, S, V]，其中B是批次大小，S是序列长度，V是词汇表大小
    index：S中每个位置对应的正确token索引，形状为 [B, S]
    """
    # 1. 展平 logits: [B, S, V] -> [B*S, V]
    # 2. 展平 index:  [B, S]    -> [B*S]
    # 3. reduction='none':保留每个 token 的 loss，不求和
    nll_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        index.reshape(-1),#.reshape()，它可以自动处理非连续内存的数据
        reduction='none'
    )# 每个位置的单独损失值[B*S]
    ''' 
    logits = torch.tensor([
    [[1.0, 2.0, 3.0, 4.0],   # 样本1，token1
     [0.5, 1.5, 2.5, 3.5],   # 样本1，token2
     [1.2, 2.2, 3.2, 4.2]],  # 样本1，token3
     
    [[0.8, 1.8, 2.8, 3.8],   # 样本2，token1
     [1.1, 2.1, 3.1, 4.1],   # 样本2，token2
     [0.9, 1.9, 2.9, 3.9]]   # 样本2，token3
    ])

    # index: 真实标签
    index = torch.tensor([
        [2, 1, 0],  # 样本1的真实标签
        [3, 2, 1]   # 样本2的真实标签
    ])
    '''
    
    # 4. 恢复形状并取负号: Loss 是负对数似然，取负号变回 log probability
    return -nll_loss.reshape(index.shape)


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """仅基于有效数据（非 mask 部分）计算统计量（均值、方差）"""
    # 计算masked的均值和方差
    mean, var = _masked_mean(values, mask), _masked_var(values, mask)
    # 标准化：(值 - 均值) / 根号(方差 + 小常数)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    # 如果不需要平移均值，加回均值
    if not shift_mean:
        whitened += mean
    return whitened


def truncate_sequences_at_eos(
        sequences: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int
) -> torch.Tensor:
    """
    高效地将批处理中的序列在第一个EOS标记处截断（保留第一个EOS，mask其后的内容）。
    Args:
        sequences (torch.Tensor): (batch_size, seq_len)
        eos_token_id (int): EOS ID
        pad_token_id (int): PAD ID
    Returns:
        torch.Tensor: 截断并填充后的新序列
    """
    # 标记所有 EOS 的位置 (B, L) -> Bool
    eos_mask = (sequences == eos_token_id)

    # 计算累积和，用于检测“当前是否已经出现过 EOS”
    # 如果序列是 [0, 0, 1, 0, 1] (1代表EOS)
    # cumsum后: [0, 0, 1, 1, 2]
    # 凡是 cumsum > 0 的地方，说明前面（或当前）已经遇到过 EOS 了
    eos_cumsum = eos_mask.cumsum(dim=1)

    # 我们希望保留第一个 EOS，但填充它后面的所有内容。
    # 逻辑推导：
    # - EOS前 (0, 0): cumsum=0, mask=0 -> 0 > 0 (False, 保留)
    # - 第1个EOS (1, 1): cumsum=1, mask=1 -> 1 > 1 (False, 保留)
    # - EOS后非EOS (1, 0): cumsum=1, mask=0 -> 1 > 0 (True, 填充)
    # - 第2个EOS (2, 1): cumsum=2, mask=1 -> 2 > 1 (True, 填充)
    # 利用 PyTorch 的自动广播，这里会隐式将 bool 转为 int/long 进行比较
    pad_mask = eos_cumsum > eos_mask.to(eos_cumsum.dtype)

    # 执行填充（使用 masked_fill 创建新张量，保持函数式纯粹性）
    return sequences.masked_fill(pad_mask, pad_token_id)


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    """禁用模型中的所有Dropout层（用于推理或评估阶段）"""
    # 遍历模型的所有模块
    for module in model.modules():
        # 如果模块是Dropout层
        if isinstance(module, torch.nn.Dropout):
            # 禁用Dropout
            module.p = 0


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """仅基于有效数据（非 mask 部分）计算均值"""
    if axis is not None:
        # 沿指定轴计算：(值*mask)的和 / mask的和
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        # 全局计算：(值*mask)的总和 / mask的总和
        return (values * mask).sum() / mask.sum()


def _masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """仅基于有效数据（非 mask 部分）计算方差"""
    # 仅基于有效数据（非 mask 部分）计算均值
    mean = _masked_mean(values, mask)
    # 计算中心化的值（值 - 均值）
    centered_values = values - mean
    # 仅基于有效数据（非 mask 部分）计算方差（未校正）
    variance = _masked_mean(centered_values**2, mask)
    # 如果需要无偏估计（贝塞尔校正）
    if unbiased:
        # mask 是一个布尔张量，sum() 统计了有多少个 True (即有效数据 n)
        mask_sum = mask.sum()
        if mask_sum == 0:
            # 如果mask总和为0，抛出异常（通常是batch_size=1导致）
            raise ValueError(
                "mask总和为0, 通常是batch_size=1导致"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # 数学推导： (Sum / n) * (n / n-1) = Sum / (n-1)
        # 巧妙地把分母里的 n 消掉，换成了 n-1
        bessel_correction = mask_sum / (mask_sum - 1)
        # 应用校正：
        # 原有方差 * 校正因子 = 无偏方差
        variance = variance * bessel_correction
    # 返回方差
    return variance


def _selective_log_softmax(logits, index) -> torch.Tensor:
    """
    计算指定 token 的 log_softmax 值。
    """
    # CrossEntropyLoss 等价于 -log(p(x))
    # 所以：log(p(x)) = -CrossEntropyLoss
    
    return -F.cross_entropy(
        input=logits.view(-1, logits.size(-1)), # [Batch*Seq, Vocab]
        target=index.view(-1),                  # [Batch*Seq]
        reduction='none'                        # 关键：保留每个位置的 Loss，不求和
    ).view(index.shape)                         # 恢复形状 [Batch, Seq]


def _mask_prompt(labels):
    """将labels中的prompt部分设为-100（不计算loss）"""
    # 获取tokenizer实例
    tokenizer = TrainerTools().tokenizer
    # 支持多轮会话的mask → 遍历每个批次样本
    for batch, label in enumerate(labels):
        # 初始化prompt起始索引为-1（未开始）
        start_index = -1
        # 遍历当前样本的每个token
        for index, token in enumerate(label):
            # 如果token是system或user标识，标记为prompt起始位置
            if token == tokenizer.system or token == tokenizer.user:
                start_index = index
            # 如果token是end标识且已标记prompt起始位置
            elif token == tokenizer.end and start_index != -1:
                # 将prompt部分的labels设为-100（忽略）
                labels[batch, start_index:index + 1] = -100
                # 重置prompt起始索引
                start_index = -1

    # 返回处理后的labels
    return labels


def _zero_pad_sequences(
    sequences: list[torch.Tensor], side: str = "left"
) -> torch.Tensor:
    """对序列列表进行零填充（支持左侧或右侧填充）"""
    # 断言side必须是left或right
    assert side in ("left", "right")
    # 计算序列列表中的最大长度
    max_len = max(seq.size(0) for seq in sequences)
    # 存储填充后的序列
    padded_sequences = []
    # 遍历每个序列
    for seq in sequences:
        # 计算需要填充的长度
        pad_len = max_len - seq.size(0)
        # 确定填充方式：左侧填充为(pad_len, 0)，右侧填充为(0, pad_len)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        # 对序列进行填充
        padded_sequences.append(F.pad(seq, padding))
    # 将填充后的序列堆叠为批次张量
    return torch.stack(padded_sequences, dim=0)
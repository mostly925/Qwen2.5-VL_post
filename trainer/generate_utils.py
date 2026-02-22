from typing import Union, Optional, List, Any, Dict
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from .tools import TrainerTools
from .utils import (
    autocast,
    batch_repeat_image_tok
)
from qwen_vl_utils import process_vision_info


def _suppress_warper(logits: torch.Tensor, suppress_tokens: List[int]) -> torch.Tensor:
    """
    抑制特殊token输出
    logits: 模型输出的logits
    suppress_tokens: 需要抑制的token ID列表
    """
    suppress_tokens = torch.tensor(suppress_tokens, device=logits.device)
    # 创建一个包含所有词表索引的tensor [0, 1, ..., vocab_size-1]
    vocab_tensor = torch.arange(logits.shape[-1], device=logits.device)
    # 判断vocab_tensor中的每个索引是否是suppress_tokens中的值，生成布尔掩码，True表示需要抑制：tensor([False, False,  True, False, False,  True, False,  True, False, False])
    suppress_token_mask = torch.isin(vocab_tensor, suppress_tokens)
    # 使用掩码将掩码为True的位置的 logits 设为负无穷（这样 softmax 后概率趋近于 0，不会被选中）
    logits = torch.where(suppress_token_mask, -float("inf"), logits)

    return logits


def _temperature_warper(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    应用temperature 温度参数
    """
    # 将logits除以温度系数。温度>1会平滑概率分布（增加随机性），温度<1会尖锐化分布（减少随机性）
    logits = logits / temperature
    return logits


def _top_k_warper(logits: torch.Tensor, k: int, device: Union[str, torch.device, int] = None) -> torch.Tensor:
    """
    top k采样 保留概率最大的k个token
    """
    # 获取logits中最大的k个值及其索引，维度为 [batch, k]
    topk_logits, _ = torch.topk(logits, k=k)
    # 取每个 batch 中 top-k 的最后一个值（即每个 batch 里 top-k 的最小值），作为阈值。
    min_val= topk_logits[:, -1]
    # 将logits中小于阈值的部分设置为负无穷大，大于等于阈值的部分保持不变
    # logits形状[batch_size, vocab_size]，min_val.unsqueeze(-1) 是为了广播维度以匹配logits形状
    logits = torch.where(logits < min_val.unsqueeze(-1), torch.tensor(-torch.inf).to(device), logits)
    return logits


def _top_p_warper(logits: torch.Tensor, p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
    """
    top p 核采样  累积概率阈值
    :param min_tokens_to_keep: 最少保留的token数量
    """
    # 对logits进行升序排列。sorted_logits: 排序后的值, sorted_indices: 对应的原始索引
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=False)
    
    # 累积概率
    # 计算softmax概率，并进行累积求和 (cumsum)
    # 结果类似 [0.01, 0.05, ..., 0.99, 1.0]
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    
    # 创建移除掩码：如果累积概率 <= (1-p)，则标记为True（表示这些低概率的词需要被移除）
    # 注意：因为是升序排列，所以我们切掉的是左边（低概率）的部分
    # True表示该位置的 token 要移除，False保留
    sorted_indices_to_remove = cumulative_probs <= (1 - p)
    
    # 将掩码的最后几位设为0 (False)：强制保留最后 min_tokens_to_keep 个token（即概率最大的那几个），防止全部被mask掉
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0
    
    # .scatter：带着sorted_indices_to_remove回到原始sorted_indices索引的位置
    indices_to_remove = sorted_indices_to_remove.scatter(1, index=sorted_indices, src=sorted_indices_to_remove)

    # 使用掩码将需要移除的token的logits值设为负无穷大
    scores_processed = logits.masked_fill_(indices_to_remove, -float("Inf"))

    return scores_processed


def _generate(
        model: torch.nn.Module,
        *,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: Optional[float],
        k: Optional[int],
        p: Optional[float],
        pixel_values: Optional[torch.Tensor] = None,
        tokens_per_image: int = -1,
        suppress_tokens: Optional[List[int]] = None,
        device: Union[str, torch.device, int]
):
    use_kv_cache = True  # 默认启用KV Cache加速推理,如果为False则生成速度会变慢很多，但能省下大量显存（GB级别）

    # 如果是1维则升维：确保输入维度是 [Batch, Seq]
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    # 创建全1的attention_mask，形状与tokens相同
    attention_mask = torch.ones_like(tokens, device=device, dtype=torch.long)

    # 如果是视觉语言模型(VlmModel)，需要处理图像占位符
    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        # 将图像token在序列中进行重复/扩展以匹配embedding层需求
        # 模型的图像编码输出的图像特征通常对应固定长度的 token 序列（比如 ViT-L/14 输出576个图像 patch token）。
        # 而文本序列中原本只有1 个占位 token 比如<image>，无法匹配图像特征的长度 —— 因此需要把这个占位 token重复多次
        tokens = batch_repeat_image_tok(tokens, tokens_per_image)

    # 如果有图像输入，将其移动到指定设备
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)

    # 初始化KV Cache的值为None
    kv_cache: Optional[Any] = None
    # 复制一份tokens用于存储完整的生成结果序列
    generate_tokens = tokens.clone()

    # 开启推理模式，禁用梯度计算，节省显存和计算
    with torch.inference_mode():
        # 循环生成每一个新的token
        for _ in range(max_new_tokens):
            # 当前输入的tokens（如果是KV Cache模式，这里后续只会是最新生成的那个token）
            t = tokens
            # 使用自动混合精度上下文（如fp16/bf16）
            with autocast(TrainerTools().parallel.device_type):
                # 模型前向传播
                result = model(
                    t,
                    attention_mask=attention_mask,
                    past_key_values=kv_cache,
                    use_cache=use_kv_cache,
                    pixel_values=pixel_values
                )

                # 获取输出的logits，形状通常为 (batch, seq_len, vocab_size)
                logits = result['logits']
                # 更新KV Cache，用于下一次迭代
                kv_cache = result['past_key_values']

            # 打字机模式：提取每个样本最后一个时间步的所有词表 logits  形状(batch, vocab_size)
            logits = logits[:, -1, :]

            # 如果设置了抑制token，调用warper处理
            if suppress_tokens and len(suppress_tokens) != 0:
                logits = _suppress_warper(logits, suppress_tokens)

            # 标记是否需要随机采样
            multinomial = False
            # 如果温度参数有效，应用温度warper并标记启用采样
            if temperature and temperature > 0:
                multinomial = True
                logits = _temperature_warper(logits, temperature)

            # 如果Top-K参数有效，应用Top-K warper
            if k and k != 0:
                logits = _top_k_warper(logits, k, device)

            # 如果Top-P参数有效，应用Top-P warper
            if p and 0 < p <= 1:
                logits = _top_p_warper(logits, p)

            # 根据策略选择下一个token
            # 多项式采样：带随机性（生成多样文本）
            if multinomial:
                # 将 logits 转换为0~1 之间的概率值
                prob = logits.softmax(dim=-1)
                # 从prob的概率分布中，随机抽取num_samples=1个样本，返回的是样本对应的下标（token id）
                # 概率越高的 token 被选中的概率越大，但不是必然选中
                next_token = torch.multinomial(prob, num_samples=1)
            else:
                # 贪婪搜索：直接选择概率最大
                # keepdim=True：保持维度不变 —— 确保输出形状和多项式采样一致（[batch_size, 1]），方便后续拼接序列
                next_token = logits.argmax(dim=-1, keepdim=True)

            # 生成器：当前的token和是否结束的标志(False)
            yield next_token, False

            # 更新给大模型的输入序列
            if use_kv_cache:
                # 如果使用KV Cache，下一次输入只需要是最新的token
                tokens = next_token
                # 将新token拼接到完整序列中
                generate_tokens = torch.cat((generate_tokens, next_token), dim=-1)
            else:
                # 如果不使用KV Cache，输入是累积的完整序列
                tokens = torch.cat((tokens, next_token), dim=-1)

            # 在生成新 token 时，同步扩展 attention mask 的长度，确保新生成的 token 位置被标记为 “有效”
            # (tokens.shape[0], 1)  构造形状为[batch_size, 1]的张量
            new_mask_bit = torch.ones((tokens.shape[0], 1), device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, new_mask_bit), dim=-1)

            # 检查是否生成了结束符（EOS token）
            if next_token.item() == TrainerTools().tokenizer.end:
                break

    # 循环结束，yield完整的生成序列和结束标志(True)
    # 根据是否使用cache返回对应的完整tensor
    yield tokens if not use_kv_cache else generate_tokens, True


def _streaming_generate(
        model: torch.nn.Module,
        *,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int,
        temperature: Optional[float] = 1.0,
        k: Optional[int] = None,
        p: Optional[float] = None,
        pixel_values: Optional[torch.Tensor] = None,
        tokens_per_image: int = -1,
        suppress_tokens: Optional[List[int]] = None,
        device: Union[str, torch.device, int] = None
):
    """
    内部流式生成包装器，处理输入编码
    """
    # 确定设备，如果未指定则从工具类获取
    device = TrainerTools().parallel.device if not device else device

    # 处理输入Prompt
    if isinstance(prompt, torch.Tensor):
        # 如果已经是Tensor，直接移至设备
        encoded_tokens = prompt.to(device)
    else:
        # 如果是字符串，进行分词编码，增加batch维度，并转换为Tensor移至设备
        encoded_tokens = TrainerTools().tokenizer.encode(prompt, unsqueeze=True, covert_tensor=True).to(device)

    # 调用核心生成器 _generate
    generate_text_iterator = _generate(
        model=model,
        tokens=encoded_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        k=k,
        p=p,
        pixel_values=pixel_values,
        tokens_per_image=tokens_per_image,
        suppress_tokens=suppress_tokens,
        device=device
    )

    # 迭代生成器结果并向外yield
    for (token, is_full_result) in generate_text_iterator:
        yield token, is_full_result


def streaming_generate(
        model: torch.nn.Module,
        *,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int,
        temperature: Optional[float] = 1.0,
        k: Optional[int] = None,
        p: Optional[float] = None,
        pixel_values: Optional[torch.Tensor] = None,
        tokens_per_image: int = -1,
        suppress_tokens: Optional[List[int]] = None,
        device: Union[str, torch.device, int] = None,
        return_token: bool = False
):
    """
    对外流式生成接口，处理解码逻辑
    :param return_token: 是否直接返回token ID而不是解码后的文本
    """
    # 调用内部流式生成器
    text_iterator = _streaming_generate(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        k=k,
        p=p,
        pixel_values=pixel_values,
        tokens_per_image=tokens_per_image,
        suppress_tokens=suppress_tokens,
        device=device
    )

    # 遍历生成结果
    for (token, is_full_result) in text_iterator:
        # 只处理中间生成的单个 token（流式输出的核心），忽略最终的完整序列结果
        if not is_full_result:
            # 返回原始 token ID
            if return_token:
                # 去掉token的batch 维度（因为生成的 token 通常带batch_size=1的维度）
                yield token.squeeze(0)
            # 返回人类可读的文本
            else:
                # 将token解码为字符串并返回
                yield TrainerTools().tokenizer.decode(token.squeeze(0))


def generate(
        model: torch.nn.Module,
        *,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int,
        temperature: Optional[float] = 1.0,
        k: Optional[int] = None,
        p: Optional[float] = None,
        pixel_values: Optional[torch.Tensor] = None,
        tokens_per_image: int = -1,
        suppress_tokens: Optional[List[int]] = None,
        device: Union[str, torch.device, int] = None,
        return_token: bool = False
):
    """
    非流式生成接口，等待所有生成完成后一次性返回
    """
    # 调用内部流式生成器
    text_iterator = _streaming_generate(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        k=k,
        p=p,
        suppress_tokens=suppress_tokens,
        pixel_values=pixel_values,
        tokens_per_image=tokens_per_image,
        device=device
    )

    # 遍历生成器
    for (token, is_full_result) in text_iterator:
        # 只有当 is_full_result 为 True 时才处理（即生成结束）
        if is_full_result:
            if return_token:
                # 返回完整的token序列
                return token.squeeze(0)
            else:
                # 返回解码后的完整文本
                return TrainerTools().tokenizer.decode(token.squeeze(0))

    return None  # 如果没有结果返回None


def batch_generate(
        model: torch.nn.Module,
        *,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: Optional[float] = None,
        k: Optional[int] = None,
        p: Optional[float] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        tokens_per_image: int = -1,
        suppress_tokens: Optional[List[int]] = None,
        device: Union[str, torch.device, int],
        **kwargs
):
    """
    批量生成：支持同时处理多个序列
    """
    use_kv_cache = True  # 启用KV Cache
    end_token = TrainerTools().tokenizer.end  # 获取结束符ID
    pad_token_id = TrainerTools().tokenizer.pad  # 获取填充符ID

    # VLM模型处理：扩展图像token
    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        tokens = batch_repeat_image_tok(tokens, tokens_per_image)

    # 移动图像数据到设备
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)

    # 备份原始输入tokens
    orig_tokens = tokens.clone()
    # 克隆attention_mask，因为后续会修改它
    full_attention_mask = attention_mask.clone()

    # 初始化KV Cache的值
    kv_cache: Optional[Any] = None
    # 获取当前batch大小
    batch_size = tokens.shape[0]

    # 预分配最大长度的buffer，避免循环中反复cat造成内存碎片和性能下降
    # 初始化为pad_token_id
    generated_tokens_buffer = torch.full(
        (batch_size, max_new_tokens),
        pad_token_id,
        dtype=torch.long,
        device=device
    )

    # 记录当前 Batch 里，每一句话是否已经生成完毕。   False：表示这句话还没结束    True：表示这句话已经生成了结束符（<EOS>），生成完毕
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    # 当前输入tokens（初始为用户输入）
    current_tokens = tokens

    # 用于存储所有步的logits的变量（可选）
    padded_logits = None
    # 记录实际生成的长度
    actual_gen_len = 0

    # 把一个普通的 pad_token_id 转换成一个在指定设备上的 Tensor
    pad_token_tensor = torch.tensor(pad_token_id, device=device, dtype=torch.long)

    # 开启推理模式
    with torch.inference_mode():
        for i in range(max_new_tokens):
            # 如果所有样本都已完成生成，提前退出循环
            if done.all():
                break

            # 更新实际生成长度
            actual_gen_len = i + 1

            # 确保输入类型为long
            if current_tokens.dtype != torch.long:
                current_tokens = current_tokens.long()

            # 混合精度上下文
            with autocast(TrainerTools().parallel.device_type):
                # 模型前向传播
                result = model(
                    current_tokens,
                    attention_mask=full_attention_mask,
                    past_key_values=kv_cache,
                    use_cache=use_kv_cache,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw
                )
                logits = result['logits']
                kv_cache = result['past_key_values']

            # 取最后一个token的logits
            logits = logits[:, -1, :]

            # 如果需要记录logits，首次进行初始化
            if padded_logits is None:
                # 获取词表大小
                vocab_size = logits.shape[-1]
                # 申请一块巨大的显存空间
                # 形状是: [Batch数, 最大生成长度, 词表大小]
                padded_logits = torch.zeros(
                    (batch_size, max_new_tokens, vocab_size),
                    dtype=logits.dtype,
                    device=device
                )

            # 保存当前步的logits
            padded_logits[:, i, :] = logits

            # 应用各种warper进行采样控制
            if suppress_tokens:
                logits = _suppress_warper(logits, suppress_tokens)

            multinomial = False
            if temperature and temperature > 0:
                multinomial = True
                logits = _temperature_warper(logits, temperature)
            if k and k != 0:
                logits = _top_k_warper(logits, k, device)
            if p and 0 < p <= 1:
                logits = _top_p_warper(logits, p)

            # 采样下一个token
            if multinomial:
                prob = logits.softmax(dim=-1)
                next_token_active = torch.multinomial(prob, num_samples=1)
            else:
                next_token_active = logits.argmax(dim=-1, keepdim=True)

            # 关键逻辑：如果该样本已经done，则强制填入pad token，否则填入采样得到的token
            # where(condition, x, y) -> condition为True选x，否则选y
            # done.unsqueeze(1)————>升维[Batch, 1]即[True, False, False]变[[True],
                                                                       # [False],
                                                                       # [False]]
            next_token = torch.where(
                done.unsqueeze(1),
                pad_token_tensor,
                next_token_active
            )

            # 将生成的token存入预分配的buffer中
            generated_tokens_buffer[:, i] = next_token.squeeze(-1)

            # 更新done状态：如果当前生成的token是EOS，则标记为True
            new_done = (next_token.squeeze(-1) == end_token)
            # 只要你之前结束过，或者你现在刚结束，你都算结束
            # True | False = True
            done = done | new_done

            # 更新current_tokens为当前生成的token，供下一步KV Cache使用
            current_tokens = next_token

            # 更新attention_mask
            # 只有未完成（~done逻辑取反）的样本，新位置的mask才是1，已完成的是0（即mask掉pad token）
            new_mask = (~done).long().to(full_attention_mask.dtype)
            # 将新mask拼接到full_attention_mask后面
            full_attention_mask = torch.cat((full_attention_mask, new_mask.unsqueeze(-1)), dim=-1)

    # 截取有效的生成结果（去掉预分配多余的部分）
    final_generated_tokens = generated_tokens_buffer[:, :actual_gen_len]

    # 截取有效的logits
    if padded_logits is not None:
        final_padded_logits = padded_logits[:, :actual_gen_len, :]
    else:
        final_padded_logits = None

    # 将原始输入和生成结果拼接，形成完整的序列
    final_full_sequences = torch.cat((orig_tokens, final_generated_tokens), dim=1)

    # 返回完整序列和对应的logits
    return final_full_sequences, final_padded_logits


def generate_with_messages(
        model: torch.nn.Module,
        processor: Any,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 1.0,
        device: Union[str, torch.device, int] = None
):
    """
    使用 Qwen2.5-VL 的官方推荐方式进行生成
    """
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    if device is None:
        device = model.device
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]
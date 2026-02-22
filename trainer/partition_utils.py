from typing import Optional
from contextlib import contextmanager  # 从 contextlib 导入 contextmanager，用于创建上下文管理器（即支持 with 语句）
import itertools  # 导入 itertools 模块，用于处理迭代器（如链接多个参数列表）
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist  # 导入 PyTorch 的分布式通信模块

from .tools import TrainerTools
from .parallel import DsParallel, DdpParallel


@contextmanager
def unwrap_model_for_generation(model: nn.Module):
    """
    用于在生成任务中解包模型。
    修改说明：针对 ZeRO-3 + 低显存场景，禁止 GatheredParameters 全量收集参数，
    而是直接返回 model (DeepSpeedEngine)，利用其 Forward 时的自动参数获取机制。
    """
    if isinstance(TrainerTools().parallel, DsParallel):
        # 你的环境是 DeepSpeed
        import deepspeed
        assert isinstance(model, deepspeed.DeepSpeedEngine)

        # 【核心修改】：注释掉原本的 GatheredParameters 逻辑
        # ZeRO-3 显存不够时，千万不能全量 Gather！
        # 直接 yield model，后续的 batch_generate 调用 model() 时，
        # DeepSpeed 会自动一层层加载参数，算完一层释放一层，从而不爆显存。
        
        # 移除钩子会导致参数无法自动获取，所以这里也不能移除钩子
        # _remove_hooks(model) 
        
        # 直接返回原始 model (DeepSpeedEngine)
        yield model 
        
        # _add_hooks(model)

    elif isinstance(TrainerTools().parallel, DdpParallel):
        yield unwrap_model(model)
    else:
        yield model


def sync_model_params(_from: nn.Module, _to: Optional[nn.Module], mixup_alpha: float = 1.0):
    """
        必须在所有rank上调用，非rank0, _to 可以设置为None.
        功能：将 _from 模型的参数同步到 _to 模型。支持直接覆盖或加权平均（Mixup/EMA）。
    """
    # 1. 获取源模型 (_from) 的参数状态字典 (state_dict)
    if isinstance(TrainerTools().parallel, DsParallel):
        # ZeRO-3 的参数切分在各个显卡上的。要获取完整的参数 (state_dict)，必须在 Rank 0 上重组，Rank 0 把重组好的完整参数通过网络发送给其他所有 Rank
        # 从源模型 _from 提取参数，如果当前没有目标模型（_to 为 None），则只需要在主进程（Rank 0）保留参数即可
        # only_rank0：是否只在 Rank 0 上保留数据
        state_dict = _get_ds_model_params(_from, only_rank0=_to is None)
    elif isinstance(_from, DDP):
        # 如果是 DDP 模式，通过 .module 获取内部模型的 state_dict
        state_dict = _from.module.state_dict()
    else:
        # 普通模型直接获取 state_dict
        state_dict = _from.state_dict()

    
    # 2. 将参数加载到目标模型
    # 如果目标模型不存在（如在非主进程上）或者没有获取到 state_dict，直接返回
    if not _to or not state_dict:
        return
    # 解包目标模型
    unwrap_to_model = unwrap_model(_to)# 得到原始 nn.Module
    
    # 如果混合比例为 1.0，表示完全替换，直接加载状态字典
    if mixup_alpha == 1.0:
        # strict=False 允许 state_dict 和模型结构有细微差异
        unwrap_to_model.load_state_dict(state_dict, strict=False)
    else:
        # 混合参数（通常用于 EMA - 指数移动平均）
        # 遍历目标模型的所有参数
        for param_name, target_param in unwrap_to_model.named_parameters():
            # 只有当参数名在源 state_dict 中存在时才更新
            if param_name in state_dict:
                from_param_tensor = state_dict[param_name]  # 获取源参数
                # 执行原地更新操作：target = target * (1 - alpha) + source * alpha
                # mul_ 是原地乘法
                target_param.data.mul_(1.0 - mixup_alpha).add_(
                    from_param_tensor.data.to(target_param.device),  # 确保设备相同
                    alpha=mixup_alpha  # add_ 函数的 alpha 参数，表示加法前的缩放系数
                )

def unwrap_model(model) -> nn.Module:
    """
    通用函数：剥离各种包装器（DeepSpeed, DDP），返回原始的 nn.Module
    """
    try:
        import deepspeed
        if isinstance(model, deepspeed.DeepSpeedEngine):
            return model.module  # 如果是 DeepSpeedEngine，返回其内部的 .module
    except: ...  # 如果导入失败或出错，忽略，继续检查其他类型

    if isinstance(model, DDP):
        return model.module  # 如果是 DDP 模型，返回其内部的 .module

    return model  # 如果都没有包装，直接返回原对象


def _get_ds_full_state_dict_on_rank0(model: nn.Module) -> Optional[dict]:
    """
        必须在所有rank上调用，虽然只有rank0有值，但其他rank的逻辑也必须有，不然Rank之间不会传递信息
        用于处理 DeepSpeed 模型，特别是 ZeRO-3 模式下的参数收集。
        CPU便宜，把参数深拷贝到CPU
    """
    import deepspeed
    assert isinstance(model, deepspeed.DeepSpeedEngine)  # 断言必须是 DeepSpeed 引擎

    # 如果不是 ZeRO-3 阶段（即 ZeRO-0/1/2）
    if model.zero_optimization_stage() != 3:
        # 如果是主进程 (Rank 0)
        if TrainerTools().parallel.is_main_process:
            # 直接克隆参数到 CPU 并返回字典
            return {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
        return None  # 非主进程返回 None

    # --- ZeRO-3 处理逻辑 ---
    # ZeRO-3 参数被切分。GatheredParameters 上下文管理器负责从所有 rank 收集参数
    # modifier_rank=0 表示参数将被收集并重组到 Rank 0 上
    with deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=0):
        # 检查是否为主进程
        if TrainerTools().parallel.is_main_process:
            # 在这个 'with' 代码块内，rank 0 上的 model.module 拥有完整的参数
            # 所以我们可以像操作普通模型一样直接调用 state_dict()
            full_state_dict = model.module.state_dict()

            # 将其克隆到 CPU 并返回
            return {k: v.cpu().clone() for k, v in full_state_dict.items()}

    # 其他 rank 执行到这里时，上下文结束，直接返回 None
    return None


def _get_ds_model_params(model: nn.Module, only_rank0=False):
    """
        从一个正在运行的 DeepSpeedEngine 中高效地提取完整的 FP32 state_dict，
        兼容 ZeRO Stages 0, 1, 2, 3。
    """

    import deepspeed
    assert isinstance(model, deepspeed.DeepSpeedEngine)  # 断言类型
    # 调用辅助函数，在 Rank 0 上获取完整的 state_dict
    state_dict = _get_ds_full_state_dict_on_rank0(model)

    # 现在，只有 rank 0 上的 state_dict 是一个有效的字典，其他 rank 上是 None。
    # 逻辑判断：如果不只是需要在 rank0 上有数据 (即所有 rank 都需要)，且多卡环境
    if not only_rank0 and TrainerTools().parallel.world_size > 1:
        # 放在列表里方便传输
        object_list = [state_dict] if TrainerTools().parallel.is_main_process else [None]
        '''
        rank0——object_list：[{参数字典...}]
        rank1——object_list：[None]
        rank2——object_list：[None]
        .......
        '''
        # 调用PyTorch 的分布式通信模块dist广播
        # 阻塞：将 src=0 (Rank 0) 的数据发送给所有进程的object_list
        # 注意：这涉及将整个模型权重序列化并网络传输，开销较大
        dist.broadcast_object_list(object_list, src=0)
        # 每个Rank都会执行state_dict = object_list[0]去列表化，所有进程都有了state_dict，并且数据相同：整个模型权重
        state_dict = object_list[0]

    return state_dict  # 返回参数字典

def _add_hooks(model: nn.Module) -> None:
    """恢复 DeepSpeed ZeRO-3 模型的优化器钩子（用于在推理后恢复训练状态）。"""
    import deepspeed
    assert isinstance(model, deepspeed.DeepSpeedEngine)

    if not hasattr(model, "optimizer"):
        return  # 如果模型还没有优化器（训练开始前），直接返回
    # ZeRO-Offload：开启     优化器的状态和计算会被移到 CPU 上，省显存，此时 model.optimizer 是一个“壳”，真正干活的核心是optimizer.parameter_offload
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    # ZeRO-Offload：不开启    优化器的状态和计算都在 GPU 上
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("不支持无优化器") # 报错：不支持无优化器的情况
    
    # 把钩子装回去
    # _register_deepspeed_module：遍历这个模型的所有层，给每一层注册Forward Pre-Hook，Forward Hook，Backward Hook
    optimizer_offload._register_deepspeed_module(optimizer_offload.module)# optimizer_offload.module：优化器对象里存着的要优化的那个模型的引用
    


def _remove_hooks(model: nn.Module) -> None:
    """移除 DeepSpeed ZeRO-3 模型的优化器钩子（用于让模型可以进行正常的 forward 推理，而不触发参数切分逻辑）。"""
    import deepspeed
    assert isinstance(model, deepspeed.DeepSpeedEngine)  # 断言模型类型

    if not hasattr(model, "optimizer"):
        return  # 如果没有优化器，无需移除，直接返回
    # 获取优化器 offload 对象，同 _add_hooks
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("不支持无优化器")

    # 遍历模块的所有参数（包括递归子模块）
    for param in _iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()  # 清除 DeepSpeed 记录的活跃子模块状态

    # 移除前向传播钩子
    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    # 移除反向传播钩子
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    # 清空钩子列表，防止残留引用
    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def _iter_params(module, recurse=False):
    """
    辅助函数：获取模块参数列表的值
    """
    return [param for _, param in _get_all_parameters(module, recurse)]


def _get_all_parameters(sub_module, recurse=False):
    """
    辅助函数：获取所有参数，包括 DeepSpeed 可能存在的外部参数
    """
    # sub_module.named_parameters(recurse=recurse)：返回模型里所有注册过的标准参数，生成器，每次生成一个元组 (参数名, 参数张量)。例如：("layer1.weight", Tensor(...))
    # recurse=True  递归不仅查当前这个模块的参数，还会深入查它的子模块、……直到把这棵树下所有的参数都找出来
    # sub_module.ds_external_parameters()：返回那些 被 DeepSpeed 特殊管理、没有在 PyTorch 标准名单里注册的参数，比如MoE 里的专家参数
    # itertools.chain拼接两个可迭代对象
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())
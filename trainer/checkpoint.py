import os
from typing import Optional, Union
# 导入文件操作模块
import shutil
import torch
from torch import nn
from torch.optim import Optimizer
# 导入分布式数据并行包装器
from torch.nn.parallel import DistributedDataParallel as DDP

# 导入 safetensors
from safetensors.torch import save_file, load_file

# 导入DeepSpeed并行类
from .parallel import DsParallel
# 导入学习率调度器
from .scheduler import LRScheduler
# 导入训练工具类
from .tools import TrainerTools

# 默认的checkpoint文件名，修改为 safetensors
DEFAULT_CHECKPOINT_NAME = "checkpoint.safetensors"

def save_checkpoint(
        model: nn.Module,  # 要保存的模型
        optimizer: Optional[Optimizer] = None  # 可选的优化器
):
    """
    保存模型检查点 (Safetensors 格式)
    注意：Safetensors 主要用于保存 Tensor 数据，不建议混合保存复杂的 Python 对象（如 Optimizer 状态）。
    通常对于大模型，建议将模型权重保存为 safetensors，而 Optimizer 状态等可以使用 torch.save 单独保存或忽略（如果只用于推理）。
    如果是为了恢复训练，建议将 Optimizer 状态保存为单独的 .pt 文件。
    """
    # 检查是否使用DeepSpeed并行
    if isinstance(TrainerTools().parallel, DsParallel):
        # 导入DeepSpeed的checkpoint保存函数
        from .ds_checkpoint import save_ds_checkpoint
        # 使用DeepSpeed的保存方法 (DeepSpeed 内部管理格式，通常是分片的 .pt 或 safetensors)
        save_ds_checkpoint(model)
    else:
        # 只在主进程中保存checkpoint
        if TrainerTools().parallel.is_main_process:
            # 从环境变量获取checkpoint文件名，如果未设置则使用默认值
            checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)
            
            # 模型未被DDP包装
            if not isinstance(model, DDP):
                raw_model = model
            # 模型被DDP包装，原始模型存储在DDP实例的.module属性中
            else:
                raw_model = model.module
            
            # 获取模型状态字典
            state_dict = raw_model.state_dict()
            
            # 保存模型权重为 safetensors
            # 如果文件名以 .pt 或 .bin 结尾，强制改为 .safetensors (除非用户特别指定)
            if checkpoint_name.endswith('.pt') or checkpoint_name.endswith('.bin') or checkpoint_name.endswith('.pth'):
                checkpoint_name = os.path.splitext(checkpoint_name)[0] + ".safetensors"
                
            save_file(state_dict, checkpoint_name)
            
            # 如果提供了优化器，则将优化器状态保存为单独的 .pt 文件 (safetensors 不支持保存 optimizer state_dict 这种复杂结构)
            if optimizer:
                optim_checkpoint_name = os.path.splitext(checkpoint_name)[0] + "_optim.pt"
                torch.save(optimizer.state_dict(), optim_checkpoint_name)


def save_best_checkpoint(
        current_loss: float,  # 当前损失值
        last_best_checkpoint_loss: Optional[float] = None  # 上次最佳checkpoint的损失值
) -> bool:
    """
    保存最佳检查点（基于损失值的最小值）
    """
    # 检查环境变量，如果指定不保存最佳checkpoint则直接返回
    if os.environ.get('SAVE_BEST_CHECKPOINT', '1') != '1':
        return False

    # 判断是否需要替换：如果还没有历史最佳损失，或者当前损失更好/相等
    if last_best_checkpoint_loss is None or current_loss <= last_best_checkpoint_loss:
        need_replace = True
    else:
        need_replace = False
    # 如果需要替换且当前是主进程
    if need_replace and TrainerTools().parallel.is_main_process:
        try:
            # 检查是否使用DeepSpeed并行
            if isinstance(TrainerTools().parallel, DsParallel):
                # 获取分布式checkpoint目录
                checkpoint_dir = os.environ.get('DIST_CHECKPOINT_DIR', 'checkpoint')
                # 构建最佳checkpoint目录名
                if checkpoint_dir.endswith('/'):# 如果目录名以/结尾，去掉/后添加_best后缀
                    best_checkpoint_dir = f'{checkpoint_dir[:-1]}_best'
                else:
                    # 否则直接添加_best后缀
                    best_checkpoint_dir = f'{checkpoint_dir}_best'
                # 如果best_checkpoint_dir目录不存在，则创建
                if not os.path.exists(best_checkpoint_dir):
                    os.makedirs(best_checkpoint_dir)
                # 如果原checkpoint目录存在
                if os.path.exists(checkpoint_dir):
                    # 删除旧的最佳checkpoint目录
                    if os.path.exists(best_checkpoint_dir):
                        shutil.rmtree(best_checkpoint_dir)
                    # 复制当前checkpoint目录到最佳checkpoint目录
                    shutil.copytree(checkpoint_dir, best_checkpoint_dir)
            else:
                # 非DeepSpeed模式下的处理
                # 获取checkpoint文件名
                checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)
                if checkpoint_name.endswith('.pt') or checkpoint_name.endswith('.bin') or checkpoint_name.endswith('.pth'):
                     checkpoint_name = os.path.splitext(checkpoint_name)[0] + ".safetensors"
                     
                # 构建最佳checkpoint文件名
                best_checkpoint_name = os.path.splitext(checkpoint_name)[0] + "_best.safetensors"
                
                # 如果当前checkpoint文件存在
                if os.path.exists(checkpoint_name):
                    # 如果已存在旧的最佳checkpoint文件，先删除
                    if os.path.exists(best_checkpoint_name):
                        os.remove(best_checkpoint_name)

                    # 复制当前checkpoint为最佳checkpoint
                    shutil.copy2(checkpoint_name, best_checkpoint_name)
                    
                # 同时也处理 optimizer 文件
                optim_checkpoint_name = os.path.splitext(checkpoint_name)[0] + "_optim.pt"
                best_optim_checkpoint_name = os.path.splitext(checkpoint_name)[0] + "_best_optim.pt"
                if os.path.exists(optim_checkpoint_name):
                    if os.path.exists(best_optim_checkpoint_name):
                        os.remove(best_optim_checkpoint_name)
                    shutil.copy2(optim_checkpoint_name, best_optim_checkpoint_name)
                    
        except Exception as e: 
            print(f"Warning: Failed to save best checkpoint: {e}")
            pass

    # 等待所有进程
    TrainerTools().parallel.wait('save best checkpoint')
    # 返回是否需要替换的标志
    return need_replace


def load_checkpoint(
        model: nn.Module,  # 要加载权重的模型
        optimizer: Optional[Optimizer] = None,  # 优化器
        device: Optional[Union[torch.device, str]] = None,
        load_module_only: bool = False  # 是否仅加载模型参数
):
    # 检查是否使用DeepSpeed并行
    if isinstance(TrainerTools().parallel, DsParallel):
        # 导入DeepSpeed的checkpoint加载函数
        from .ds_checkpoint import load_ds_checkpoint
        # 使用DeepSpeed的加载方法
        load_ds_checkpoint(model, load_module_only=load_module_only)
    else:
        # 从环境变量获取checkpoint文件名
        checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)
        
        # 尝试加载 safetensors
        if not os.path.exists(checkpoint_name) and not checkpoint_name.endswith('.safetensors'):
             # 尝试加上 .safetensors 后缀查找
             temp_name = os.path.splitext(checkpoint_name)[0] + ".safetensors"
             if os.path.exists(temp_name):
                 checkpoint_name = temp_name

        # 如果checkpoint文件存在
        if os.path.exists(checkpoint_name):
            print(f"Loading checkpoint from {checkpoint_name}...")
            # 如果模型被DDP包装，提取原始模型，否则直接使用
            raw_model = model.module if isinstance(model, DDP) else model

            if checkpoint_name.endswith('.safetensors'):
                # 使用 safetensors 加载
                state_dict = load_file(checkpoint_name, device=str(device) if device else 'cpu')
                raw_model.load_state_dict(state_dict)
            else:
                # 兼容旧的 torch.load
                state_dict = torch.load(checkpoint_name, weights_only=True, map_location=device)
                if 'model_state_dict' in state_dict:
                    raw_model.load_state_dict(state_dict['model_state_dict'])
                else:
                    raw_model.load_state_dict(state_dict)

            # 如果提供了优化器，则加载优化器状态 (通常在单独的文件中)
            if optimizer and not load_module_only:
                # 尝试寻找 _optim.pt 文件
                optim_checkpoint_name = os.path.splitext(checkpoint_name)[0] + "_optim.pt"
                if os.path.exists(optim_checkpoint_name):
                    print(f"Loading optimizer state from {optim_checkpoint_name}...")
                    optim_state_dict = torch.load(optim_checkpoint_name, map_location=device)
                    optimizer.load_state_dict(optim_state_dict)
                elif not checkpoint_name.endswith('.safetensors') and 'optim_state_dict' in state_dict:
                    # 兼容旧版：如果在同一个文件中
                    optimizer.load_state_dict(state_dict['optim_state_dict'])



def load_checkpoint_for_eval(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None
):
    """
    为评估加载模型检查点（只加载模型参数，不加载优化器状态）
    """
    # 检查是否使用DeepSpeed并行
    if isinstance(TrainerTools().parallel, DsParallel):
        # 导入DeepSpeed的评估用checkpoint加载函数
        from .ds_checkpoint import load_ds_checkpoint_for_eval
        # 使用DeepSpeed的评估加载方法
        load_ds_checkpoint_for_eval(model)
    else:
        # 调用普通加载函数，不传入优化器
        load_checkpoint(model, None, device)


def save_steps(
    global_steps: int,  # 全局步数
    lr_scheduler: Optional[LRScheduler] = None,  # 学习率调度器
):
    """
    保存训练步数和学习率调度器状态
    """
    # 暂时只保存主进程的步数信息
    if TrainerTools().parallel.is_main_process:
        # 文件名
        steps_checkpoint_name = f"{os.environ.get('LOG_DIR', './')}steps.pt"
        # 创建checkpoint字典，包含全局步数
        ckpt = {'global_steps': global_steps}
        # 如果提供了学习率调度器，将其状态也加入checkpoint
        if lr_scheduler:
            ckpt.update(lr_scheduler.get_ckpt_dict())

        # 保存步数checkpoint到文件
        torch.save(ckpt, steps_checkpoint_name)


def load_steps() -> Optional[dict]:
    """
    加载训练步数和学习率调度器状态
    """
    # 文件名
    steps_checkpoint_name = f"{os.environ.get('LOG_DIR', './')}steps.pt"
    # 如果步数checkpoint文件存在
    if os.path.exists(steps_checkpoint_name):
        # 加载并返回步数checkpoint
        return torch.load(steps_checkpoint_name, weights_only=True)

    # 文件不存在则返回None
    return None

import os
from glob import glob  # 导入 glob 模块，用于查找符合特定规则的文件路径名
import shutil  # 导入 shutil 模块，用于高级文件操作（如递归删除目录）
from torch import nn
from .tools import TrainerTools  # 从当前包的 tools 模块导入 TrainerTools 工具类

try:
    from deepspeed import DeepSpeedEngine  # 尝试导入 DeepSpeed 的核心引擎类
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint  # 导入从 ZeRO 检查点提取 FP32 权重的工具
except: ...  # 如果导入失败（例如未安装 deepspeed），则忽略错误继续执行



def save_ds_checkpoint(model: nn.Module):
    assert isinstance(model, DeepSpeedEngine)  # 断言传入的模型必须是 DeepSpeedEngine 的实例
    ckpt_dir = os.environ.get('DIST_CHECKPOINT_DIR', 'checkpoint')  # 从环境变量获取检查点保存目录，默认为 'checkpoint'

    try:
        # 包括model、optimizer等状态
        model.save_checkpoint(save_dir=ckpt_dir)  # 调用 DeepSpeed 引擎保存检查点（包含模型参数、优化器状态、LR调度器等）
    except: ...  # 如果保存过程中发生异常，忽略错误

    # 只在main rank上执行
    if TrainerTools().parallel.is_main_process:  # 检查当前进程是否为主进程（通常是 Rank 0），只让主进程处理文件清理
        max_to_keep = int(os.environ.get('CKPT_MAX_TO_KEEP', '2'))  # 从环境变量读取最大保留的检查点数量，默认为 2 个
        # 删除历史checkpoint
        ckpt_paths = glob(os.path.join(ckpt_dir, "global_*"))  # 获取保存目录下所有以 "global_" 开头的检查点列表
        if len(ckpt_paths) > max_to_keep:
            # 按修改时间排序，找到最旧的目录
            oldest_ckpt = sorted(ckpt_paths, key=os.path.getmtime)[0]  # 按文件最后修改时间升序排序，取第一个（最旧的）
            try:
                shutil.rmtree(oldest_ckpt)  # 递归删除最旧的检查点目录及其内容
            except: ...  # 如果删除失败，忽略错误

    TrainerTools().parallel.wait('remove old ds checkpoint')  # 设置进程同步屏障，等待主进程完成清理，确保所有进程在继续前状态一致


def load_ds_checkpoint(
        model: nn.Module,
        load_module_only: bool = False  # 参数：是否只加载模型权重（忽略优化器状态），默认为否
):
    assert isinstance(model, DeepSpeedEngine)  # 断言模型必须是 DeepSpeedEngine 的实例
    ckpt_dir = os.environ.get('DIST_CHECKPOINT_DIR', 'checkpoint')  # 获取检查点目录路径

    # 包括model、optimizer等状态
    if os.path.exists(ckpt_dir):  # 检查检查点目录是否存在
        model.load_checkpoint(  # 调用 DeepSpeed 引擎加载检查点
            load_dir=ckpt_dir,  # 指定加载目录
            load_module_only=load_module_only  # 指定是否仅加载模型参数（用于微调或推理）或恢复完整训练状态
        )

# 专门用于评估的权重加载工具，核心解决 DeepSpeed ZeRO 优化训练后的权重分片问题 —— 将 ZeRO 分片存储的权重合并为标准 PyTorch 模型可直接加载的状态字典
def load_ds_checkpoint_for_eval(model: nn.Module):
    ckpt_dir = os.environ.get('DIST_CHECKPOINT_DIR', 'checkpoint')
    state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)  # 将分散在不同进程 / 文件中的分片权重合并拼接成一个完整的 FP32 精度的模型状态字典（state_dict）
    model.load_state_dict(state_dict)  # 将提取出的标准状态字典加载到 PyTorch 模型中
    
"""
函数	功能	是否加载模型到内存	是否保存到文件 主要用途
get_fp32_state_dict_from_zero_checkpoint	从 ZeRO 检查点提取 FP32 状态字典	否	否	获取模型权重，用于推理、迁移等
load_state_dict_from_zero_checkpoint	从 ZeRO 检查点加载模型和优化器状态	是	否	恢复训练状态，继续训练
convert_zero_checkpoint_to_fp32_state_dict	将 ZeRO 检查点转换为独立的 FP32 状态字典文件	否	是	创建可移植的 FP32 权重文件，用于部署、分享等
"""
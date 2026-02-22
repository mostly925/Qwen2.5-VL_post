import torch

from .generate_utils import generate_with_messages  # 导入生成函数，用于执行模型的推理/生成过程
from .tools import TrainerTools  # 导入训练工具类
from .train_configs import TrainConfig  # 导入训练配置类，包含各种超参数
from .log import _get_log_dir  # 导入获取日志目录路径的辅助函数

def submit_gen_task(
        eval_model: torch.nn.Module,  # 待评估的模型对象
        train_config: TrainConfig,    # 包含训练和评估参数的配置对象
        tag,                          # 标签，通常用于标识当前的训练步数或阶段
        prompt,                       # 输入给模型的提示词 (Prompt)
        image_path=None,              # 图片路径
        processor=None                # 处理器
):
    """
    文本生成任务，通常用于在训练过程中验证模型效果。
    """
    
    # 构造 messages
    messages = [
        {
            "role": "user",
            "content": []
        }
    ]
    
    if image_path:
        messages[0]["content"].append({"type": "image", "image": image_path})
    
    messages[0]["content"].append({"type": "text", "text": prompt})

    # 从配置中获取允许生成的最大新 Token 数量
    max_new_tokens = train_config.eval_config.max_new_tokens

    # 调用核心生成函数进行推理
    gen_result = generate_with_messages(
        eval_model,                                      # 传入模型
        processor=processor,                             # 传入处理器
        messages=messages,                               # 传入消息
        max_new_tokens=max_new_tokens,                   # 最大生成长度
        temperature=train_config.eval_config.temperature, # 采样温度（控制随机性）
        top_k=train_config.eval_config.top_k,            # Top-K 采样参数
        top_p=train_config.eval_config.top_p,            # Top-P (Nucleus) 采样参数
        device=TrainerTools().parallel.device            # 获取当前运行的设备 (CPU/GPU)
    )

    # 打开日志目录下的 'gen.txt' 文件，追加写入
    with open(f'{_get_log_dir()}gen.txt', 'a') as f:
        # 将本次生成的标签和结果写入文件，格式为 "tag, gen->result"
        f.write(f"{tag}, gen->{gen_result}\n")

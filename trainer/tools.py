import os
from abc import ABC, abstractmethod
import torch
from .tokenizer import Tokenizer
from .parallel import DsParallel, DdpParallel, NoneParallel
from .log import log


# 定义支持的并行类型字典，将字符串键映射到对应的并行类
parallel_types = {
    'ds': DsParallel,       # DeepSpeed 并行
    'ddp': DdpParallel,     # 分布式数据并行
    'none': NoneParallel    # 不使用并行
}

# 定义支持的数据类型字典，将字符串键映射到 PyTorch 的数据类型
dtypes = {
    'float': torch.float,       # 32位浮点数
    'float16': torch.float16,   # 16位半精度浮点数
    'float32': torch.float32,   # 32位浮点数
    'float64': torch.float64    # 64位双精度浮点数
}

class TrainerTools:
    def __init__(self):
        if not hasattr(TrainerTools, "_first_init"):
            TrainerTools._first_init = True  # 设置已初始化标记
            self.parallel = self._new_parallel()  # 调用内部方法创建并行处理对象
            self.tokenizer = Tokenizer()  # 实例化分词器对象
            # 判断是否使用自动混合精度：设备是 cuda 且 并行方式不是 DeepSpeed (DeepSpeed 内部自行管理精度)
            self.use_amp = 'cuda' in self.parallel.device and not isinstance(self.parallel, DsParallel)
            
            log(f'word_size={self.parallel.world_size}, use_amp={self.use_amp}')

    def _new_parallel(self):
        # 定义内部方法，用于根据环境变量创建并行对象
        # 加上 .strip() 去掉可能存在的 \r 或空格
        parallel_type = os.environ.get('PARALLEL_TYPE', 'none').strip()  # 获取环境变量 PARALLEL_TYPE，默认为 'none'
        
        log(f'parallel_type={parallel_type}')
        
        return parallel_types[parallel_type]()  # 根据类型查表并实例化对应的并行类返回

    def __new__(cls, *args, **kwargs):
        # 重写了 __new__ 方法
        if not hasattr(TrainerTools, "_instance"):
            # 如果类还没有 "_instance" 属性（即没有被实例化过）
            TrainerTools._instance = object.__new__(cls)  # 调用父类方法创建一个新实例并保存

        return TrainerTools._instance  # 如果已经被实例化过，返回该类的唯一实例


class FileDataset(ABC):
    # 定义一个抽象基类 FileDataset
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx) -> str: ...


def estimate_data_size(
        file_dataset: FileDataset,  # 参数：文件数据集对象
        processor,                  # 参数：处理器
        max_seq_len: int,           # 参数：最大序列长度
        type: str                   # 参数：任务类型 (如 sft, dpo, pretrain)
) -> int:
    """
    估计数据集大小
    """
    data_size = 0  # 初始化数据总量计数器
    files_count = len(file_dataset)  # 获取文件数据集中的文件总数
    
    # 如果任务类型是 SFT (监督微调)
    if type == 'sft':
        from .dataset import SFTDataset
        for idx in range(files_count):
            # 遍历每一个文件
            dataset = SFTDataset(file_dataset[idx], processor, max_seq_len)
            data_size += len(dataset)  # 累加当前文件包含的样本数量到总大小
    # 如果任务类型是 DPO (直接偏好优化)
    elif type == 'dpo':
        from .dataset import DPODataset
        for idx in range(files_count):
            # 遍历每一个文件
            dataset = DPODataset(file_dataset[idx], processor, max_seq_len)  # 实例化 DPO 数据集
            data_size += len(dataset)  # 累加样本数量
    # 如果任务类型是 GRPO 或 PPO (强化学习相关)
    elif type == 'grpo' or type == 'ppo':
        from .dataset import RLDataset
        for idx in range(files_count):
            # 遍历每一个文件
            dataset = RLDataset(file_dataset[idx], processor, max_seq_len)  # 实例化 RL 数据集
            data_size += len(dataset)  # 累加样本数量

    return data_size  # 返回计算出的总数据量大小
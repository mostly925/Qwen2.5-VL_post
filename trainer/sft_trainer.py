from typing import Optional, Tuple, List
from torch.utils.data import Dataset

from .trainer import Trainer
from .train_configs import TrainConfig, VLMConfig
from .dataset import SFTDataset
from .utils import get_sft_collate_fn


# 定义监督微调（SFT）训练器类
class SFTTrainer(Trainer):
    # 初始化方法，使用关键字参数传入配置
    def __init__(
            self,
            *,
            train_config: TrainConfig,  # 训练配置对象
            eval_prompts: List[str],    # 评估用的提示词列表
            eval_image_tags: Optional[List[str]] = None  # [可选]评估图像标签列表，默认为None
    ):
        # 调用父类Trainer的初始化方法，传入训练配置、评估提示词和评估图像标签
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            eval_image_tags=eval_image_tags
        )
        # 设置是否使用序列打包，SFT场景默认关闭
        self.packed_sequences = False

    
    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        # 获取 SFT 专用的数据整理函数
        sft_collate_fn = get_sft_collate_fn(self.train_config.mask_prompt)
        # 调用父类方法，获取基础配置
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()
        # 用 SFT 专用函数替换父类的 collate_fn
        data_loader_kwargs.update({"collate_fn": sft_collate_fn})
        return parallel_kwargs, data_loader_kwargs, sampler_kwargs


    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        max_seq_len = self.train_config.max_seq_len
        
        # 创建SFT数据集实例并返回，同时返回文件路径
        return SFTDataset(file_path, self.processor, max_seq_len), file_path
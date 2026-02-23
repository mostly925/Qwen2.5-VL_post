from dataclasses import dataclass


@dataclass
class ERLConfig:
    """ERL 训练配置。

    说明：
    - 所有注释均使用中文，便于后续直接二次开发。
    - 参数设置偏向最小复现，保证 CPU 可运行。
    """

    tau: float = 0.8
    memory_max_size: int = 64
    learning_rate: float = 0.15
    distill_rate: float = 0.4
    episodes: int = 500
    seed: int = 7

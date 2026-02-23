from dataclasses import dataclass


@dataclass
class ERLConfig:
    tau: float = 0.8
    memory_max_size: int = 64
    learning_rate: float = 0.15
    distill_rate: float = 0.4
    episodes: int = 500
    seed: int = 7

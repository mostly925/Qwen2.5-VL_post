from importlib import import_module
from typing import Any

_LAZY_IMPORTS = {
    "Trainer": ("trainer", "Trainer"),
    "SFTTrainer": ("sft_trainer", "SFTTrainer"),
    "DPOTrainer": ("dpo_trainer", "DPOTrainer"),
    "PPOTrainer": ("ppo_trainer", "PPOTrainer"),
    "GRPOTrainer": ("grpo_trainer", "GRPOTrainer"),
    "TrainerTools": ("tools", "TrainerTools"),
    "FileDataset": ("tools", "FileDataset"),
    "estimate_data_size": ("tools", "estimate_data_size"),
    "generate": ("generate_utils", "generate"),
    "streaming_generate": ("generate_utils", "streaming_generate"),
}

__all__ = [
    "Trainer",
    "SFTTrainer",
    "DPOTrainer",
    "PPOTrainer",
    "GRPOTrainer",
    "TrainerTools",
    "FileDataset",
    "estimate_data_size",
    "generate",
    "streaming_generate",
]


def __getattr__(name: str) -> Any:
    item = _LAZY_IMPORTS.get(name)
    if item is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, attr_name = item
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

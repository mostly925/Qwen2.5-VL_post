import argparse
import importlib
import os
import re
from typing import TYPE_CHECKING, Any, List, Optional

from trainer.erl import (
    ERLConfig,
    ERLTrainer,
    RLVRTrainer,
    TabularPolicy,
    Task,
    ToySparseControlEnv,
    evaluate_policy,
)


def _apply_qwen2_rmsnorm_patch() -> None:
    torch = importlib.import_module("torch")
    modeling_qwen2 = importlib.import_module("transformers.models.qwen2.modeling_qwen2")

    def fixed_qwen2_rmsnorm_forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.clone() * hidden_states.to(input_dtype)

    modeling_qwen2.Qwen2RMSNorm.forward = fixed_qwen2_rmsnorm_forward
    print("✅ 已应用 Qwen2RMSNorm Monkey Patch (Weight Clone Version)")


if TYPE_CHECKING:
    from trainer.tools import FileDataset as _FileDatasetBase
else:
    _FileDatasetBase = object


class JsonlFileDataset(_FileDatasetBase):
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> str:
        return self.file_paths[idx]


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def get_last_number(text: str) -> Optional[float]:
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def compute_accuracy_reward(generated: str, target: str) -> float:
    gen_answer = extract_answer(generated)
    target_answer = extract_answer(target)
    if gen_answer.lower() == target_answer.lower():
        return 1.0
    if target_answer.lower() in gen_answer.lower():
        return 0.7
    gen_num = get_last_number(gen_answer)
    target_num = get_last_number(target_answer)
    if gen_num is not None and target_num is not None:
        diff = abs(gen_num - target_num)
        if diff == 0:
            return 1.0
        if target_num != 0 and diff < 0.1 * abs(target_num):
            return 0.8
        if target_num != 0 and diff < 0.5 * abs(target_num):
            return 0.5
        return 0.2
    return 0.0


def compute_format_reward(text: str) -> float:
    reward = 0.0
    if "<think>" in text.lower() and "</think>" in text.lower():
        reward += 0.3
    if "<answer>" in text.lower() and "</answer>" in text.lower():
        reward += 0.3
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_len = len(think_match.group(1).strip())
        if think_len > 50:
            reward += 0.2
        elif think_len > 20:
            reward += 0.1
    return reward


def reward_func(
    prompts: List[Any], completion_ids: Any, answers: List[Optional[Any]]
) -> List[float]:
    from trainer.tools import TrainerTools

    tokenizer = TrainerTools().tokenizer
    rewards = []
    for i in range(len(completion_ids)):
        generated_text = tokenizer.decode(completion_ids[i], skip_special_tokens=True)
        answer_item = answers[i]
        target_text = answer_item if isinstance(answer_item, str) else str(answer_item)
        accuracy_reward = compute_accuracy_reward(generated_text, target_text)
        format_reward = compute_format_reward(generated_text)
        total_reward = 0.7 * accuracy_reward + 0.3 * format_reward
        rewards.append(total_reward)
    return rewards


def build_demo_tasks() -> List[Task]:
    return [
        Task(task_id="t1", target_actions="UURD"),
        Task(task_id="t2", target_actions="LLDR"),
        Task(task_id="t3", target_actions="RDLU"),
        Task(task_id="t4", target_actions="DRUL"),
    ]


def run_erl_or_rlvr(mode: str, args: argparse.Namespace) -> None:
    tasks = build_demo_tasks()
    env = ToySparseControlEnv(action_space="UDLR")

    config = ERLConfig(
        episodes=args.erl_episodes,
        tau=args.erl_tau,
        learning_rate=args.erl_lr,
        distill_rate=args.erl_distill_rate,
        memory_max_size=args.erl_memory_size,
        seed=args.erl_seed,
    )

    policy = TabularPolicy(actions=env.valid_actions(), seed=args.erl_seed)
    if mode == "erl":
        trainer = ERLTrainer(env=env, policy=policy, config=config)
    else:
        trainer = RLVRTrainer(env=env, policy=policy, config=config)

    stats = trainer.train(tasks)
    eval_reward = evaluate_policy(env, policy, tasks)

    print("=== 训练完成 ===")
    print(f"模式: {mode}")
    print(
        f"训练均值(first/second): "
        f"{stats.first_reward_mean:.3f}/{stats.second_reward_mean:.3f}"
    )
    print(f"贪心评估奖励: {eval_reward:.3f}")
    if isinstance(trainer, ERLTrainer):
        print(f"记忆条目数: {len(trainer.memory)}")


def run_grpo() -> None:
    from trainer.grpo_trainer import GRPOTrainer
    from trainer.tools import TrainerTools
    from trainer.train_configs import (
        DataLoaderConfig,
        DsConfig,
        DsOffloadConfig,
        DsZero2Config,
        GRPOConfig,
        OptimConfig,
        TrainConfig,
    )

    _apply_qwen2_rmsnorm_patch()

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["CKPT_MAX_TO_KEEP"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PARALLEL_TYPE"] = "ds"

    model_path = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"
    os.environ["TOKEN_DIR"] = model_path
    data_path = "/root/autodl-tmp/data/grpo_dataset.jsonl"
    file_dataset = JsonlFileDataset([data_path])

    ds_config = DsConfig(
        zero_config=DsZero2Config(
            offload_optimizer=DsOffloadConfig(device="cpu"),
        ),
        gradient_clipping=1.0,
        activation_checkpointing=None,
    )

    grpo_config = GRPOConfig(
        grpo_steps=1,
        group_size=4,
        mixup_alpha=0.9,
        loss_beta=0.01,
        loss_clip_eps=3e-4,
        loss_clip_eps_high=4e-4,
        loss_delta=None,
        loss_importance_sampling_level="seq",
        loss_type="grpo",
        gen_max_new_tokens=256,
        gen_temperature=0.7,
        gen_k=None,
        gen_p=0.95,
        gen_suppress_tokens=None,
    )

    train_config = TrainConfig(
        n_epochs=3,
        batch_size=1,
        model_name_or_path=model_path,
        file_dataset=file_dataset,
        max_seq_len=2048,
        eval_batch_interval=100,
        use_lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        ds_config=ds_config,
        grpo_config=grpo_config,
        optim_config=OptimConfig(initial_lr=5e-7),
        gradient_accumulation_steps=4,
        data_loader_config=DataLoaderConfig(
            data_loader_num_workers=0,
            data_loader_shuffle=True,
            data_loader_drop_last=False,
            data_loader_pin_memory=True,
        ),
    )

    trainer = GRPOTrainer(
        train_config=train_config,
        reward_func=reward_func,
        eval_prompts=[],
    )

    if hasattr(trainer.train_model, "gradient_checkpointing_disable"):
        trainer.train_model.gradient_checkpointing_disable()
        print("✅ 已显式禁用 Trainer 模型的梯度检查点")

    if hasattr(trainer.train_model, "module") and hasattr(
        trainer.train_model.module, "gradient_checkpointing_disable"
    ):
        trainer.train_model.module.gradient_checkpointing_disable()
        print("✅ 已显式禁用 DeepSpeed Module 的梯度检查点")

    if hasattr(trainer.train_model, "model") and hasattr(
        trainer.train_model.model, "gradient_checkpointing_disable"
    ):
        trainer.train_model.model.gradient_checkpointing_disable()
        print("✅ 已显式禁用 Base Model 的梯度检查点")

    print("🚀 开始在 A800 上进行高速 GRPO 训练 (ZeRO-2, No-GC)...")
    trainer.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer_mode",
        choices=["grpo", "erl", "rlvr"],
        default="grpo",
    )
    parser.add_argument("--erl_episodes", type=int, default=600)
    parser.add_argument("--erl_tau", type=float, default=0.8)
    parser.add_argument("--erl_lr", type=float, default=0.12)
    parser.add_argument("--erl_distill_rate", type=float, default=0.35)
    parser.add_argument("--erl_memory_size", type=int, default=64)
    parser.add_argument("--erl_seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.trainer_mode in {"erl", "rlvr"}:
        run_erl_or_rlvr(args.trainer_mode, args)
    else:
        run_grpo()

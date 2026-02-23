import argparse
import importlib
import os
import re
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING, Any, Iterable, List, Optional


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


_ANSWER_SPLIT_PATTERN = re.compile(r"[，,、;；。.!?？\n]+")
_CONNECTOR_PATTERN = re.compile(r"\b(?:and|&)\b|以及|并且|并|和|与|及")
_PUNCT_STRIP_PATTERN = re.compile(r"[\s\"“”'‘’()（）\[\]{}<>《》【】:_：-]")

_ANSWER_SYNONYMS = {
    "巴黎铁塔": "埃菲尔铁塔",
    "埃菲尔塔": "埃菲尔铁塔",
    "eiffel tower": "埃菲尔铁塔",
    "eiffeltower": "埃菲尔铁塔",
    "tower bridge": "伦敦塔桥",
    "london tower bridge": "伦敦塔桥",
    "跳起": "跳跃",
    "跳起来": "跳跃",
    "跃起": "跳跃",
    "jumping": "跳跃",
    "jump": "跳跃",
    "英短蓝猫": "英国短毛猫",
}


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    no_think = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    return no_think.strip()


def _normalize_unicode(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    for src, dst in _ANSWER_SYNONYMS.items():
        normalized = normalized.replace(src, dst)
    return normalized.strip()


def _normalize_segment(segment: str) -> str:
    normalized = _normalize_unicode(segment)
    for prefix in ("答案是", "答案:", "答案：", "答:", "答：", "是", "为"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
    normalized = _PUNCT_STRIP_PATTERN.sub("", normalized)
    for src, dst in _ANSWER_SYNONYMS.items():
        normalized = normalized.replace(src, dst)
    return normalized


def _split_answer_segments(text: str) -> List[str]:
    normalized = _normalize_unicode(text)
    normalized = _CONNECTOR_PATTERN.sub("，", normalized)
    segments = [
        _normalize_segment(part) for part in _ANSWER_SPLIT_PATTERN.split(normalized)
    ]
    return [seg for seg in segments if seg]


def get_last_number(text: str) -> Optional[float]:
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def _counter_f1(pred_items: Iterable[str], gold_items: Iterable[str]) -> float:
    pred_counter = Counter(pred_items)
    gold_counter = Counter(gold_items)
    common = sum((pred_counter & gold_counter).values())
    pred_total = sum(pred_counter.values())
    gold_total = sum(gold_counter.values())
    if pred_total == 0 or gold_total == 0 or common == 0:
        return 0.0
    precision = common / pred_total
    recall = common / gold_total
    return 2.0 * precision * recall / (precision + recall)


def _char_ngram_f1(pred_text: str, gold_text: str, n: int = 2) -> float:
    pred = pred_text.strip()
    gold = gold_text.strip()
    if not pred or not gold:
        return 0.0

    if len(pred) < n:
        pred_items = [pred]
    else:
        pred_items = [pred[idx : idx + n] for idx in range(len(pred) - n + 1)]

    if len(gold) < n:
        gold_items = [gold]
    else:
        gold_items = [gold[idx : idx + n] for idx in range(len(gold) - n + 1)]

    return _counter_f1(pred_items, gold_items)


def _number_match_reward(generated: str, target: str) -> float:
    gen_num = get_last_number(generated)
    target_num = get_last_number(target)
    if gen_num is None or target_num is None:
        return 0.0

    diff = abs(gen_num - target_num)
    if diff == 0:
        return 1.0

    scale = max(abs(target_num), 1e-6)
    rel_err = diff / scale
    if rel_err <= 0.02:
        return 0.9
    if rel_err <= 0.1:
        return 0.75
    if rel_err <= 0.5:
        return 0.5
    return 0.2


def compute_accuracy_reward(generated: str, target: str) -> float:
    gen_answer = extract_answer(generated)
    target_answer = extract_answer(target)

    gen_segments = _split_answer_segments(gen_answer)
    target_segments = _split_answer_segments(target_answer)

    if not target_segments:
        return 0.0

    if gen_segments == target_segments:
        return 1.0

    if sorted(gen_segments) == sorted(target_segments):
        return 1.0

    segment_f1 = _counter_f1(gen_segments, target_segments)
    gen_flat = "".join(sorted(gen_segments))
    target_flat = "".join(sorted(target_segments))
    ngram_f1 = _char_ngram_f1(gen_flat, target_flat)
    number_reward = _number_match_reward(gen_answer, target_answer)

    similarity = max(segment_f1, ngram_f1)
    if similarity >= 0.95:
        return max(0.95, number_reward)
    if similarity >= 0.8:
        return max(0.8, number_reward)
    if similarity >= 0.6:
        return max(0.6, number_reward)
    if similarity >= 0.4:
        return max(0.4, number_reward)
    return max(number_reward, 0.0)


def reward_func(
    prompts: List[Any], completion_ids: Any, answers: List[Optional[Any]]
) -> List[float]:
    _ = prompts
    from trainer.tools import TrainerTools

    tokenizer = TrainerTools().tokenizer
    rewards = []
    for idx in range(len(completion_ids)):
        generated_text = tokenizer.decode(completion_ids[idx], skip_special_tokens=True)
        answer_item = answers[idx]
        target_text = answer_item if isinstance(answer_item, str) else str(answer_item)
        rewards.append(compute_accuracy_reward(generated_text, target_text))
    return rewards


def _build_rl_train_config(args: argparse.Namespace):
    from trainer.train_configs import (
        DataLoaderConfig,
        DsConfig,
        DsOffloadConfig,
        DsZero2Config,
        GRPOConfig,
        OptimConfig,
        TrainConfig,
    )

    os.environ["TOKEN_DIR"] = args.model_path
    file_dataset = JsonlFileDataset([args.rl_dataset_path])

    ds_config = DsConfig(
        zero_config=DsZero2Config(
            offload_optimizer=DsOffloadConfig(device="cpu"),
        ),
        gradient_clipping=1.0,
        activation_checkpointing=None,
    )

    top_k = None if args.rl_gen_top_k <= 0 else args.rl_gen_top_k

    grpo_config = GRPOConfig(
        grpo_steps=1,
        group_size=args.rl_group_size,
        mixup_alpha=0.9,
        loss_beta=0.01,
        loss_clip_eps=3e-4,
        loss_clip_eps_high=4e-4,
        loss_delta=None,
        loss_importance_sampling_level="seq",
        loss_type="grpo",
        gen_max_new_tokens=args.rl_gen_max_new_tokens,
        gen_temperature=args.rl_gen_temperature,
        gen_k=top_k,
        gen_p=args.rl_gen_top_p,
        gen_suppress_tokens=None,
    )

    return TrainConfig(
        n_epochs=args.rl_n_epochs,
        batch_size=args.rl_batch_size,
        model_name_or_path=args.model_path,
        file_dataset=file_dataset,
        max_seq_len=args.rl_max_seq_len,
        eval_batch_interval=args.rl_eval_batch_interval,
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


def _disable_gradient_checkpointing(trainer: Any) -> None:
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


def _setup_rl_runtime_env() -> None:
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["CKPT_MAX_TO_KEEP"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PARALLEL_TYPE"] = "ds"


def run_grpo_baseline(args: argparse.Namespace) -> None:
    from trainer.grpo_trainer import GRPOTrainer

    _apply_qwen2_rmsnorm_patch()
    _setup_rl_runtime_env()

    train_config = _build_rl_train_config(args)
    trainer = GRPOTrainer(
        train_config=train_config,
        reward_func=reward_func,
        eval_prompts=[],
    )
    _disable_gradient_checkpointing(trainer)

    print("🚀 开始 GRPO 多模态基线训练...")
    trainer.train()


def run_erl_multimodal(args: argparse.Namespace) -> None:
    from trainer.erl_vlm_trainer import ERLVLMTrainer

    _apply_qwen2_rmsnorm_patch()
    _setup_rl_runtime_env()

    train_config = _build_rl_train_config(args)
    trainer = ERLVLMTrainer(
        train_config=train_config,
        reward_func=reward_func,
        eval_prompts=[],
        erl_tau=args.erl_tau,
        erl_memory_size=args.erl_memory_size,
        erl_reflection_max_new_tokens=args.erl_reflection_max_new_tokens,
        erl_reflection_history_size=args.erl_reflection_history_size,
    )
    _disable_gradient_checkpointing(trainer)

    print("🚀 开始 ERL 多模态训练（first attempt → reflection → second attempt）...")
    trainer.train()
    metrics = trainer.latest_erl_metrics
    print(
        "✅ ERL 训练统计: "
        f"first_reward_mean={metrics['first_reward_mean']:.4f}, "
        f"second_reward_mean={metrics['second_reward_mean']:.4f}, "
        f"reflection_ratio={metrics['reflection_ratio']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_mode", choices=["grpo", "erl"], default="grpo")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--rl_dataset_path", type=str, default="data/grpo_dataset.jsonl"
    )
    parser.add_argument("--rl_n_epochs", type=int, default=3)
    parser.add_argument("--rl_batch_size", type=int, default=1)
    parser.add_argument("--rl_group_size", type=int, default=4)
    parser.add_argument("--rl_max_seq_len", type=int, default=2048)
    parser.add_argument("--rl_eval_batch_interval", type=int, default=100)
    parser.add_argument("--rl_gen_max_new_tokens", type=int, default=256)
    parser.add_argument("--rl_gen_temperature", type=float, default=0.7)
    parser.add_argument("--rl_gen_top_p", type=float, default=0.95)
    parser.add_argument("--rl_gen_top_k", type=int, default=0)

    parser.add_argument("--erl_tau", type=float, default=0.8)
    parser.add_argument("--erl_memory_size", type=int, default=64)
    parser.add_argument("--erl_reflection_max_new_tokens", type=int, default=128)
    parser.add_argument("--erl_reflection_history_size", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.trainer_mode == "erl":
        run_erl_multimodal(args)
    else:
        run_grpo_baseline(args)

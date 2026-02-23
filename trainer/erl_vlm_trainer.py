import json
import importlib
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, cast

from .generate_utils import batch_generate
from .grpo_trainer import GRPOTrainer
from .tools import TrainerTools
from .utils import left_pad_sequence


class ERLVLMTrainer(GRPOTrainer):
    def __init__(
        self,
        *,
        erl_tau: float,
        erl_memory_size: int,
        erl_reflection_max_new_tokens: int,
        erl_reflection_history_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.erl_tau = erl_tau
        self.erl_reflection_max_new_tokens = erl_reflection_max_new_tokens
        self.erl_reflection_history_size = erl_reflection_history_size
        self.erl_memory: Dict[str, Deque[str]] = defaultdict(
            lambda: deque(maxlen=erl_memory_size)
        )
        self.latest_erl_metrics: Dict[str, float] = {
            "first_reward_mean": 0.0,
            "second_reward_mean": 0.0,
            "reflection_ratio": 0.0,
        }

    def _build_task_keys(self, batch_data: List[dict], group_size: int) -> List[str]:
        base_keys: List[str] = []
        for idx, item in enumerate(batch_data):
            if item.get("task_id"):
                base_keys.append(str(item["task_id"]))
                continue

            prompt_obj = item.get("prompt")
            if prompt_obj is None:
                base_keys.append(f"sample_{idx}")
                continue

            try:
                prompt_key = json.dumps(prompt_obj, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                prompt_key = str(prompt_obj)
            base_keys.append(prompt_key)

        return [key for key in base_keys for _ in range(group_size)]

    def _memory_hint(self, task_key: str) -> str:
        memories = self.erl_memory.get(task_key)
        if not memories:
            return ""
        recent = list(memories)[-self.erl_reflection_history_size :]
        return "\n".join(recent)

    @staticmethod
    def _truncate_to_max_seq(tokens: Any, max_seq_len: int) -> Any:
        if tokens.numel() <= max_seq_len:
            return tokens
        return tokens[-max_seq_len:]

    def _build_reflection_prompt_suffix(
        self,
        first_answer_text: str,
        memory_hint: str,
    ) -> str:
        answer = first_answer_text.strip()
        if memory_hint:
            return (
                "\n你上一轮的回答如下：\n"
                f"{answer}\n"
                "历史高分反思如下：\n"
                f"{memory_hint}\n"
                "请先给出简洁反思，指出关键错误与修正方向。"
            )
        return (
            "\n你上一轮的回答如下：\n"
            f"{answer}\n"
            "请先给出简洁反思，指出关键错误与修正方向。"
        )

    @staticmethod
    def _build_second_attempt_prompt_suffix(reflection_text: str) -> str:
        reflect = reflection_text.strip()
        return f"\n反思内容：\n{reflect}\n请基于反思给出修正后的最终答案。"

    def _decode_valid_tokens(self, token_row: Any, mask_row: Any) -> str:
        valid_tokens = token_row[mask_row.bool()]
        if valid_tokens.numel() == 0:
            return ""
        return TrainerTools().tokenizer.decode(
            valid_tokens.tolist(), skip_special_tokens=True
        )

    def _generate_rollout_data(self, generate_model, batch_data: List[dict]):
        torch = importlib.import_module("torch")
        pad_sequence = importlib.import_module("torch.nn.utils.rnn").pad_sequence

        grpo_config_optional = self.train_config.grpo_config
        if grpo_config_optional is None:
            raise ValueError("ERL 多模态训练需要可用的 grpo_config")
        grpo_config = cast(Any, grpo_config_optional)

        first_rollout = super()._generate_rollout_data(generate_model, batch_data)

        first_rewards = self.reward_func(
            first_rollout["repeated_prompts"],
            first_rollout["completion_ids"],
            first_rollout["repeated_answers"],
        )

        device = first_rollout["input_ids"].device
        pad_token_id = TrainerTools().tokenizer.pad
        group_size = grpo_config.group_size
        repeated_task_keys = self._build_task_keys(batch_data, group_size)

        first_rewards_tensor = torch.tensor(
            first_rewards,
            dtype=torch.float32,
            device=device,
        )
        needs_reflection = first_rewards_tensor < self.erl_tau

        second_rewards_tensor = first_rewards_tensor.clone()
        use_second_attempt = torch.zeros_like(needs_reflection)

        prompt_len = (
            first_rollout["input_ids"].shape[1]
            - first_rollout["completion_ids"].shape[1]
        )
        prompt_ids = first_rollout["input_ids"][:, :prompt_len]
        first_completion_ids = first_rollout["completion_ids"]
        first_completion_mask = first_rollout["completion_mask"]

        second_completion_by_idx: Dict[int, Any] = {}

        if needs_reflection.any():
            reflect_indices = needs_reflection.nonzero(as_tuple=False).view(-1).tolist()
            tokenizer = TrainerTools().tokenizer
            max_seq_len = self.train_config.max_seq_len

            reflect_prompt_list: List[Any] = []
            reflect_base_prompt_list: List[Any] = []
            reflect_first_completion_list: List[Any] = []

            for idx in reflect_indices:
                base_prompt = prompt_ids[idx][prompt_ids[idx] != pad_token_id]
                first_completion = first_completion_ids[idx][
                    first_completion_mask[idx].bool()
                ]
                first_answer_text = tokenizer.decode(
                    first_completion.tolist(), skip_special_tokens=True
                )
                memory_hint = self._memory_hint(repeated_task_keys[idx])
                suffix_text = self._build_reflection_prompt_suffix(
                    first_answer_text=first_answer_text,
                    memory_hint=memory_hint,
                )
                suffix_ids = torch.tensor(
                    tokenizer.encode(suffix_text),
                    device=device,
                    dtype=torch.long,
                )
                reflect_prompt = torch.cat(
                    (base_prompt, first_completion, suffix_ids), dim=0
                )
                reflect_prompt = self._truncate_to_max_seq(reflect_prompt, max_seq_len)

                reflect_prompt_list.append(reflect_prompt)
                reflect_base_prompt_list.append(base_prompt)
                reflect_first_completion_list.append(first_completion)

            reflect_prompt_ids = left_pad_sequence(
                reflect_prompt_list,
                padding_value=pad_token_id,
            ).to(device)
            reflect_attention_mask = reflect_prompt_ids != pad_token_id

            pixel_values = first_rollout.get("pixel_values")
            image_grid_thw = first_rollout.get("image_grid_thw")
            reflect_pixel_values = (
                pixel_values[reflect_indices] if pixel_values is not None else None
            )
            reflect_image_grid_thw = (
                image_grid_thw[reflect_indices] if image_grid_thw is not None else None
            )

            reflect_outputs, _ = batch_generate(
                model=generate_model,
                tokens=reflect_prompt_ids,
                attention_mask=reflect_attention_mask,
                max_new_tokens=self.erl_reflection_max_new_tokens,
                temperature=grpo_config.gen_temperature,
                k=grpo_config.gen_k,
                p=grpo_config.gen_p,
                device=device,
                suppress_tokens=grpo_config.gen_suppress_tokens,
                pixel_values=reflect_pixel_values,
                image_grid_thw=reflect_image_grid_thw,
            )
            reflection_ids = reflect_outputs[:, reflect_prompt_ids.shape[1] :]
            reflection_mask = (reflection_ids != pad_token_id).int()

            second_prompt_list: List[Any] = []
            reflection_texts: List[str] = []

            for row_idx, _ in enumerate(reflect_indices):
                reflection_text = self._decode_valid_tokens(
                    reflection_ids[row_idx],
                    reflection_mask[row_idx],
                )
                reflection_texts.append(reflection_text)

                second_suffix_text = self._build_second_attempt_prompt_suffix(
                    reflection_text=reflection_text,
                )
                second_suffix_ids = torch.tensor(
                    tokenizer.encode(second_suffix_text),
                    device=device,
                    dtype=torch.long,
                )
                second_prompt = torch.cat(
                    (
                        reflect_base_prompt_list[row_idx],
                        reflect_first_completion_list[row_idx],
                        second_suffix_ids,
                    ),
                    dim=0,
                )
                second_prompt = self._truncate_to_max_seq(second_prompt, max_seq_len)
                second_prompt_list.append(second_prompt)

            second_prompt_ids = left_pad_sequence(
                second_prompt_list,
                padding_value=pad_token_id,
            ).to(device)
            second_attention_mask = second_prompt_ids != pad_token_id

            second_outputs, _ = batch_generate(
                model=generate_model,
                tokens=second_prompt_ids,
                attention_mask=second_attention_mask,
                max_new_tokens=grpo_config.gen_max_new_tokens,
                temperature=grpo_config.gen_temperature,
                k=grpo_config.gen_k,
                p=grpo_config.gen_p,
                device=device,
                suppress_tokens=grpo_config.gen_suppress_tokens,
                pixel_values=reflect_pixel_values,
                image_grid_thw=reflect_image_grid_thw,
            )
            second_completion_ids = second_outputs[:, second_prompt_ids.shape[1] :]
            second_completion_mask = (second_completion_ids != pad_token_id).int()

            subset_prompts = [
                first_rollout["repeated_prompts"][idx] for idx in reflect_indices
            ]
            subset_answers = [
                first_rollout["repeated_answers"][idx] for idx in reflect_indices
            ]
            second_rewards_subset = self.reward_func(
                subset_prompts,
                second_completion_ids,
                subset_answers,
            )

            for row_idx, global_idx in enumerate(reflect_indices):
                second_reward = float(second_rewards_subset[row_idx])
                second_rewards_tensor[global_idx] = second_reward
                use_second_attempt[global_idx] = True

                second_completion_valid = second_completion_ids[row_idx][
                    second_completion_mask[row_idx].bool()
                ]
                second_completion_by_idx[global_idx] = second_completion_valid

                if second_reward > self.erl_tau:
                    reflection_text = reflection_texts[row_idx].strip()
                    if reflection_text:
                        self.erl_memory[repeated_task_keys[global_idx]].append(
                            reflection_text
                        )

        selected_prompt_list: List[Any] = []
        selected_completion_list: List[Any] = []
        end_token_id = TrainerTools().tokenizer.end

        for idx in range(first_rollout["completion_ids"].shape[0]):
            prompt_valid = prompt_ids[idx][prompt_ids[idx] != pad_token_id]
            selected_prompt_list.append(prompt_valid)

            if bool(use_second_attempt[idx]):
                selected_completion = second_completion_by_idx.get(idx)
                if selected_completion is None or selected_completion.numel() == 0:
                    selected_completion = torch.tensor(
                        [end_token_id],
                        device=device,
                        dtype=torch.long,
                    )
                selected_completion_list.append(selected_completion)
            else:
                first_valid = first_completion_ids[idx][
                    first_completion_mask[idx].bool()
                ]
                if first_valid.numel() == 0:
                    first_valid = torch.tensor(
                        [end_token_id],
                        device=device,
                        dtype=torch.long,
                    )
                selected_completion_list.append(first_valid)

        selected_prompt_ids = left_pad_sequence(
            selected_prompt_list,
            padding_value=pad_token_id,
        ).to(device)
        selected_completion_ids = pad_sequence(
            selected_completion_list,
            batch_first=True,
            padding_value=pad_token_id,
        ).to(device)
        selected_completion_mask = (selected_completion_ids != pad_token_id).int()
        selected_prompt_mask = selected_prompt_ids != pad_token_id

        selected_input_ids = torch.cat(
            (selected_prompt_ids, selected_completion_ids), dim=1
        )
        selected_attention_mask = torch.cat(
            (selected_prompt_mask, selected_completion_mask), dim=1
        )

        selected_old_log_probs, _ = self._compute_log_probs(
            generate_model,
            selected_input_ids,
            selected_attention_mask,
            pixel_values=first_rollout.get("pixel_values"),
            image_grid_thw=first_rollout.get("image_grid_thw"),
        )
        if self.ref_model:
            selected_ref_log_probs, _ = self._compute_log_probs(
                self.ref_model,
                selected_input_ids,
                selected_attention_mask,
                pixel_values=first_rollout.get("pixel_values"),
                image_grid_thw=first_rollout.get("image_grid_thw"),
            )
        else:
            selected_ref_log_probs = None

        if first_rewards_tensor.numel() > 0:
            first_reward_mean = float(first_rewards_tensor.mean().item())
            second_reward_mean = float(second_rewards_tensor.mean().item())
            reflection_ratio = float(needs_reflection.float().mean().item())
        else:
            first_reward_mean = 0.0
            second_reward_mean = 0.0
            reflection_ratio = 0.0

        self.latest_erl_metrics = {
            "first_reward_mean": first_reward_mean,
            "second_reward_mean": second_reward_mean,
            "reflection_ratio": reflection_ratio,
        }

        rollout_pixel_values = first_rollout.get("pixel_values")
        rollout_image_grid_thw = first_rollout.get("image_grid_thw")

        if rollout_pixel_values is not None:
            output_pixel_values = rollout_pixel_values.clone().detach()
        else:
            output_pixel_values = None

        if rollout_image_grid_thw is not None:
            output_image_grid_thw = rollout_image_grid_thw.clone().detach()
        else:
            output_image_grid_thw = None

        return {
            "input_ids": selected_input_ids.clone().detach(),
            "attention_mask": selected_attention_mask.clone().detach(),
            "completion_mask": selected_completion_mask.clone().detach(),
            "old_log_probs": selected_old_log_probs.clone().detach(),
            "ref_log_probs": (
                selected_ref_log_probs.clone().detach()
                if selected_ref_log_probs is not None
                else None
            ),
            "completion_ids": selected_completion_ids.clone().detach(),
            "pixel_values": output_pixel_values,
            "image_grid_thw": output_image_grid_thw,
            "repeated_prompts": first_rollout["repeated_prompts"],
            "repeated_answers": first_rollout["repeated_answers"],
        }

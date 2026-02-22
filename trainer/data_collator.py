import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any

class VLMDataCollator:
    """
    通用 VLM 数据整理器 (适配 Qwen2.5-VL)
    支持 SFT, DPO, RL 模式
    """
    def __init__(self, pad_token_id: int, mode: str = "sft"):
        self.pad_token_id = pad_token_id
        assert mode in ["sft", "dpo", "rl"], "Mode must be sft, dpo, or rl"
        self.mode = mode

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.mode == "sft":
            return self._collate_sft(batch)
        elif self.mode == "dpo":
            return self._collate_dpo(batch)
        elif self.mode == "rl":
            return self._collate_rl(batch)

    def _pad_tensor(self, tensors, padding_value):
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

    def _collate_sft(self, batch):
        input_ids = [x['input_ids'] for x in batch]
        labels = [x['labels'] for x in batch]
        
        batch_out = {
            "input_ids": self._pad_tensor(input_ids, self.pad_token_id),
            "labels": self._pad_tensor(labels, -100),
            "attention_mask": self._pad_tensor([torch.ones_like(x) for x in input_ids], 0)
        }
        self._merge_vision_features(batch, batch_out, prefix="")
        return batch_out

    def _collate_dpo(self, batch):
        # DPO 需要把 chosen 和 rejected 拼在一起处理，或者分开
        # 这里为了简单，返回独立的 batch
        chosen_ids = [x['chosen_input_ids'] for x in batch]
        rejected_ids = [x['rejected_input_ids'] for x in batch]
        
        batch_out = {
            "chosen_input_ids": self._pad_tensor(chosen_ids, self.pad_token_id),
            "chosen_attention_mask": self._pad_tensor([torch.ones_like(x) for x in chosen_ids], 0),
            "rejected_input_ids": self._pad_tensor(rejected_ids, self.pad_token_id),
            "rejected_attention_mask": self._pad_tensor([torch.ones_like(x) for x in rejected_ids], 0),
        }
        
        # 处理 Chosen 图片
        self._merge_vision_features(batch, batch_out, prefix="chosen_")
        # 处理 Rejected 图片
        self._merge_vision_features(batch, batch_out, prefix="rejected_")
        
        return batch_out

    def _collate_rl(self, batch):
        # RL (GRPO) 只需要 Prompt 的 tensor
        input_ids = [x['input_ids'] for x in batch]
        
        batch_out = {
            "input_ids": self._pad_tensor(input_ids, self.pad_token_id),
            "attention_mask": self._pad_tensor([torch.ones_like(x) for x in input_ids], 0),
            # 保留原始答案列表，不转 Tensor
            "answers": [x['answer_text'] for x in batch] 
        }
        self._merge_vision_features(batch, batch_out, prefix="")
        return batch_out

    def _merge_vision_features(self, batch, batch_out, prefix=""):
        """合并视觉特征 (pixel_values 和 image_grid_thw)"""
        # 检查 batch 中是否有 pixel_values
        key_pv = f"{prefix}pixel_values"
        key_grid = f"{prefix}image_grid_thw"
        
        if key_pv in batch[0] and batch[0][key_pv].numel() > 0:
            # Qwen2.5-VL: 简单的 cat
            batch_out[key_pv] = torch.cat([x[key_pv] for x in batch], dim=0)
            
        if key_grid in batch[0] and batch[0][key_grid].numel() > 0:
            batch_out[key_grid] = torch.cat([x[key_grid] for x in batch], dim=0)
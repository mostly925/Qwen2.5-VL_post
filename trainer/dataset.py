import torch
from torch.utils.data import Dataset
import json
import random
from qwen_vl_utils import process_vision_info
from transformers import ProcessorMixin

class BaseVLDataset(Dataset):
    """
    基础 Dataset 类，负责加载 JSONL 和初始化 Processor
    """
    def __init__(self, file_path: str, processor: ProcessorMixin, max_seq_len: int = 2048):
        self.data = []
        self.processor = processor
        self.max_seq_len = max_seq_len
        
        print(f"\n[Dataset] Loading data from: {file_path}")
        
        # --- 1. 读取数据 ---
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 尝试逐行读取 (JSONL)
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            self.data.append(obj)
                    except json.JSONDecodeError:
                        pass # 忽略非json行
        except Exception as e:
            print(f"[Dataset] Error reading file: {e}")

        # 如果为空，尝试读取整个文件为数组
        if not self.data:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content.startswith('[') and content.endswith(']'):
                        self.data = json.loads(content)
            except:
                pass

        # --- 2. 关键：数据有效性自检 ---
        if not self.data:
            raise ValueError(f"❌ 错误：未从 {file_path} 加载到任何数据！请检查文件路径或格式。")
        
        print(f"[Dataset] Loaded {len(self.data)} raw samples.")
        
        # 检查第一条数据
        first_item = self.data[0]
        print(f"[Dataset] Debug - First sample keys: {list(first_item.keys())}")
        
        # 我们通过检查类名来判断任务类型，或者简单地打印警告
        has_conversations = 'conversations' in first_item  # SFT
        has_prompt = 'prompt' in first_item                # GRPO / RL
        has_dpo = 'chosen' in first_item and 'rejected' in first_item # DPO
        has_text = 'text' in first_item                    # Pretrain

        if not (has_conversations or has_prompt or has_dpo or has_text):
            print("\n" + "!"*50)
            print("[FATAL ERROR] 数据集格式校验失败！")
            print(f"数据内容Keys: {list(first_item.keys())}")
            print("支持的格式如下:")
            print("1. SFT: 必须包含 'conversations'")
            print("2. GRPO/RL: 必须包含 'prompt'")
            print("3. DPO: 必须包含 'chosen' 和 'rejected'")
            print("!"*50 + "\n")
            # 这里抛出异常
            raise ValueError("Dataset validation failed: Unknown dataset format.")

    def __len__(self):
        return len(self.data)
    
    def process_qwen_input(self, conversations):
        """处理 Qwen2.5-VL 特有的图像/视频输入格式"""
        image_inputs, video_inputs = process_vision_info(conversations)
        text = self.processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
        return text, image_inputs, video_inputs

# ================= SFT Dataset =================
class SFTDataset(BaseVLDataset):
    """
    SFT 数据集：返回 input_ids, labels, pixel_values 等
    """
    def __getitem__(self, idx):
        # 使用循环代替递归，最多重试 10 次，防止死循环
        retry_count = 0
        max_retries = 10
        
        while retry_count < max_retries:
            # 确保索引在范围内
            current_idx = (idx + retry_count) % len(self.data)
            item = self.data[current_idx]
            
            # 检查数据完整性
            if not isinstance(item, dict) or 'conversations' not in item:
                # 仅在第一次失败时打印，避免刷屏
                if retry_count == 0:
                    print(f"[Warning] Sample {current_idx} invalid (missing 'conversations'). Skipping...")
                retry_count += 1
                continue
            
            try:
                # --- 核心处理逻辑 ---
                text, image_inputs, video_inputs = self.process_qwen_input(item['conversations'])
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=False,
                    return_tensors="pt",
                )
                
                result = {k: v.squeeze(0) for k, v in inputs.items()}
                result["labels"] = result["input_ids"].clone()
                return result
                
            except Exception as e:
                print(f"[Error] Failed to process sample {current_idx}: {e}")
                retry_count += 1
                continue
        
        # 如果重试多次都失败，抛出异常停止训练
        raise RuntimeError(f"Failed to find a valid sample after {max_retries} retries. Please check your dataset format.")

# ================= DPO Dataset =================
class DPODataset(BaseVLDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            c_text, c_imgs, c_vids = self.process_qwen_input(item['chosen'])
            c_inputs = self.processor(text=[c_text], images=c_imgs, videos=c_vids, padding=False, return_tensors="pt")
            
            r_text, r_imgs, r_vids = self.process_qwen_input(item['rejected'])
            r_inputs = self.processor(text=[r_text], images=r_imgs, videos=r_vids, padding=False, return_tensors="pt")

            # dpo_collate_fn 期望的是 token ID 列表，而不是 tensor
            # 返回键名为 'chosen' 和 'rejected' 以匹配 collate_fn
            return {
                "chosen": c_inputs["input_ids"].squeeze(0).tolist(),
                "rejected": r_inputs["input_ids"].squeeze(0).tolist(),
            }
        except Exception as e:
            print(f"Error in DPO dataset: {e}")
            return self.__getitem__((idx + 1) % len(self.data))

# ================= RL Dataset =================
class RLDataset(BaseVLDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            # RL Dataset 需要 'prompt' 字段
            if 'prompt' not in item:
                return self.__getitem__((idx + 1) % len(self.data))

            p_imgs, p_vids = process_vision_info(item['prompt'])
            text = self.processor.apply_chat_template(
                item['prompt'], tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[text], images=p_imgs, videos=p_vids, padding=False, return_tensors="pt")
            result = {k: v.squeeze(0) for k, v in inputs.items()}
            
            # [新增]：必须把原始的 prompt 对象传递出去，Trainer 需要用它来生成
            result["prompt"] = item['prompt']  
            
            # [注意]：dataset.py 里的原始代码可能是 "answer_text"，
            # 但 grpo_trainer.py 第 222 行找的是 item["answer"]
            # 所以这里最好保持字段名一致
            result["answer"] = item['answer'] 
            
            return result
        except Exception as e:
            print(f"Error loading RL sample {idx}: {e}") # 建议加上打印
            return self.__getitem__((idx + 1) % len(self.data))

# ================= Pretrain Dataset =================
# 仅预训练需要，SFT/DPO/GRPO/PPO 不需要
# class PretrainDataset(BaseVLDataset):
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         if 'conversations' in item:
#             text, image_inputs, video_inputs = self.process_qwen_input(item['conversations'])
#         elif 'text' in item:
#             text = item['text']
#             image_inputs, video_inputs = None, None
#         else:
#             return self.__getitem__((idx + 1) % len(self.data))

#         inputs = self.processor(
#             text=[text], images=image_inputs, videos=video_inputs, padding=False, return_tensors="pt",
#         )
#         result = {k: v.squeeze(0) for k, v in inputs.items()}
#         result["labels"] = result["input_ids"].clone()
#         return result
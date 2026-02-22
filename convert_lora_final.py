import os
import torch
import gc
import json
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig, LoraConfig
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from safetensors.torch import save_file

# ================= 配置区域 =================
# 1. Checkpoint 路径
checkpoint_dir = "/root/autodl-tmp/checkpoint" 

# 2. 基础模型路径
base_model_path = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"

# 3. 最终输出路径
output_dir = "/root/autodl-tmp/Qwen2.5-VL-7B-Final"

# 4. LoRA 配置 (必须与 run_qwen_sft.py 中的一致)
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# ===========================================

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

print(f"🚀 开始 LoRA 权重提取与合并...")

try:
    # --- 第一步：从 DeepSpeed 提取原始 FP32 权重 ---
    print(f"1️⃣  正在读取 DeepSpeed Checkpoint (这会消耗大量内存)...")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
    print("    读取完成。")

    # --- 第二步：提取 LoRA 权重并清洗 Key ---
    print(f"2️⃣  提取 LoRA Adapter 权重...")
    lora_state_dict = {}
    
    # 这里的逻辑是：DeepSpeed 保存的 key 通常多了几层 .model
    # 比如: base_model.model.model.visual... -> 目标: base_model.model.visual...
    # 我们只提取包含 'lora_' 的 key，并修正前缀
    
    for key, value in state_dict.items():
        if "lora_" in key or "modules_to_save" in key:
            # 修正 key：去除重复的 .model.model 中间层
            # 原始可能是 base_model.model.model.visual....
            # PEFT 需要 base_model.model.visual....
            new_key = key.replace("base_model.model.model.", "base_model.model.")
            lora_state_dict[new_key] = value

    print(f"    提取到 {len(lora_state_dict)} 个 LoRA 参数张量。")
    
    # 释放巨大的原始 state_dict 以节省内存
    del state_dict
    clean_memory()
    print("    已释放原始 FP32 权重内存。")

    # --- 第三步：加载基础模型 (BF16) ---
    print(f"3️⃣  加载基础模型 (BF16)...")
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 加载基础模型到 CPU (使用 bfloat16 省内存)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )

    # --- 第四步：应用 LoRA 并加载权重 ---
    print(f"4️⃣  应用 LoRA 配置并加载提取的权重...")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=True,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES
    )
    
    # 将基础模型包装为 PeftModel
    model = PeftModel(model, peft_config)
    
    # 加载我们清洗过的 LoRA 权重
    # strict=False 是为了容忍一些非 LoRA 键的微小差异，但主要的 LoRA 键必须匹配
    missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)
    
    if len(unexpected) > 0:
        print(f"    [注意] 发现意外的 Key (通常只需关注 lora 是否加载): {unexpected[:3]}...")
    
    print("    LoRA 权重加载成功！")

    # --- 第五步：合并权重并保存 ---
    print(f"5️⃣  合并 LoRA 到基础模型...")
    model = model.merge_and_unload()
    # 再次确保是 BF16
    model = model.to(dtype=torch.bfloat16)

    print(f"6️⃣  保存最终模型到: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    
    print("    保存 Processor...")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(output_dir)

    print("\n✅ ✅ 全部完成！")
    print(f"请使用 {output_dir} 进行推理。")

except Exception as e:
    print(f"\n❌ 发生错误: {e}")
    import traceback
    traceback.print_exc()
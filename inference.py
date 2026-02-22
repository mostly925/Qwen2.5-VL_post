import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 配置 =================
# 指向刚才转换成功的目录
model_dir = "/root/autodl-tmp/Qwen2.5-VL-7B-Final"

# 强制使用第一张显卡，避免 A800 双卡通信死锁
device = "cuda:0" 
# =======================================

print(f"🚀 正在加载最终模型: {model_dir}")

# 1. 加载模型
# 注意：现在是一个完整的模型，不需要再挂载 Peft/LoRA
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,  # 必须使用 bf16，与转换时一致
    device_map=device,           # 核心：指定单卡
    trust_remote_code=True
)

# 2. 加载处理器
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

print("✅ 模型加载成功，准备推理...")

# 3. 构造测试输入
# (你可以替换成你训练集里的某个问题，或者一张新图片来测试泛化能力)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "请详细描述这张图片的内容。"},
        ],
    }
]

# 4. 预处理
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# 确保数据也在同一张卡上
inputs = inputs.to(device)

# 5. 生成
# max_new_tokens 可以根据需要调大
generated_ids = model.generate(**inputs, max_new_tokens=256)

# 6. 解码
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("-" * 30)
print("🤖 模型回答:")
print(output_text[0])
print("-" * 30)
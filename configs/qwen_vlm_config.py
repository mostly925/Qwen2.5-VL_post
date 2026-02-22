def get_qwen_vlm_config() -> VLMConfig:
    """
    创建 Qwen2.5-VL 的配置。
    由于现在使用 HF AutoModel，这里主要用于返回一个占位符或特定超参。
    """
    # 返回一个兼容的 VLMConfig 对象
    # 如果后续需要特定参数（如 patch_size），可以在 VLMConfig 中添加并在这里赋值
    return VLMConfig(
        vision_tower="Qwen/Qwen2.5-VL-7B-Instruct" 
    )
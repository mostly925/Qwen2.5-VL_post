import os
import re
from typing import List, Dict, Union
from transformers import AutoTokenizer
import torch
from .log import log


class Tokenizer:
    def __init__(self):
        # 从环境变量 'TOKEN_DIR' 指定的目录加载预训练的分词器
        # Qwen2.5-VL 需要 trust_remote_code=True (如果它是自定义模型代码)，或者它是 HF 官方支持的
        
        token_dir = os.environ.get('TOKEN_DIR', 'Qwen/Qwen2.5-VL-7B-Instruct')
        
        # 如果路径不存在且看起来像模型ID，尝试从 ModelScope 下载
        if not os.path.exists(token_dir) and '/' in token_dir:
             try:
                 from modelscope import snapshot_download
                 print(f"本地路径 {token_dir} 不存在，尝试从 ModelScope 下载...")
                 token_dir = snapshot_download(token_dir)
                 print(f"Tokenizer已下载至: {token_dir}")
                 # 更新环境变量，以便后续使用
                 os.environ['TOKEN_DIR'] = token_dir
             except Exception as e:
                 print(f"⚠ ModelScope 下载 Tokenizer 失败: {e}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            token_dir, 
            trust_remote_code=True,
            use_fast=True   # 分词模式
        )

        # Qwen2.5-VL 特殊标记
        self.text_end = '<|im_end|>'
        self.text_pad = '<|endoftext|>'
        
        # 确保 tokenizer 识别这些标记
        if self.tokenizer.eos_token != self.text_end:
            self.tokenizer.eos_token = self.text_end
        if self.tokenizer.pad_token != self.text_pad:
            self.tokenizer.pad_token = self.text_pad
            
        self.end = self.tokenizer.convert_tokens_to_ids(self.text_end)
        self.pad = self.tokenizer.convert_tokens_to_ids(self.text_pad)
        # Qwen2.5-VL 没有 unk_token，通常使用 pad 或其他
        self.text_unk = self.tokenizer.unk_token if self.tokenizer.unk_token else '<|endoftext|>'
        self.unk = self.tokenizer.convert_tokens_to_ids(self.text_unk)
        
        # ChatML 格式标记
        self.text_im_start = '<|im_start|>'
        self.text_im_end = '<|im_end|>'
        self.im_start = self.tokenizer.convert_tokens_to_ids(self.text_im_start)
        self.im_end = self.tokenizer.convert_tokens_to_ids(self.text_im_end)
        
        # 角色标记 (仅用于 get_special_tokens_dict 映射，实际模板生成使用 ChatML 格式)
        self.text_user = 'user'
        self.text_assistant = 'assistant'
        self.text_system = 'system'
        # 这里我们不直接转换 user/assistant 字符串为 ID，因为在 ChatML 中它们是文本的一部分
        # 但为了保持兼容性，我们可以保留一些占位符或映射到 im_start
        self.user = self.im_start 
        self.assistant = self.im_start
        self.system = self.im_start

        # 定义思维链（CoT）- 如果不在词表中，作为普通文本处理
        self.text_think_start = '<think>'
        self.text_think_end = '</think>'
        # 尝试获取 ID，如果不存在则为 None (调用 encode 时会按字符处理)
        self.think_start = self.tokenizer.convert_tokens_to_ids(self.text_think_start) if '<think>' in self.tokenizer.get_vocab() else None
        self.think_end = self.tokenizer.convert_tokens_to_ids(self.text_think_end) if '</think>' in self.tokenizer.get_vocab() else None
        
        # 定义回答部分
        self.text_answer_start = '<answer>'
        self.text_answer_end = '</answer>'
        self.answer_start = self.tokenizer.convert_tokens_to_ids(self.text_answer_start) if '<answer>' in self.tokenizer.get_vocab() else None
        self.answer_end = self.tokenizer.convert_tokens_to_ids(self.text_answer_end) if '</answer>' in self.tokenizer.get_vocab() else None
        
        # 定义图片占位符 - 对应 Qwen2.5-VL 的 <|image_pad|> (151655)
        self.text_image = '<|image_pad|>'
        self.image = 151655 # 硬编码 Qwen2.5-VL 的 image token ID
        # 也可以尝试 self.tokenizer.convert_tokens_to_ids('<|image_pad|>') 验证

        # 获取词表大小
        self.vocab_size = len(self.tokenizer)

    def encode(
            self,
            text: str,
            unsqueeze: bool = False,
            covert_tensor: bool = False
    ) -> Union[torch.Tensor, List[int]]:
        # 调用基础分词器将文本编码为 ID 列表，add_special_tokens=False 表示不自动添加首尾特殊标记
        encoded = self.tokenizer.encode(text, add_special_tokens=False)

        if unsqueeze:
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        else:
            if covert_tensor:
                return torch.tensor(encoded, dtype=torch.long)
            return encoded

    def batch_encode(
            self,
            text: List[str],
            padding = False,
            truncation = False,
            covert_tensor: bool = False,
            return_attention_mask: bool = False
    ) -> Union[torch.Tensor, List[List[int]]]:
        encoded = self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
            add_special_tokens=False # 确保不添加额外的特殊标记，因为 apply_chat_template 已经处理了
        )['input_ids']

        if covert_tensor:
            encoded = torch.tensor(encoded, dtype=torch.long)

        return encoded

    def decode(
            self,
            token: Union[torch.Tensor, List[int]],
            skip_special_tokens: bool = False
    ) -> str:
        return self.tokenizer.decode(token, skip_special_tokens=skip_special_tokens)

    def batch_decode(
            self,
            tokens: Union[torch.Tensor, List[int], List[List[int]]],
            skip_special_tokens: bool = False
    ) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
            self,
            conversations: List[Dict[str, str]],
            tokenizer: bool = True,
            add_answer_tag_for_assistant: bool = True,
            unsqueeze=False,
            covert_tensor=False
    ):
        """
        将对话列表格式化为 Qwen2.5-VL (ChatML) 格式。
        """

        chat_template = ''
        image_paths = [] # 收集该对话中的所有图片路径
        
        for conversation in conversations:
            role = conversation['role']
            content = conversation['content']
            
            # 1. 提取图片路径并替换标签
            # 匹配 <img>...</img>
            img_matches = re.findall(r'<img>(.*?)</img>', content)
            if img_matches:
                image_paths.extend(img_matches)
                # 将 <img>path</img> 替换为 <|image_pad|>
                # 注意：这里我们替换为单个 token，dataset.py/utils.py 会负责将其扩展
                content = re.sub(r'<img>.*?</img>', self.text_image, content)

            # 兼容旧逻辑：如果 content 只有 <image> 且没有 vision 标签
            if '<image>' in content and '<|vision_start|>' not in content:
                 content = content.replace('<image>', self.text_image)

            # 处理思维链
            if 'think' in conversation:
                content = f"{self.text_think_start}\n{conversation['think']}\n{self.text_think_end}\n{content}"
                
            # 处理助手回答标签
            if add_answer_tag_for_assistant and role == 'assistant':
                if self.text_answer_start not in content:
                    content = f"{self.text_answer_start}\n{content}\n{self.text_answer_end}"

            chat_template += f"{self.text_im_start}{role}\n{content}{self.text_im_end}\n"

        if tokenizer:
            encoded = self.encode(chat_template, unsqueeze, covert_tensor)
            # 返回 (token_ids, image_paths) 元组
            return encoded, image_paths

        return chat_template, image_paths

    def get_special_tokens_dict(self):
        return {
            self.text_end: self.end,
            self.text_pad: self.pad,
            self.text_unk: self.unk,
            self.text_im_start: self.im_start,
            self.text_im_end: self.im_end,
            self.text_image: self.image,
        }
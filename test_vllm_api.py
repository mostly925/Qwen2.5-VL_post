from openai import OpenAI

# vLLM 兼容 OpenAI API
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# 测试图片 URL
image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

chat_response = client.chat.completions.create(
    model="qwen2.5-vl",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "详细描述这张图片。"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ],
    max_tokens=512,
    temperature=0.1,
    top_p=0.95,
)

print("vLLM 回复:", chat_response.choices[0].message.content)
from transformers import AutoProcessor
from qw_cof.model import Qwen2_5_VLForConditionalGeneration_COF
# from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import os

MODEL_PATH = "D:\models\Qwen\Qwen2.5-VL-7B-Instruc"


# 加载模型
model = Qwen2_5_VLForConditionalGeneration_COF.from_pretrained(
    MODEL_PATH, 
    torch_dtype="auto", 
    device_map="auto"
)

# 加载处理器并设置所有必要参数
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    use_fast=True,  # 使用快速处理器
)
processor.tokenizer.padding_side = 'left'  # 设置padding方向

# 批处理消息
batch_messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "1748866241318.jpg"},
                {"type": "text", "text": """Please first think through the reasoning process before answering the question. First locate the relate region by generating a bounding box in
 the following format: <|box_start|> [x1, y1, x2, y2] <|box_end|>. Then present your reason
ing process (including the bounding box) and answer enclosed within these tags: <think> reasoning process here </think>
 <answer> answer here </answer>.  Question: Tell me the number of that 
player who is shooting. And give the related bounding box."""},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "1748866323723.jpg"},
                {"type": "text", "text": """Please first think through the reasoning process before answering the question. First locate the relate region by generating a bounding box in
 the following format: <|box_start|> [x1, y1, x2, y2] <|box_end|>. Then present your reason
ing process (including the bounding box) and answer enclosed within these tags: <think> reasoning process here </think>
 <answer> answer here </answer>. Question: Tell me what is the dog eating. And give the related bounding box."""
                 },
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "1748866323723.jpg"},
                {"type": "text", "text": """Please first think through the reasoning process before answering the question. First locate the relate region by generating a bounding box in
 the following format: <|box_start|> [x1, y1, x2, y2] <|box_end|>. Then present your reason
ing process (including the bounding box) and answer enclosed within these tags: <think> reasoning process here </think>
 <answer> answer here </answer>. Question: Tell me what is the dog eating. And give the related bounding box."""
                 },
            ],
        }
    ],
]

# 处理批处理数据
texts = []
image_inputs_batch = []

for messages in batch_messages:
    # 准备文本输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    texts.append(text)
    
    # 准备视觉输入
    image_inputs, _ = process_vision_info(messages)  # 忽略视频输入
    image_inputs_batch.extend(image_inputs if image_inputs else [])

# exit()
# 处理批输入
inputs = processor(
    text=texts,
    images=image_inputs_batch if image_inputs_batch else None,
    padding=True,
    return_tensors="pt",
).to("cuda")


#在inputs字典中添加额外信息
inputs['processor'] = processor
inputs['image_inputs_batch'] = image_inputs_batch if image_inputs_batch else [None]*len(texts)
inputs['use_cof'] = True


# 生成响应
generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,
)

# 处理并输出结果
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_texts = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=False, 
    clean_up_tokenization_spaces=False
)

for i, output_text in enumerate(output_texts):
    print(f"Response for batch item {i+1}:")
    print(output_text)
    print("\n" + "="*50 + "\n")
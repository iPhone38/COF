from transformers import AutoProcessor
from qw_cof.model import Qwen2_5_VLForConditionalGeneration_COF
# from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import os
from qw_cof.utils.COF import COF

MODEL_PATH = "/root/workspace/qwvl/Qwen/Qwen2.5-VL-7B-Cof-Instruct"


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
                {"type": "text", "text": """<think> To determine the jersey number of the player taking the shot, I need to locate the player near the free-throw line 
where the action is likely happening. However, the relevant details are not clearly visible. I focus on the identified player within the bounding box 
<|box_start|>[385, 346, 423, 464]<|box_end|>. I zoom in again and extract a new visual embedding 
<|image_zoomin|><image>, which clearly shows that the player is wearing the number 8 jersey.  </think>
 <answer> 8 </answer>"""},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "1748866323723.jpg"},
                {"type": "text", "text": """ <think> To determine the letters on the laptop, I need to zoom in step by step, However, the brand details are not clearly visible. To improve visibility, I need to 
explore step by step. I first locate the key region <|box_start|>[30, 145, 148, 216]<|box_end|>, obtain refined embeddings <|image_zoomin|>, and identify that the dog is eating a cake. </think>
 <answer> a cake </answer>"""
                 },
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "1748866323723.jpg"},
                {"type": "text", "text": """ <think> To determine the letters on the laptop, I need to zoom in step by step, However, the brand details are not clearly visible. To improve visibility, I need to 
explore step by step. I first locate the key region <|box_start|>[30, 145, 148, 216]<|box_end|>, obtain refined embeddings <|image_zoomin|>, and identify that the dog is eating a cake. </think>
 <answer> a cake </answer>"""
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



# 处理批输入
inputs = processor(
    text=texts,
    images= None,
    padding=True,
    return_tensors="pt",
).to("cuda")

# exit()
# 在inputs字典中添加额外信息
inputs['processor'] = processor
inputs['image_inputs_batch'] = image_inputs_batch if image_inputs_batch else [None]*len(texts)
inputs['use_cof'] = True 

inputs_id = inputs["input_ids"]
CoF = COF(processor, image_inputs_batch)
for i in range(inputs_id.shape[-1]):
    next_tokens = inputs_id[:,i]
    CoF.process_next_tokens(next_tokens)
    CoF.print_state()
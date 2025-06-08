from transformers import AutoTokenizer, AutoConfig
from qw_cof import Qwen2_5_VLForConditionalGeneration_COF
import torch
import os

# 路径配置
MODEL_PATH = "/root/workspace/qwvl/Qwen/Qwen2.5-VL-7B-Instruct"
NEW_MODEL_PATH = "/root/workspace/qwvl/Qwen/Qwen2.5-VL-7B-Cof-Instruct"
os.makedirs(NEW_MODEL_PATH, exist_ok=True)

# 1. 加载原模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = Qwen2_5_VLForConditionalGeneration_COF.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda:0"
)

# 2. 添加新 token 并扩展模型
new_token = "<|image_zoomin|>"
tokenizer.add_tokens([new_token])
new_token_id = tokenizer.convert_tokens_to_ids(new_token)

# 检查输入/输出层是否共享权重
is_shared = model.get_input_embeddings().weight is model.lm_head.weight
print(f"输入/输出层共享权重: {is_shared}")

# 扩展输入嵌入层（自动处理共享权重情况）
model.resize_token_embeddings(len(tokenizer))

# 显式扩展输出层（如果未共享权重）
if not is_shared:
    print("扩展独立输出层...")
    old_lm_head_weight = model.lm_head.weight.data
    new_lm_head = torch.nn.Linear(
        in_features=model.config.hidden_size,
        out_features=len(tokenizer),
        bias=False,
        dtype=torch.bfloat16  # 确保新层也是 bfloat16
    ).to(model.device)
    
    # 复制旧权重，新 token 初始化为随机值
    new_lm_head.weight.data[:old_lm_head_weight.shape[0]] = old_lm_head_weight
    new_lm_head.weight.data[old_lm_head_weight.shape[0]:] = torch.randn(
        (len(tokenizer) - old_lm_head_weight.shape[0], model.config.hidden_size),
        device=model.device,
        dtype=torch.bfloat16  # 确保随机初始化的张量也是 bfloat16
    )
    model.lm_head = new_lm_head

# 更新模型配置
model.config.vocab_size = len(tokenizer)

# 3. 验证新 token 的嵌入初始化
with torch.no_grad():
    new_embed = model.get_input_embeddings().weight[new_token_id]
    print(f"新 token 嵌入向量范数: {torch.norm(new_embed)}")  # 应为非零

# 4. 测试生成功能
test_text = "User: <image>\n<|image_zoomin|> x=100 y=200\nAssistant:"
inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print("测试生成结果:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# 5. 保存新模型
tokenizer.save_pretrained(NEW_MODEL_PATH)
model.save_pretrained(NEW_MODEL_PATH)
print(f"新模型已保存到: {NEW_MODEL_PATH}")

# 6. 验证加载
new_tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL_PATH)
new_model = Qwen2_5_VLForConditionalGeneration_COF.from_pretrained(
    NEW_MODEL_PATH,
    torch_dtype="auto"
)
assert new_token in new_tokenizer.get_vocab(), "新 token 未成功添加！"
print("验证通过：模型和分词器已完整更新！")
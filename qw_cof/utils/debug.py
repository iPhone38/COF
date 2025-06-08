import torch
def print_model_inputs(model_inputs, max_length=5):
    """
    美观打印模型输入字典，支持自动截断长序列和特殊类型处理
    
    参数:
        model_inputs (dict): 模型输入字典
        max_length (int): 序列最大显示长度（防止控制台输出过长）
    """
    print("\n" + "=" * 80)
    print("🛠️ Model Inputs Structure")
    print("-" * 80)
    
    # 遍历字典所有键值对
    for key, value in model_inputs.items():
        print(f"🔑 {key}:  {value}")
        
        
    
    print("=" * 80 + "\n")
    

def print_outputs(outputs, max_lines=3, max_items_per_line=6):
    print("=" * 60)
    print("Qwen2_5_VLCausalLMOutputWithPast Detailed Summary")
    print("=" * 60)
    
    # Helper function to print truncated tensor
    def print_tensor(tensor, name, indent=0):
        indent_str = " " * indent
        print(f"{indent_str}[{name}] Shape: {tuple(tensor.shape)} | Dtype: {tensor.dtype} | Device: {tensor.device}")
        if tensor.numel() == 0:
            print(f"{indent_str}  (Empty tensor)")
            return
        
        # Print sample values for non-scalar tensors
        if tensor.dim() > 0:
            print(f"{indent_str}Sample values (truncated):")
            view = tensor.flatten()[:max_items_per_line]
            print(f"{indent_str}  {', '.join([f'{x:.4f}' for x in view.tolist()])}" + 
                  (" ..." if tensor.numel() > max_items_per_line else ""))
    
    # Print logits
    if hasattr(outputs, 'logits'):
        print("\n[Logits]")
        print_tensor(outputs.logits, "logits", indent=2)
        
        # Additional batch-wise details if rank >= 2
        if outputs.logits.dim() >= 2:
            print("\n  Per-batch summary:")
            for i in range(min(outputs.logits.shape[0], max_lines)):
                batch_logits = outputs.logits[i]
                print(f"    Batch {i}: Shape {tuple(batch_logits.shape)}")
                if batch_logits.dim() >= 1:  # Show first token's logits
                    first_token = batch_logits[0][:max_items_per_line]
                    print(f"      First token: {', '.join([f'{x:.4f}' for x in first_token.tolist()])}" +
                          (" ..." if batch_logits.shape[-1] > max_items_per_line else ""))
            if outputs.logits.shape[0] > max_lines:
                print(f"    ... (showing first {max_lines}/{outputs.logits.shape[0]} batches)")
    
    # Print past_key_values
    if hasattr(outputs, 'past_key_values'):
        print("\n[Past Key Values]")
        print(f"  Type: {type(outputs.past_key_values)}")
        if hasattr(outputs.past_key_values, 'seen_tokens'):
            print(f"  Seen tokens: {outputs.past_key_values.seen_tokens}")
        if hasattr(outputs.past_key_values, '_cache'):
            print("  Cache structure:")
            for i, (k_cache, v_cache) in enumerate(outputs.past_key_values._cache.items()):
                print(f"    Layer {i}:")
                print(f"      Key cache:   Shape {tuple(k_cache.shape)}")
                print(f"      Value cache: Shape {tuple(v_cache.shape)}")
    
    # Print rope_deltas
    if hasattr(outputs, 'rope_deltas') and outputs.rope_deltas is not None:
        print("\n[RoPE Deltas]")
        print_tensor(outputs.rope_deltas, "rope_deltas", indent=2)
        print(f"  Values: {outputs.rope_deltas.tolist()}")
    
    # Print other fields
    other_fields = [attr for attr in ['loss', 'hidden_states', 'attentions'] 
                   if hasattr(outputs, attr) and getattr(outputs, attr) is not None]
    if other_fields:
        print("\n[Other Fields]")
        for field in other_fields:
            val = getattr(outputs, field)
            if isinstance(val, torch.Tensor):
                print_tensor(val, field, indent=2)
            else:
                print(f"  {field}: {val}")

# Usage:
# print_outputs(outputs)
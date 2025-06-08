import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
from PIL import Image
import torch
from torchvision import transforms
from typing import Union, List
import torch.nn.functional as F


@dataclass
class StepData:
    bounding_boxes: List[List[float]]  # List of bounding boxes in [x1,y1,x2,y2] format
    output_tokens: List[int]  # All output tokens for this step
    region_image: Optional[Image.Image] = None
    visual_tokens: Optional[torch.Tensor] = None  # Visual tokens if zoomed in
    
class COF:
    def __init__(self, processor, image_inputs_batch: torch.Tensor):
        """
        Initialize the COF class.
        
        Args:
            processor: The image processor that can crop/resize/encode image regions
            image_inputs_batch: Batch of input images (bs, C, H, W)
        """
        
        self.processor = processor
        self.PAD_TOKEN_ID = self.processor.tokenizer.pad_token_id
        self.BOX_START_ID = self.processor.tokenizer.convert_tokens_to_ids("<|box_start|>")
        self.BOX_END_ID = self.processor.tokenizer.convert_tokens_to_ids("<|box_end|>")
        self.ZOOMIN_TOKEN_ID = self.processor.tokenizer.convert_tokens_to_ids("<|image_zoomin|>")
        self.image_inputs_batch = image_inputs_batch
        self.batch_size = len(image_inputs_batch)
        self.device = None
        
        # Initialize history for each sample in the batch
        self.history: List[List[StepData]] = [[] for _ in range(self.batch_size)]
        
    def update_output_tokens(self, next_tokens: torch.Tensor):
        """
        Update output_tokens for each request's current step.
        If a request has no current step (just started), create a new step.
        
        Args:
            next_tokens: Tensor of shape [bs] containing the next token for each request
        """
        for i in range(self.batch_size):
            token = next_tokens[i].item()
            
            # Get current step or create new one if doesn't exist
            if not self.history[i] or self.history[i][-1].region_image is not None:
                # Either no history or last step is complete (has visual tokens)
                self.history[i].append(StepData(bounding_boxes=[], output_tokens=[]))
            
            # Append the token to current step's output
            self.history[i][-1].output_tokens.append(token)
    
    def extract_boxes(self):
        """
        Extract bounding boxes from the output tokens with complete format validation.
        Expected format: <|box_start|>[x1,y1,x2,y2]<|box_end|>
        """
        # Get special token IDs from processor
        for i in range(self.batch_size):
            current_step = self.history[i][-1] if self.history[i] else None
            if not current_step:
                continue
            
            output_tokens = current_step.output_tokens
            
            if not output_tokens:
                continue
            
            current_token_idx = len(output_tokens) - 1
            
            # Only proceed if we have a box end token
            if output_tokens[current_token_idx] == self.BOX_END_ID:
                # Find the matching box_start (search backwards)
                start_idx = current_token_idx
                cnt = 0
                while start_idx >= 0 and output_tokens[start_idx] != self.BOX_START_ID:
                    start_idx -= 1
                    cnt += 1
                
                # Validate we found the start token and correct format length
                if start_idx < 0:
                    raise ValueError("No matching <|box_start|> found")
                
                
                # Extract the coordinate tokens (skip separators)
                
                box_tokens = output_tokens[start_idx:current_token_idx+1]
                box_text = self.processor.tokenizer.decode(box_tokens)
                # box_text :<|box_start|>[74, 134, 612, 272]<|box_end|>
                # 使用正则表达式提取方括号内的数字
                match = re.search(r'\[([\d,\s]+)\]', box_text)
                if match:
                    coords_str = match.group(1) 
                    coords = [int(num.strip()) for num in coords_str.split(",")]
                else:
                    print("Bounding Box Format Error!")
                    
                # Add to current step's boxes if valid
                if len(coords) == 4:
                    current_step.bounding_boxes.append(coords)
                
        
    def detect_zoomin(self, next_tokens: torch.Tensor) -> torch.Tensor:
        """
        Detect if any of the next_tokens is the zoomin token.
        For requests that have the zoomin token, perform the zoom-in operation.
        
        Args:
            next_tokens: Tensor of shape [bs] containing the next token for each request
            
        Returns:
            is_zoomin: Boolean tensor of shape [bs] indicating which requests have zoomin token
        """
        is_zoomin = next_tokens == self.ZOOMIN_TOKEN_ID

        for i in range(self.batch_size):
            if is_zoomin[i] and self.history[i]:
                current_step = self.history[i][-1]
                
                if current_step.bounding_boxes:
                    # Get the last bounding box (most recent)
                    bbox = current_step.bounding_boxes[-1]
                    
                    # Crop and process the image region
                    image = self.image_inputs_batch[i]
                    # print(image,bbox) 
                    # #<PIL.Image.Image image mode=RGB size=196x280 at 0x7F7C58C553C0> [36, 455, 320, 628]
                    cropped_region = self._crop_image(image, bbox)
                    
                    #to do : 
                    # visual_tokens = self.processor(cropped_region)
                    region_image = cropped_region
                    
                    # Store the visual tokens in the current step
                    current_step.region_image = region_image
        
        return is_zoomin
    
    def get_visual_tokens(self, is_zoomin: torch.Tensor, save_dir: str = None) -> dict:
        """
        获取视觉token，并可选择将图片保存到指定路径
        
        Args:
            is_zoomin: 布尔张量，指示哪些样本需要视觉token
            save_dir: 可选，图片保存路径。如果为None则不保存
            
        Returns:
            处理后的输入字典或None
        """
        # 准备输入列表
        texts = []
        valid_region_images = []
        valid_indices = []
        
        # 收集所有样本的文本和图像
        for i in range(self.batch_size):
            if is_zoomin[i] and self.history[i] and self.history[i][-1].region_image is not None:
                # 需要视觉token的样本
                texts.append("<|image_pad|>")
                valid_region_images.append(self.history[i][-1].region_image)
                valid_indices.append(i)
            else:
                # 不需要视觉token的样本
                texts.append("")  # 空字符串
        
        # 如果没有需要处理的样本，直接返回None
        if not valid_region_images:
            return None
        
        # 如果需要保存图片
        if save_dir is not None:
            import os
            from PIL import Image
            import time
            
            # 创建保存目录
            os.makedirs(save_dir, exist_ok=True)
            
            # 为每张图片生成唯一文件名并保存
            for idx, img in enumerate(valid_region_images):
                timestamp = int(time.time() * 1000)
                filename = f"{valid_indices[idx]}_image_{timestamp}_{idx}.png"
                save_path = os.path.join(save_dir, filename)
                
                if isinstance(img, Image.Image):
                    img.save(save_path)
                elif isinstance(img, torch.Tensor):
                    # 假设是[C,H,W]格式的tensor
                    transforms.ToPILImage()(img).save(save_path)
                else:
                    raise TypeError(f"Unsupported image type: {type(img)}")
        
        # 处理所有样本（包括不需要视觉token的样本）
        inputs = self.processor(
            text=texts,
            images=valid_region_images if valid_region_images else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # 存储视觉token到当前步骤（仅有效样本）
        visual_token_idx = 0
        for idx, batch_pos in enumerate(valid_indices):
            visual_token_num = torch.prod(inputs.image_grid_thw[idx])
            print(visual_token_num)
            visual_induce_start = visual_token_idx
            visual_induce_end = visual_induce_start + visual_token_num
            visual_token_idx = visual_induce_end
            current_step = self.history[batch_pos][-1]
            current_step.visual_tokens = {
                'input_ids': inputs.input_ids[idx],
                'pixel_values': inputs.pixel_values[visual_induce_start:visual_induce_end],
                'attention_mask': inputs.attention_mask[idx],
                'image_grid_thw': inputs.image_grid_thw[idx]
            }
        # print(inputs)
        return inputs
    
    def _crop_image(
        self, 
        image: Union[Image.Image, torch.Tensor], 
        bbox: List[int],  # [x1, y1, x2, y2]
        return_type: str = "same",  # "same"|"pil"|"tensor"
        resize_back: bool = True    # 新增参数：是否恢复原始尺寸
    ) -> Union[Image.Image, torch.Tensor]:
        """
        裁剪图像中的指定区域，并可选恢复原始尺寸
        
        Args:
            image: 输入图像 (PIL.Image 或 torch.Tensor [C,H,W])
            bbox: 边界框坐标 [x1, y1, x2, y2]
            return_type: 返回格式 ("same"|"pil"|"tensor")
            resize_back: 是否将裁剪结果resize回原图尺寸
            
        Returns:
            裁剪后的图像 (格式取决于 return_type)
        """
        # 检查bbox合法性
        x1, y1, x2, y2 = map(int, bbox)
        assert x1 < x2 and y1 < y2, f"Invalid bbox coordinates: {bbox}"

        # 保存原始尺寸 (用于后续resize)
        orig_size = (image.width, image.height) if isinstance(image, Image.Image) \
                   else (image.shape[2], image.shape[1])

        # 处理PIL输入
        if isinstance(image, Image.Image):
            cropped = image.crop((x1, y1, x2, y2))
            if resize_back:
                cropped = cropped.resize(orig_size, Image.BILINEAR)  # 恢复原始尺寸
            if return_type == "tensor":
                return transforms.ToTensor()(cropped)
            return cropped

        # 处理Tensor输入 (假设格式为 [C,H,W])
        elif isinstance(image, torch.Tensor):
            assert image.dim() == 3, f"Tensor must be [C,H,W], got {image.shape}"
            
            # 检查坐标是否在图像范围内
            _, h, w = image.shape
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            cropped = image[:, y1:y2, x1:x2]
            if resize_back:
                cropped = F.interpolate(
                    cropped.unsqueeze(0),  # 增加batch维度 [1,C,H,W]
                    size=(h, w),           # 目标尺寸 (原始H,W)
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)              # 移除batch维度
            if return_type == "pil":
                return transforms.ToPILImage()(cropped)
            return cropped

        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
    
    def process_next_tokens(self, next_tokens: torch.Tensor):
        """
        Main function to process the next tokens generated by the model.
        
        Args:
            next_tokens: Tensor of shape [bs] containing the next token for each request
        """
        # Step 1: Update output tokens for each request's current step
        self.device = next_tokens.device
        self.update_output_tokens(next_tokens)
        
        # Step 2: Extract any bounding boxes from the output
        self.extract_boxes()
        
        # Step 3: Detect zoomin tokens and handle them
        is_zoomin = self.detect_zoomin(next_tokens)
        
        zoomin_exist = is_zoomin.any()
        if zoomin_exist:
            return zoomin_exist, self.get_visual_tokens(is_zoomin)
        else:
            return zoomin_exist, None
        
        
    def append_visual_tokens( self,
        append_inputs: Dict,       # 来自 get_visual_tokens() 的视觉 tokens
        model_kwargs: Dict,        # 当前的 model_kwargs（会被 in-place 修改）
        input_ids: torch.Tensor,    # 当前的 input_ids（纯文本 tokens）
    ) -> Dict:
        """
        将视觉 tokens 插入到模型的输入中，并更新 input_ids 和 attention_mask。
        
        Args:
            append_inputs: 包含视觉 tokens 的字典，应有：
                - 'input_ids' (视觉 token 占位符，如 `<|image_pad|>`)
                - 'pixel_values' (视觉特征)
                - 'attention_mask' (视觉 tokens 的 attention mask)
                - 'image_grid_thw' (视觉 token 的网格尺寸)
            model_kwargs: 当前模型的前向传播参数（会被修改）
            input_ids: 当前的文本 token 序列（会被更新）
        
        Returns:
            更新后的 model_kwargs（包含视觉 tokens）
        """
        if append_inputs is None:
            return model_kwargs, input_ids

        # 确保 model_kwargs 有必要的 keys
        if "pixel_values" not in model_kwargs:
            model_kwargs["pixel_values"] = []
        if "image_grid_thw" not in model_kwargs:
            model_kwargs["image_grid_thw"] = []

        # 1. 插入视觉 token 占位符到 input_ids
        visual_input_ids = append_inputs["input_ids"].to(input_ids.device)
        updated_input_ids = torch.cat([input_ids, visual_input_ids], dim=1)  # 追加到末尾（或插入到 `<|image_zoomin|>` 之后）
        
        # 2. 更新 attention_mask（合并文本和视觉 tokens）
        text_attention_mask = model_kwargs.get("attention_mask", torch.ones_like(input_ids))
        visual_attention_mask = append_inputs["attention_mask"].to(input_ids.device)
        updated_attention_mask = torch.cat([text_attention_mask, visual_attention_mask], dim=1)
        
        # 3. 更新 pixel_values
        append_visul_token_num = append_inputs["pixel_values"].shape[0]
        updata_pixel_values = torch.cat(model_kwargs["pixel_values"], append_inputs["pixel_values"].to(input_ids.device), dim = 0)
        
        
        # 4. 更新 image_grid_thw
        updata_image_grid_thw = torch.cat(model_kwargs["image_grid_thw"], append_inputs["image_grid_thw"].to(input_ids.device), dim = 0)
        
        # 5. 更新 cache_position（增加视觉 token 的长度）
        visual_token_length = visual_input_ids.shape[1]  # 视觉 token 的序列长度
        updated_cache_position = torch.arange(model_kwargs["cache_position"] , model_kwargs["cache_position"] + visual_token_length + 1)
        updated_cache_position = updated_cache_position.to(input_ids.device)
        
        # 6. 更新 model_kwargss,input_ids
        model_kwargs.update({
            "attention_mask": updated_attention_mask,  # 更新 attention_mask
            "pixel_values": updata_pixel_values,  # 追加视觉特征
            "image_grid_thw": updata_image_grid_thw,  # 追加视觉网格
            "cache_position": updated_cache_position  # 追加视觉token 的对应cache位置
        })
        input_ids = updated_input_ids
        
        return model_kwargs, input_ids, append_visul_token_num
        
        
    def print_state(self, verbose: bool = True, show_tokens: bool = False, show_visual_shapes: bool = True):
        """
        Print the current state of the COF processor.
        
        Args:
            verbose: If True, prints detailed information for each step
            show_tokens: If True, prints all output tokens (can be long)
            show_visual_shapes: If True, shows shapes of visual tokens
        """
        print("\n" + "="*80)
        print(f"COF Processor State (batch size: {self.batch_size})")
        print(f"Zoom-in token ID: {self.ZOOMIN_TOKEN_ID}")
        print("="*80)
        
        for i, request_history in enumerate(self.history):
            print(f"\nRequest {i} - {len(request_history)} step(s):")
            
            if not request_history:
                print("  No steps recorded yet")
                continue
                
            for step_idx, step in enumerate(request_history):
                print(f"\n  Step {step_idx}:")
                
                # Print bounding boxes
                if step.bounding_boxes:
                    print(f"    Bounding Boxes ({len(step.bounding_boxes)}):")
                    for box_idx, box in enumerate(step.bounding_boxes):
                        print(f"      Box {box_idx}: [{box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f}]")
                else:
                    print("    No bounding boxes")
                
                # Print output tokens
                print(f"    Output Tokens ({len(step.output_tokens)}):")
                if show_tokens or len(step.output_tokens) <= 10:
                    print(f"      {step.output_tokens}")
                else:
                    print(f"      First 5: {self.processor.tokenizer.decode(step.output_tokens[:5])}")
                    print(f"      Last 5: {self.processor.tokenizer.decode(step.output_tokens[-5:])}")
                
                # Print visual tokens info
                if step.region_image is not None:
                    if show_visual_shapes:
                        print(f"    region_image: {step.region_image}")
                        print(f"    visual Tokens: {step.visual_tokens['pixel_values'].shape} | {step.visual_tokens['image_grid_thw']}")
                    else:
                        print("    region_image: [present]")
                    
                    # Print which box these visual tokens correspond to
                    if step.bounding_boxes:
                        last_box = step.bounding_boxes[-1]
                        print(f"      From box: [{last_box[0]:.4f}, {last_box[1]:.4f}, {last_box[2]:.4f}, {last_box[3]:.4f}]")
                else:
                    print("    region_image: [none]")
                    print(f"    visual Tokens: [none]")
                
                # Print step completion status
                if step.region_image is not None:
                    print("    Step Status: COMPLETED (has visual tokens)")
                else:
                    print("    Step Status: IN PROGRESS")
            
            # Print current step status
            current_step = request_history[-1] if request_history else None
            if current_step and current_step.region_image is None:
                print(f"\n  Current Step {len(request_history)-1} is still in progress")
                print(f"  Waiting for zoom-in token: {self.ZOOMIN_TOKEN_ID}")
        
        print("\n" + "="*80 + "\n")
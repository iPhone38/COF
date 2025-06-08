# COF (Chain-of-Focus) Visual Reasoning Project

## 📌 Overview

This repository implements a **Chain-of-Focus (COF)** mechanism for the Qwen2.5-VL visual language model, enabling dynamic region zooming and multi-step reasoning for complex visual tasks like object detection and visual question answering.

Key features:

- 🎯 **Bounding box generation** with `<|box_start|> [x1,y1,x2,y2] <|box_end|>` syntax
- 🔍 **Adaptive zoom-in** via `<|image_zoomin|>` tokens
- 🤖 **Multi-step reasoning** with `<think>...</think>` and `<answer>...</answer>` tags

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/iPhone38/COF.git
cd COF
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

为确保与 GPU 兼容，请安装支持 CUDA 的最新版本的 PyTorch、TorchVision 和 TorchAudio。即使已经安装了 PyTorch，您在运行 Web 应用程序时也可能会遇到问题，因此最好更新：

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. (Optional) Install with Mamba (faster than Conda)

```bash
mamba create -n cof python=3.10
mamba activate cof
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Basic Usage

```python
from qw_cof.model import Qwen2_5_VLForConditionalGeneration_COF
from transformers import AutoProcessor

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration_COF.from_pretrained(
    "path/to/Qwen2.5-VL-7B-Cof-Instruct",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("path/to/Qwen2.5-VL-7B-Cof-Instruct")

# Prepare input
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "basketball.jpg"},
        {"type": "text", "text": "What number is the shooting player? <|box_start|>[x1,y1,x2,y2]<|box_end|>"}
    ]
}]

# Generate response
inputs = processor(text=messages, images=image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=128)
```

---

## 🧠 How COF Works

### Chain-of-Focus Pipeline

1. **Initial Observation**: Model analyzes the whole image
2. **Region Proposal**: Generates bounding boxes for key regions
3. **Zoom-in**: Extracts high-resolution features from focused regions
4. **Multi-step Reasoning**: Combines visual and textual information iteratively

Example output format:

```text
<think>
Located player at [385,346,423,464]. 
Zoomed in to confirm jersey number is...
</think>
<answer>8</answer>
```

---

## 📂 File Structure

```
COF/
├── qw_cof/                # Core COF implementation
│   ├── model.py           # Modified Qwen2.5-VL model
│   └── utils/COF.py       # Focus mechanism logic
├── main.py                # Batch processing demo
├── testcof.py             # Interactive testing
└── requirements.txt       # Dependencies
```

---

## 🌟 Key Components

### `COF` Class (in `utils/COF.py`)

- `process_next_tokens()`: Main processing pipeline
- `detect_zoomin()`: Handles region zooming
- `append_visual_tokens()`: Merges visual features with text tokens

### Custom Model

`Qwen2_5_VLForConditionalGeneration_COF` extends the base Qwen model with:

- COF-aware generation
- Dynamic visual token integration
- Bounding box parsing

---

## 💡 Example Tasks

TO DO

---

## ⚠️ Notes

- Requires GPU with ≥16GB VRAM for 7B model
- Image paths in demos are relative - modify for your system
- For custom datasets, implement your own `process_vision_info()` function

---

## 📜 License

[Apache 2.0](LICENSE) (Same as Qwen base model)

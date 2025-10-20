import torch
import matplotlib.pyplot as plt
from peft import PeftModel

# === Load LoRA-enhanced model ===
base_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
lora_path = r"C:\Personal\Educational\Projects\NLP-Reg\trained_regressor.pt"

# Load only LoRA weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PeftModel.from_pretrained(base_model_name, lora_path, device_map="auto")

# === Collect LoRA weights ===
lora_weights = {}
for name, param in model.named_parameters():
    if "lora" in name:
        lora_weights[name] = param.detach().cpu().numpy()

# === Visualization ===
for name, weight in lora_weights.items():
    plt.figure(figsize=(6,4))
    plt.hist(weight.flatten(), bins=100, color='skyblue')
    plt.title(f"LoRA Parameter Distribution: {name}")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

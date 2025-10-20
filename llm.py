# === Imports ===
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

# === TensorBoard Setup ===
writer = SummaryWriter(log_dir="./runs/price_regressor")

# === Load Data ===
data = np.load(r"C:\Personal\Educational\Projects\NLP-Reg\LLM_training_preprocessed.npz", allow_pickle=True)
df = pd.DataFrame({key: data[key] for key in data.files})
df = df[['sample_id', 'catalog_content', 'log_price', 'img_pca_128']]
df = df.sample(n=30000, random_state=42).reset_index(drop=True)
print(df.head())

# === Standardize Image Embeddings ===
img_array = np.stack(df['img_pca_128'].values)
scaler = StandardScaler()
img_array = scaler.fit_transform(img_array)
df['img_pca_128'] = list(img_array)

# === Helper Functions ===
def smape(y_true, y_pred):
    return 100 * torch.mean(2 * torch.abs(y_pred - y_true) / (torch.abs(y_true) + torch.abs(y_pred) + 1e-8))

def anti_log(x):
    x = torch.clamp(x, min=-10, max=20)  # clamp to avoid overflow
    return torch.exp(x) - 1

# === Dataset ===
class PriceDataset(Dataset):
    def __init__(self, df, add_noise=False, noise_std=0.01):
        self.texts = df['catalog_content'].tolist()
        self.img_embs = df['img_pca_128'].tolist()
        self.labels = df['log_price'].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self.add_noise = add_noise
        self.noise_std = noise_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        img_emb = torch.tensor(self.img_embs[idx], dtype=torch.float32)
        if self.add_noise:
            img_emb += torch.randn_like(img_emb) * self.noise_std
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "img_emb": img_emb, "label": label}

# === Regression Head with LayerNorm + Dropout ===
class RegressionHead(nn.Module):
    def __init__(self, hidden_size, img_emb_size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size + img_emb_size)
        # Optional BatchNorm instead of LayerNorm (for larger batch sizes)
        # self.norm = nn.BatchNorm1d(hidden_size + img_emb_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + img_emb_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, text_feat, img_feat):
        x = torch.cat([text_feat, img_feat], dim=-1)
        x = self.norm(x)
        return torch.clamp(self.fc(x).squeeze(-1), min=-10, max=20)  # clamp output for stability

# === Load Base Model with 4-bit Quantization ===
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModel.from_pretrained(base_model_name, quantization_config=bnb_config, device_map="auto")
model.gradient_checkpointing_enable()

# Freeze base model
for param in model.parameters():
    param.requires_grad = False

# === Apply LoRA ===
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)

# === Regression Head ===
text_hidden_size = model.config.hidden_size
img_emb_size = len(df['img_pca_128'][0])
regression_head = RegressionHead(text_hidden_size, img_emb_size).to(device)

# === Train/Test Split (90/10) ===
dataset = PriceDataset(df, add_noise=True)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# === Optimizer & Cosine Scheduler with Warmup ===
optimizer = torch.optim.AdamW(list(regression_head.parameters()), lr=1e-4)
total_steps = len(train_loader) * 5
warmup_steps = max(1, int(0.01 * total_steps))  # 1% warmup
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

# === Training Loop with Early Stopping + Gradient Clipping ===
epochs = 5
early_stop_patience = 2
best_val_smape = np.inf
patience_counter = 0

model.eval()  # LoRA keeps model frozen
regression_head.train()

for epoch in range(epochs):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        img_emb = batch['img_emb'].to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = outputs.last_hidden_state.mean(dim=1)
            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-8)
            img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
            pred_log = regression_head(text_feat, img_emb)
            pred_price = anti_log(pred_log)
            true_price = anti_log(label)
            loss = smape(true_price, pred_price)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(regression_head.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()
        if batch_idx >= warmup_steps:
            scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"Batch SMAPE": loss.item()})
        writer.add_scalar("Batch_SMAPE", loss.item(), epoch * len(train_loader) + batch_idx)

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} Completed | Avg Train SMAPE: {avg_loss:.4f}")

    # Save checkpoint
    torch.save({
        'regression_head_state_dict': regression_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch+1
    }, r"C:\Personal\Educational\Projects\NLP-Reg\trained_regressor.pt")
    print(f"💾 Model checkpoint saved")

    # --- Validation for Early Stopping ---
    regression_head.eval()
    val_smape = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            img_emb = batch['img_emb'].to(device)
            label = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = outputs.last_hidden_state.mean(dim=1)
            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-8)
            img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
            pred_log = regression_head(text_feat, img_emb)
            pred_price = anti_log(pred_log)
            true_price = anti_log(label)
            val_smape += smape(true_price, pred_price).item()

    val_smape /= len(test_loader)
    writer.add_scalar("Val_SMAPE", val_smape, epoch)
    print(f"🔹 Validation SMAPE: {val_smape:.4f}")

    if val_smape < best_val_smape:
        best_val_smape = val_smape
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("⏹ Early stopping triggered")
            break
    regression_head.train()

# === Final Evaluation & Metrics ===
regression_head.eval()
model.eval()
all_true, all_pred = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Final Evaluation"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        img_emb = batch['img_emb'].to(device)
        label = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state.mean(dim=1)
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-8)
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
        pred_log = regression_head(text_feat, img_emb)
        pred_price = anti_log(pred_log)
        true_price = anti_log(label)

        all_true.extend(true_price.cpu().numpy())
        all_pred.extend(pred_price.cpu().numpy())

all_true = np.array(all_true)
all_pred = np.array(all_pred)
r2 = r2_score(all_true, all_pred)
rmse = np.sqrt(mean_squared_error(all_true, all_pred))
smape_val = 100 * np.mean(2 * np.abs(all_pred - all_true) / (np.abs(all_pred) + np.abs(all_true) + 1e-8))
print(f"🎯 Final Metrics -> R2: {r2:.4f}, RMSE: {rmse:.4f}, SMAPE: {smape_val:.4f}")

# === Plots ===
plt.figure(figsize=(8,6))
plt.scatter(all_true, all_pred, alpha=0.5)
plt.plot([all_true.min(), all_true.max()], [all_true.min(), all_true.max()], 'r--')
plt.xlabel("Expected Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Expected Prices")
plt.grid(True)
plt.savefig(r"C:\Personal\Educational\Projects\NLP-Reg\predicted_vs_expected.png", dpi=300)
plt.show()

plt.figure(figsize=(8,6))
plt.hist(all_pred - all_true, bins=50, color='skyblue')
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Histogram of Prediction Errors")
plt.grid(True)
plt.savefig(r"C:\Personal\Educational\Projects\NLP-Reg\prediction_error_hist.png", dpi=300)
plt.show()

# === Close TensorBoard Writer ===
writer.close()

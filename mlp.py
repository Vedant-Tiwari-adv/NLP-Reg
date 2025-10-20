# mlp_pipeline_with_clean_plot.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ Dataset Class
# ==============================
class PCAEmbeddingDataset(Dataset):
    def __init__(self, df):
        self.df = df.copy()
        if 'price' in self.df.columns:
            self.df.drop(columns=['price'], inplace=True, errors='ignore')
        
        self.txt_cols = [c for c in self.df.columns if c.startswith('txt_pca_')]
        self.img_cols = [c for c in self.df.columns if c.startswith('img_pca_')]
        self.numeric_cols = [c for c in self.df.columns if c not in self.txt_cols + self.img_cols + ['log_price','sample_id']]
        
        if self.numeric_cols:
            self.scaler_num = StandardScaler()
            self.df[self.numeric_cols] = self.scaler_num.fit_transform(self.df[self.numeric_cols])
        else:
            self.scaler_num = None
        
        self.X_txt = self.df[self.txt_cols].values.astype(np.float32)
        self.X_img = self.df[self.img_cols].values.astype(np.float32)
        self.X_num = self.df[self.numeric_cols].values.astype(np.float32) if self.numeric_cols else None
        self.y = self.df['log_price'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        txt = torch.tensor(self.X_txt[idx], dtype=torch.float32)
        img = torch.tensor(self.X_img[idx], dtype=torch.float32)
        num = torch.tensor(self.X_num[idx], dtype=torch.float32) if self.X_num is not None else torch.tensor([], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return txt, img, num, y

# ==============================
# 2️⃣ Metrics
# ==============================
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ==============================
# 3️⃣ Model Components
# ==============================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn_txt_to_img = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_img_to_txt = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj_txt = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.proj_img = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
    def forward(self, txt_feat, img_feat):
        txt_seq = txt_feat.unsqueeze(1)
        img_seq = img_feat.unsqueeze(1)
        txt_out, _ = self.attn_txt_to_img(txt_seq, img_seq, img_seq)
        img_out, _ = self.attn_img_to_txt(img_seq, txt_seq, txt_seq)
        txt_out = self.proj_txt(txt_out.squeeze(1))
        img_out = self.proj_img(img_out.squeeze(1))
        return txt_out, img_out

class SEFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim//reduction, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class ImprovedTwoBranchRegressor(nn.Module):
    def __init__(self, txt_dim, img_dim, num_dim=0, branch_dim=512, num_heads=8):
        super().__init__()
        self.txt_proj = nn.Sequential(nn.Linear(txt_dim, branch_dim), nn.BatchNorm1d(branch_dim), nn.ReLU(), nn.Dropout(0.2))
        self.img_proj = nn.Sequential(nn.Linear(img_dim, branch_dim), nn.BatchNorm1d(branch_dim), nn.ReLU(), nn.Dropout(0.2))
        self.txt_res1 = ResidualBlock(branch_dim, 0.2)
        self.txt_res2 = ResidualBlock(branch_dim, 0.15)
        self.img_res1 = ResidualBlock(branch_dim, 0.2)
        self.img_res2 = ResidualBlock(branch_dim, 0.15)
        self.cross_attn = CrossAttentionFusion(branch_dim, num_heads)
        
        if num_dim > 0:
            self.num_branch = nn.Sequential(nn.Linear(num_dim, 128), nn.ReLU(), nn.Dropout(0.1))
            combined_dim = branch_dim*2 + 128
        else:
            self.num_branch = None
            combined_dim = branch_dim*2
        
        self.se = SEFusion(combined_dim, reduction=8)
        self.head_fc1 = nn.Sequential(nn.Linear(combined_dim, 512), nn.ReLU(), nn.Dropout(0.2))
        self.head_res = ResidualBlock(512, 0.15)
        self.head_out = nn.Linear(512,1)
    
    def forward(self, txt, img, num=None):
        t = self.txt_proj(txt)
        i = self.img_proj(img)
        t = self.txt_res1(t)
        t = self.txt_res2(t)
        i = self.img_res1(i)
        i = self.img_res2(i)
        t_enr, i_enr = self.cross_attn(t,i)
        t_comb = t + t_enr
        i_comb = i + i_enr
        if self.num_branch is not None and num is not None and num.numel()>0:
            n = self.num_branch(num)
            combined = torch.cat([t_comb,i_comb,n],dim=1)
        else:
            combined = torch.cat([t_comb,i_comb],dim=1)
        combined = self.se(combined)
        x = self.head_fc1(combined)
        x = self.head_res(x)
        return self.head_out(x).squeeze(1)

# ==============================
# 4️⃣ Training and Evaluation
# ==============================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for txt, img, num, y in loader:
        txt,img,num,y = txt.to(device), img.to(device), num.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(txt,img,num)
        loss = criterion(y_pred,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for txt, img, num, y in loader:
            txt,img,num,y = txt.to(device), img.to(device), num.to(device), y.to(device)
            y_pred = model(txt,img,num)
            preds.append(y_pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return preds, targets

# ==============================
# 5️⃣ Train/Test Pipeline with Clean Plot
# ==============================
def run_train_test(df, test_size=0.2, epochs=40, batch_size=64, lr=1e-3, weight_decay=1e-5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_dataset = PCAEmbeddingDataset(train_df)
    test_dataset = PCAEmbeddingDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = ImprovedTwoBranchRegressor(
        txt_dim=train_dataset.X_txt.shape[1],
        img_dim=train_dataset.X_img.shape[1],
        num_dim=train_dataset.X_num.shape[1] if train_dataset.X_num is not None else 0,
        branch_dim=512,
        num_heads=8
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        if (epoch+1)%10 == 0 or epoch==0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.6f}")
    
    # --- Evaluate ---
    preds_log, targets_log = evaluate(model, test_loader, device)
    preds_price = np.exp(preds_log)
    targets_price = np.exp(targets_log)
    
    # --- Remove extreme outliers for plot ---
    lower = np.percentile(targets_price, 2.5)
    upper = np.percentile(targets_price, 97.5)
    mask = (targets_price >= lower) & (targets_price <= upper)
    preds_plot = preds_price[mask]
    targets_plot = targets_price[mask]
    
    val_smape = smape(targets_price, preds_price)
    val_rmse = rmse(targets_price, preds_price)
    val_r2 = r2_score(targets_price, preds_price)
    
    # --- Plot ---
    plt.figure(figsize=(6,6))
    plt.scatter(targets_plot, preds_plot, alpha=0.6)
    plt.plot([targets_plot.min(), targets_plot.max()], [targets_plot.min(), targets_plot.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual (Price)")
    plt.grid(True)
    # Show metrics as "hero" text
    plt.text(0.05, 0.95, f"SMAPE: {val_smape:.2f}\nRMSE: {val_rmse:.2f}\nR²: {val_r2:.4f}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.6))
    plt.tight_layout()
    plt.show()
    
    return model

# ==============================
# 6️⃣ Run
# ==============================
if __name__ == "__main__":
    data_path = r"C:\Personal\Educational\Projects\NLP-Reg\df_shoovitencoded.npz"
    data = np.load(data_path, allow_pickle=True)
    df = pd.DataFrame({key: data[key] for key in data.files})
    
    model = run_train_test(df, test_size=0.2, epochs=40, batch_size=64)

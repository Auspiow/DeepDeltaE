# siamese_color_train_cn.py
# 依赖：pip install numpy pandas matplotlib seaborn scipy torch torchvision colormath colour-science

import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
import json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 解决 Windows 下 OpenMP 冲突

# 颜色转换
from colormath.color_objects import XYZColor, LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# colormath 老版本兼容：避免 numpy.asscalar 缺失
if not hasattr(np, "asscalar"):
    np.asscalar = lambda x: x.item()

plt.rcParams['figure.figsize'] = (7, 5)
sns.set_theme(style="whitegrid")

# ============================================================
# 0) 全局配置
# ============================================================
DATASET_DIR = "datasets"      # JSON 数据目录
OUTPUT_DIR = "output"         # 输出文件目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
OUTPUT_DIR_ALGO = "output/siamese_weights" # 导出权重的新目录
MAX_EXPORT_SAMPLES = 500 # 导出样本数量

# ============================================================
# 1) 加载 JSON 数据集：返回 DataFrame(L1,a1,b1,L2,a2,b2,DE_human)
#    文件格式兼容 rit-dupont 等常见格式
# ============================================================
def load_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    dv = np.array(d.get("dv", []))
    pairs = np.array(d.get("pairs", []), dtype=int)
    xyz = np.array(d.get("xyz", []), dtype=float)

    # 无效数据则返回空 DataFrame
    if xyz.size == 0 or pairs.size == 0 or dv.size == 0:
        return pd.DataFrame(columns=["L1","a1","b1","L2","a2","b2","DE_human"])

    # xyz -> lab（假设 D65, 2° 观察者）
    lab_list = []
    for (x,y,z) in xyz:
        xyz_obj = XYZColor(x, y, z, observer='2', illuminant='d65')
        lab = convert_color(xyz_obj, LabColor)
        lab_list.append([lab.lab_l, lab.lab_a, lab.lab_b])
    lab_arr = np.array(lab_list)

    # 组合成行
    rows = []
    n = min(len(pairs), len(dv))
    for idx in range(n):
        i, j = pairs[idx]
        score = float(dv[idx])
        L1, a1, b1 = lab_arr[i]
        L2, a2, b2 = lab_arr[j]
        rows.append([L1,a1,b1,L2,a2,b2,score])

    return pd.DataFrame(rows, columns=["L1","a1","b1","L2","a2","b2","DE_human"])

# 扫描目录并加载
dfs = []
for fname in os.listdir(DATASET_DIR):
    if fname.lower().endswith(".json"):
        full = os.path.join(DATASET_DIR, fname)
        print("正在加载", full)
        try:
            df = load_json_dataset(full)
            if len(df):
                dfs.append(df)
        except Exception as e:
            print("读取失败:", full, e)

if not dfs:
    raise RuntimeError("datasets 目录中未找到可用 JSON 文件！")

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.dropna().reset_index(drop=True)
print("总共载入样本对数:", len(df_all))

# ============================================================
# 简单统计人类 ΔE 分布
# ============================================================
print(df_all["DE_human"].describe())

plt.figure()
sns.histplot(df_all["DE_human"].values, bins=60, kde=False)
plt.title("人类 ΔE 分布（原始）")
plt.savefig(os.path.join(OUTPUT_DIR, "hist_DE_human_raw.png"), dpi=150)
plt.close()

# ============================================================
# 2) 数据增强：
#    - 对调颜色 (c1<->c2)，使模型具有对称性
#    - 目标值使用 log1p 减弱长尾
# ============================================================
X = df_all[["L1","a1","b1","L2","a2","b2"]].values.astype(np.float32)
y_raw = df_all["DE_human"].values.astype(np.float32).reshape(-1,1)

# 对调增强
X_swap = X.copy()
X_swap[:, :3], X_swap[:, 3:] = X[:, 3:], X[:, :3]
y_swap = y_raw.copy()

X = np.concatenate([X, X_swap], axis=0)
y_raw = np.concatenate([y_raw, y_swap], axis=0)

# log1p 目标值
y_log = np.log1p(y_raw)

# 归一化
y_mean, y_std = y_log.mean(), y_log.std()
y_norm = (y_log - y_mean) / (y_std + 1e-9)

# 保存归一化常数
norm_constants = {
    "y_mean": float(y_mean),
    "y_std": float(y_std)
}
with open(os.path.join(OUTPUT_DIR_ALGO, "norm_constants.json"), "w") as f:
    json.dump(norm_constants, f, indent=2)

print(f"归一化常数 (Mean:{y_mean:.4f}, Std:{y_std:.4f}) 已保存到 norm_constants.json。")

print("增强后样本数:", len(y_norm))

# ============================================================
# 3) 平衡采样：
#    - ΔE 分桶
#    - 按逆频率分配权重
# ============================================================
num_bins = 10
bins = np.linspace(0.0, max(y_raw.max(), 1.0), num_bins+1)
bin_idx = np.digitize(y_raw.ravel(), bins) - 1
bin_idx = np.clip(bin_idx, 0, num_bins-1)

counts = np.bincount(bin_idx, minlength=num_bins).astype(float)
weights = 1.0 / (counts[bin_idx] + 1e-9)
weights = weights * (len(weights) / weights.sum())

# 转换为 Tensor
X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y_norm.astype(np.float32)).squeeze(-1)

dataset = TensorDataset(X_t, y_t)
sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

# 划分训练/验证集
train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size
indices = list(range(len(dataset)))
np.random.shuffle(indices)
train_idx, val_idx = indices[:train_size], indices[train_size:]

train_loader = DataLoader(
    dataset, batch_size=BATCH_SIZE,
    sampler=WeightedRandomSampler(weights[train_idx], num_samples=len(train_idx), replacement=True)
)
val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

print("训练集大小:", len(train_idx), "验证集大小:", len(val_idx))

# ============================================================
# 4) 模型定义：Siamese 编码器 + 距离 MLP 预测
# ============================================================
class SiameseColorNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        # (L,a,b) → 嵌入向量
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
            nn.ReLU()
        )
        # |e1 - e2| → 预测 log1p(DE)（归一化后的）
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, 1)
        )

    def forward(self, x):
        B = x.shape[0]
        colors = x.view(B, 2, 3)
        c1, c2 = colors[:,0,:], colors[:,1,:]
        e1, e2 = self.encoder(c1), self.encoder(c2)
        d = torch.abs(e1 - e2)
        out = self.head(d).squeeze(-1)
        return out

model = SiameseColorNet(emb_dim=128).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
loss_fn = nn.HuberLoss(delta=1.0)

# ============================================================
# 5) 训练循环
# ============================================================
best_val = 1e9
save_path = os.path.join(OUTPUT_DIR, "checkpoints_siamese.pth")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for epoch in range(1, EPOCHS+1):
    model.train()
    t0 = time.time()
    running_loss = 0.0
    n_seen = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE).float()
        yb = yb.to(DEVICE).float()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += loss.item() * xb.size(0)
        n_seen += xb.size(0)

    train_loss = running_loss / (n_seen + 1e-9)

    # ---- 验证 ----
    model.eval()
    vloss, vcount = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            pred = model(xb)
            vloss += ((pred - yb)**2).sum().item()
            vcount += xb.size(0)

    val_mse = vloss / (vcount + 1e-9)

    print(f"Epoch {epoch:02d}   训练loss={train_loss:.6f}   验证MSE={val_mse:.6f}   时间={time.time()-t0:.1f}s")

    if val_mse < best_val:
        best_val = val_mse
        torch.save(model.state_dict(), save_path)
        print(" 已保存当前最佳模型。")

# ============================================================
# 6) 模型评估：计算 Pearson R，与 DE2000 对比
# ============================================================
model.load_state_dict(torch.load(save_path, map_location=DEVICE))
model.eval()

with torch.no_grad():
    preds_norm = model(X_t.to(DEVICE).float()).cpu().numpy().reshape(-1,1)

# 反归一化
preds_log = preds_norm * (y_std + 1e-9) + y_mean
preds_un = np.expm1(preds_log)

true_all = y_raw.reshape(-1,1)

# ΔE2000 参考
def de2000_array_from_df(Xin):
    out = []
    for row in Xin:
        c1 = LabColor(row[0], row[1], row[2])
        c2 = LabColor(row[3], row[4], row[5])
        val = delta_e_cie2000(c1, c2)
        if hasattr(val, "item"):
            val = val.item()
        out.append(float(val))
    return np.array(out).reshape(-1,1)

print("正在计算 DE2000 基线（可能较慢）...")
de2000_vals = de2000_array_from_df(X)

# Pearson R
r_model, _ = pearsonr(preds_un.ravel(), true_all.ravel())
r_de2000, _ = pearsonr(de2000_vals.ravel(), true_all.ravel())

print(f"模型相关系数 R(model)   = {r_model:.4f}")
print(f"DE2000 相关系数 R(DE2000) = {r_de2000:.4f}")

# ============================================================
# 6.5) 【重要修改】将 Lab 转换回 RGB，导出完整数据集
# ============================================================
def lab_to_rgb(L, a, b):
    """
    使用 colormath 库将 Lab 转换回 sRGB。
    Lab -> XYZ -> sRGB，并正确处理边界和伽马校正。
    """
    try:
        lab_obj = LabColor(L, a, b)
        rgb_obj = convert_color(lab_obj, sRGBColor)
        
        r = rgb_obj.rgb_r
        g = rgb_obj.rgb_g
        b = rgb_obj.rgb_b
        
        # 裁剪到 [0, 1] 范围，并转换为 [0, 255] 的整数
        r_int = int(np.clip(r, 0, 1) * 255)
        g_int = int(np.clip(g, 0, 1) * 255)
        b_int = int(np.clip(b, 0, 1) * 255)
        
        return r_int, g_int, b_int
    
    except Exception as e:
        # 转换失败时，返回一个中性灰，防止程序崩溃
        return 128, 128, 128 


# 构建一个包含所有必要数据的数组
full_data = np.hstack([X, true_all, de2000_vals, preds_un]) # L1, a1, b1, L2, a2, b2, human, de2000, model

# 1. 创建随机索引 (大范围随机采样)
np.random.shuffle(full_data)
# 仅保留最大导出数量的样本
data_to_export = full_data[:min(len(full_data), len(full_data))] # 实际上是全部，但先保留命名

export_data = []
rgb_seen = set() # 用于去重：存储 (r1, g1, b1, r2, g2, b2) 的元组

# 2. 遍历并执行转换和去重
for row in data_to_export:
    L1, a1, b1, L2, a2, b2, human_score, de2000_score, model_score = row

    r1, g1, b1_rgb = lab_to_rgb(L1, a1, b1)
    r2, g2, b2_rgb = lab_to_rgb(L2, a2, b2)

    # **去重逻辑：检查转换后的 RGB 对是否重复**
    # 颜色顺序无关，所以我们规范化顺序
    rgb_pair = tuple(sorted((r1, g1, b1_rgb, r2, g2, b2_rgb)))

    if rgb_pair in rgb_seen:
        continue # 跳过重复的 RGB 颜色对

    rgb_seen.add(rgb_pair)

    entry = {
        "color1": {"r": int(r1), "g": int(g1), "b": int(b1_rgb)},
        "color2": {"r": int(r2), "g": int(g2), "b": int(b2_rgb)},
        "human_score": round(float(human_score), 2),
        "e2000_score": round(float(de2000_score), 2),
        "model_score": round(float(model_score), 2)
    }
    export_data.append(entry)

    # 3. 限制导出数量
    if len(export_data) >= MAX_EXPORT_SAMPLES:
        break


# 导出为 JSON
export_json_path = os.path.join(OUTPUT_DIR, "color_comparison_results.json")
with open(export_json_path, "w", encoding="utf-8") as f:
    json.dump(export_data, f, indent=2, ensure_ascii=False)

print(f"\n已导出数据到 {export_json_path}，共 {len(export_data)} 条【去重和随机采样后】的样本")

# ============================================================
# 7) 可视化结果 (保持不变)
# ============================================================
plt.figure()
plt.scatter(true_all, preds_un, s=6, alpha=0.4)
plt.xlabel("人类评分（ΔE 原始）")
plt.ylabel("模型预测（ΔE 原始）")
plt.title(f"Siamese 模型 vs 人类 (R={r_model:.4f})")
mx = max(true_all.max(), preds_un.max())
plt.plot([0,mx],[0,mx], 'r--')
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_siamese_pred_vs_human.png"), dpi=150)
plt.close()

# 误差直方图
err = (preds_un.ravel() - true_all.ravel())
plt.figure()
sns.histplot(err, bins=80, kde=True)
plt.title("预测误差分布（pred - human）")
plt.savefig(os.path.join(OUTPUT_DIR, "hist_error_siamese.png"), dpi=150)
plt.close()

# R 对比
plt.figure()
labels = ["Siamese", "ΔE2000"]
vals = [r_model, r_de2000]
sns.barplot(x=labels, y=vals)
plt.ylim(0,1)
plt.title("Pearson R 对比")
plt.savefig(os.path.join(OUTPUT_DIR, "r_comparison_siamese.png"), dpi=150)
plt.close()

print(f"已保存图像：{OUTPUT_DIR}/scatter_siamese_pred_vs_human.png, {OUTPUT_DIR}/hist_error_siamese.png, {OUTPUT_DIR}/r_comparison_siamese.png")
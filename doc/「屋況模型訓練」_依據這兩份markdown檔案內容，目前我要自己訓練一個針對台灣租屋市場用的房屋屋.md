<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 「屋況模型訓練」

依據這兩份markdown檔案內容，目前我要自己訓練一個針對台灣租屋市場用的房屋屋內狀況檢測模型，我該怎麼在我的RTX 3060 Twin Edge 12GB 192BIT GDDR6上做訓練？請提供具體步驟。

根據你兩份 Markdown 文件的完整內容，以下是針對你的 **RTX 3060 12GB** 在台灣租屋屋況檢測場景下，從零開始訓練 Layer 3（Fine-tuned ViT）的具體步驟。

***

## 環境準備

首先確認你的訓練環境：

```bash
# 1. 安裝 CUDA 對應的 PyTorch（RTX 3060 支援 CUDA 12.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. 安裝其他依賴
pip install pillow scikit-learn
```

RTX 3060 12GB 對於 `vit_b_16` 的訓練已綽綽有餘。使用 `BATCH_SIZE=32` 約佔用 6~8GB VRAM，192-bit 記憶體頻寬在此任務下不是瓶頸。[^1]

***

## Step 1：收集台灣租屋資料集

這是最重要的步驟。你需要建立以下 **6 類**資料夾結構，每類**至少 200 張** 224×224 的 patch 圖片，共 1200 張以上 ：[^1]

```
data/defects/
├── normal/    ← 正常無缺陷室內表面
├── crack/     ← 牆面、地板裂縫
├── stain/     ← 污漬、水漬
├── mold/      ← 霉斑（黑色、綠色點狀）
├── peeling/   ← 剝落的漆面、壁紙
└── worn/      ← 磁磚破損、木地板刮傷
```

**資料來源建議（三管齊下）：**

- **BD3 Dataset**：`https://github.com/Praveenkottari/BD3-Dataset`，包含 crack/spalling/stain 等類別[^1]
- **DAGM 2007**：工業表面缺陷，可對應 stain/crack[^1]
- **自行拍攝**（強烈建議）：從台灣實際租屋物件照片中截取牆面/地板/天花板 patch，這對台灣場景泛化至關重要[^1]

> **資料不足的過渡方案**：若短期無法收集到 1200 張，每類只需 30~50 張，改用 CLIP Linear Probe 方案（不需 GPU，sklearn LogisticRegression），準確率可達 80~90%，之後再換成完整 ViT 。[^2][^1]

***

## Step 2：執行訓練腳本

將以下腳本存為 `train_condition_model.py`，核心設定針對 RTX 3060 已調整 ：[^1]

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from pathlib import Path
from PIL import Image

CLASSES     = ["normal", "crack", "stain", "mold", "peeling", "worn"]
DATA_DIR    = "data/defects"
SAVE_PATH   = "condition_model.pt"
EPOCHS      = 40
BATCH_SIZE  = 32    # RTX 3060 12GB 可穩定跑此大小
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train"):
        self.samples, self.transform = [], transform
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = Path(root_dir) / cls_name
            if not cls_dir.exists(): continue
            all_imgs = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")))
            n = len(all_imgs); split_idx = int(n * 0.8)
            imgs = all_imgs[:split_idx] if split == "train" else all_imgs[split_idx:]
            for img_path in imgs:
                self.samples.append((str(img_path), cls_idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = DefectDataset(DATA_DIR, train_transform, "train")
val_ds   = DefectDataset(DATA_DIR, val_transform,   "val")
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 載入 ImageNet 預訓練 ViT-B/16，換掉分類頭
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASSES))
model = model.to(DEVICE)

# 凍結前 6 個 Transformer block，只訓練後半 + 分類頭（省 VRAM）
for i, block in enumerate(model.encoder.layers):
    for p in block.parameters():
        p.requires_grad = (i >= 6)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward(); optimizer.step()
        train_loss += loss.item()
        correct += (model(imgs).argmax(1) == labels).sum().item(); total += labels.size(0)
    model.eval(); val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            val_correct += (model(imgs).argmax(1) == labels).sum().item(); val_total += labels.size(0)
    val_acc = val_correct / val_total; scheduler.step()
    print(f"[Epoch {epoch:3d}/{EPOCHS}] loss:{train_loss/len(train_dl):.4f} | val_acc:{val_acc:.3f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ 儲存最佳模型 val_acc={val_acc:.3f}")
```


***

## Step 3：RTX 3060 專屬調校建議

| 問題 | 解法 |
| :-- | :-- |
| VRAM 不足（OOM） | `BATCH_SIZE` 調降至 16，或改用 `vit_s_16`（Small）代替 B/16 |
| 訓練太慢 | 加上 `torch.cuda.amp.autocast()` 開啟 FP16 混合精度，速度可提升 ~40% |
| 過擬合（資料少） | 已有 ColorJitter/Flip 資料擴增；另可加 `transforms.GaussianBlur(3)` |
| 凍結層數調整 | 資料量少（<500張）建議凍結更多層（`i >= 9`）；資料量充足可全部解凍 |

開啟 FP16 的方式在訓練迴圈中加入：[^1]

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    out = model(imgs)
    loss = criterion(out, labels)
scaler.scale(loss).backward()
scaler.step(optimizer); scaler.update()
```


***

## Step 4：接入主系統

訓練完成後，修改你的 `scene_semantic_analysis_v2.py` 兩個地方即可啟用 Layer 3 ：[^1]

```python
# 1. 指定模型路徑
CONDITION_MODEL_PATH = "condition_model.pt"

# 2. 初始化時自動載入
analyzer = RentalListingAnalyzer(
    enable_furniture_detection=True,
    enable_condition_inspection=True,
)
```

成功載入後 log 會顯示 `[ConditionInspector] Layer3 ViT 模型已載入` 且 `Layer3=啟用`；若顯示停用，代表路徑不對或模型尚未生成 。[^1]

***

## 各階段建議路線

在等待收集足夠資料期間，建議依序推進 ：[^2]

1. **立即（0成本）**：強化 Layer 2 的 CLIP Prompt，針對台灣租屋場景改寫（例如 `"black mold spots on a bathroom ceiling"`）
2. **短期（30分鐘）**：用 `cazzz307/yolov8-crack-detection` 替換 Layer 1 的 HoughLines 裂縫偵測，精準度大幅提升
3. **中期（2~3週）**：每類收集 30~50 張自拍室內照，做 CLIP Linear Probe，無需 GPU
4. **長期（資料量 ≥1200張）**：執行上述完整 ViT fine-tune 訓練腳本，得到最佳效果
<span style="display:none">[^3][^4]</span>

<div align="center">⁂</div>

[^1]: Wo-Xiang-Jia-Ru-Layer-3-fine-tuned-ViT-Jin-Xing-Geng-Jing-Zhun-De-Zhen-Ce-Zhi-You-Layer-1He-Laye.md

[^2]: Qing-Wen-Mu-Qian-You-Xun-Lian-Hao-De-Mo-Xing-Ke-Yi-Shi-Yong-Ma.md

[^3]: 3D Gaussian Splatting as a New Era_A Survey.pdf

[^4]: 3D Gaussian Splatting_ Survey, Technologies, Challenges, and Opportunities.pdf


<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 我想加入Layer 3（fine-tuned ViT）進行更精準的偵測，只有Layer 1和Layer 2的輸出結果不是很好。我該怎麼加入Layer 3：Fine-tuned ViT？

請問這三層架構方法是否可以單獨使用，抑或選擇性開啟使用？

三層架構完全可以獨立或選擇性使用。先回答這個問題，再說明如何訓練並接入 Layer 3。

***

## 三層架構的開關控制

目前程式碼中，三層是透過一個參數控制的：

```python
# CONDITION_MODEL_PATH = None   → 只用 Layer 1 + 2
# CONDITION_MODEL_PATH = "path" → Layer 1 + 2 + 3 全開
CONDITION_MODEL_PATH = None
```

但這樣的設計不夠靈活。建議改成獨立旗標，讓三層可以任意組合：

```python
# 在 ConditionInspector.__init__ 加入三個開關
def __init__(self, ...,
             enable_layer1: bool = True,   # OpenCV heuristic
             enable_layer2: bool = True,   # CLIP zero-shot
             enable_layer3: bool = False,  # Fine-tuned ViT
             condition_model_path: Optional[str] = None):
    self.enable_layer1 = enable_layer1
    self.enable_layer2 = enable_layer2
    self.enable_layer3 = enable_layer3 and (condition_model_path is not None)
    ...
```

`analyze()` 裡也對應改：

```python
def analyze(self, image_path, scene=""):
    l1 = self._layer1_heuristic(image_path) if self.enable_layer1 else {}
    l2 = self._layer2_clip(image_path)      if self.enable_layer2 else {}
    l3 = self._layer3_vit(image_path)       if self.enable_layer3 else None
    ...
```

這樣就可以四種組合自由切換：


| 組合 | 用途 |
| :-- | :-- |
| Layer 1 only | 最快速，純 rule-based，適合先快篩 |
| Layer 2 only | 語意理解，不需 GPU 也可跑 |
| Layer 1 + 2 | 目前預設，互補 |
| Layer 1 + 2 + 3 | 最完整，有訓練好的模型才開 |


***

## 如何加入 Layer 3：完整流程

### Step 1：準備訓練資料集

Layer 3 的 ViT 需要有標注的缺陷圖片，以下三個資料集可以組合使用：

**BD3（建議主要來源）**

```
https://github.com/Praveenkottari/BD3-Dataset
類別：crack / spalling / exposed-bars / corrosion / stain
```

**DAGM 2007（表面瑕疵）**

```
https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-inspection
類別：6 種工業表面缺陷，可對應 stain / crack
```

**自行收集補充（強烈建議）**
從你的實際室內影像中，截取牆面 / 地板 / 天花板的 patch，手動標注以下 6 類：

```
normal   → 正常無缺陷的室內表面
crack    → 裂縫（牆面、地板）
stain    → 污漬、水漬
mold     → 霉斑（黑色、綠色點狀）
peeling  → 剝落（漆面、壁紙）
worn     → 老舊（磁磚破損、木地板刮傷）
```

建議資料量：每類至少 200 張 patch（224×224），總計 1200 張以上。

***

### Step 2：訓練腳本

```python
# train_condition_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from pathlib import Path
from PIL import Image

# ── 資料集結構（資料夾對應類別）──
# data/defects/
#   ├── normal/       (*.jpg)
#   ├── crack/
#   ├── stain/
#   ├── mold/
#   ├── peeling/
#   └── worn/

CLASSES     = ["normal", "crack", "stain", "mold", "peeling", "worn"]
DATA_DIR    = "data/defects"
SAVE_PATH   = "condition_model.pt"
EPOCHS      = 40
BATCH_SIZE  = 32
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── Dataset ──
class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train"):
        self.samples   = []
        self.transform = transform
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = Path(root_dir) / cls_name
            if not cls_dir.exists():
                continue
            all_imgs = sorted(list(cls_dir.glob("*.jpg")) +
                              list(cls_dir.glob("*.png")))
            # 80/20 train/val split
            n = len(all_imgs)
            split_idx = int(n * 0.8)
            imgs = all_imgs[:split_idx] if split == "train" else all_imgs[split_idx:]
            for img_path in imgs:
                self.samples.append((str(img_path), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ── Transforms ──
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_ds = DefectDataset(DATA_DIR, train_transform, split="train")
val_ds   = DefectDataset(DATA_DIR, val_transform,   split="val")
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"訓練集：{len(train_ds)} 張 | 驗證集：{len(val_ds)} 張")

# ── 建立模型（ImageNet 預訓練 → fine-tune）──
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASSES))
model = model.to(DEVICE)

# 前 6 個 Transformer block 凍結，只更新後半 + 分類頭
for i, block in enumerate(model.encoder.layers):
    requires_grad = (i >= 6)
    for p in block.parameters():
        p.requires_grad = requires_grad

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# ── 訓練迴圈 ──
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    # --- train ---
    model.train()
    train_loss, correct, total = 0, 0, 0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    train_acc = correct / total

    # --- val ---
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            val_correct += (out.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)
    val_acc = val_correct / val_total
    scheduler.step()

    print(f"[Epoch {epoch:3d}/{EPOCHS}] "
          f"loss: {train_loss/len(train_dl):.4f} | "
          f"train acc: {train_acc:.3f} | val acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ 儲存最佳模型 (val_acc={val_acc:.3f})")

print(f"\n訓練完成！最佳驗證準確率：{best_val_acc:.3f}")
print(f"模型已儲存至：{SAVE_PATH}")
```


***

### Step 3：接入 `scene_semantic_analysis_v2.py`

訓練完後只需要改兩個地方：

```python
# 1. 設定模型路徑（全域配置區）
CONDITION_MODEL_PATH = "condition_model.pt"  # 改成你實際的路徑

# 2. 初始化時開啟 Layer 3
analyzer = RentalListingAnalyzer(
    enable_furniture_detection  = True,
    enable_condition_inspection = True,
)
# ConditionInspector.__init__ 偵測到 CONDITION_MODEL_PATH 非 None
# 且檔案存在時，會自動載入 Layer 3
```

程式碼裡的 `_load_condition_model()` 會自動處理載入，推論時 `_layer3_vit()` 就會回傳結果，`analyze()` 的決策邏輯也會自動優先採用 Layer 3 的輸出。

***

### Step 4：驗證 Layer 3 是否正確載入

執行後 log 會顯示：

```
[ConditionInspector] Layer3 ViT 模型已載入：condition_model.pt
[ConditionInspector] 就緒 ✓ (Layer3=啟用)
```

如果顯示 `Layer3=停用，使用 Layer1+2`，代表路徑不存在或模型尚未訓練。

***

## 資料不夠時的替代方案

如果短期內無法收集足夠的訓練資料，有一個折中方案：用 CLIP 做特徵提取，搭配少量標注做 **Linear Probe**：

```python
# 只需每類 20~50 張，用 CLIP 的 image feature 訓練一個線性分類器
from sklearn.linear_model import LogisticRegression
import pickle

# 提取所有訓練圖片的 CLIP feature
features, labels = [], []
for img_path, label in train_samples:
    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
    features.append(feat.cpu().numpy().squeeze())
    labels.append(label)

# 訓練線性分類器
clf = LogisticRegression(max_iter=1000, C=0.1)
clf.fit(features, labels)

# 儲存
with open("condition_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
```

這個方案只需要少量資料就能達到不錯的效果，可以作為 Layer 3 的過渡版本，等收集足夠資料後再換成完整的 fine-tuned ViT。


#!/usr/bin/env python3
"""
train.py
========
Room Inspector — Layer 3 訓練腳本
支援 ConvNeXt-T（小資料優先）和 Swin-T（資料量充足時）

使用方式：
  python train.py                          # 自動選擇架構
  python train.py --arch swin_t            # 指定 Swin-T
  python train.py --arch convnext_t        # 指定 ConvNeXt-T
  python train.py --resume checkpoints/best.pt  # 繼續訓練
  python train.py --stage linear           # 只做 CLIP Linear Probe（Stage 1）
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# ── 設定 ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data" / "defects"
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
LOG_DIR      = PROJECT_ROOT / "runs"

# 全部 6 類（訓練時若某類無資料會自動跳過）
ALL_CLASSES = ["normal", "crack", "stain", "mold", "peeling", "worn"]
CLASSES   = ALL_CLASSES   # 實際使用類別在 train() 內動態決定
N_CLASSES = len(CLASSES)

IMG_SIZE   = 224
PATCH_SIZE = 224

# 訓練超參數
EPOCHS_LINEAR   = 10    # Stage 1: Linear Probe
EPOCHS_FINETUNE = 30    # Stage 2/3: Fine-tune
BATCH_SIZE      = 32
WARMUP_EPOCHS   = 3
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY    = 0.01
BASE_LR         = 3e-4  # Linear Probe / 最後一層
FINETUNE_LR     = 1e-4  # Fine-tune（backbone）
MIN_LR          = 1e-6

# 最低資料量（Swin-T 需求更高）
MIN_SAMPLES_SWIN     = 200  # 每類至少 200 張
MIN_SAMPLES_CONVNEXT = 100  # 每類至少 100 張

SEED = 42


# ── 工具函式 ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_class_samples(data_dir: Path) -> dict:
    """統計各類別圖片數量。"""
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    counts = {}
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if cls_dir.exists():
            counts[cls] = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        else:
            counts[cls] = 0
    return counts


def compute_class_weights(counts: dict, active_classes: list) -> torch.Tensor:
    """
    計算 balanced class weights（只含有資料的類別）。
    公式：weight[i] = total / (n_classes * count[i])
    正規化使最小 weight = 1.0
    """
    total = sum(counts[c] for c in active_classes)
    n_cls = len(active_classes)
    weights = []
    for cls in active_classes:
        cnt = counts.get(cls, 1)
        w = total / (n_cls * max(cnt, 1))
        weights.append(w)
    min_w = min(weights)
    weights = [w / min_w for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


def auto_select_arch(counts: dict, active_classes: list) -> str:
    """根據資料量自動選擇訓練架構（只看有資料的類別）。"""
    min_count = min(counts[c] for c in active_classes)
    if min_count >= MIN_SAMPLES_SWIN:
        return "swin_t"
    elif min_count >= MIN_SAMPLES_CONVNEXT:
        return "convnext_t"
    else:
        print(f"[警告] 最少類別只有 {min_count} 張，建議先用 CLIP Linear Probe（--stage linear）")
        return "convnext_t"


# ── 資料增強 ──────────────────────────────────────────────────────────

def get_train_transforms():
    """完整訓練資料增強。"""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms():
    """驗證集標準前處理。"""
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),  # 256 for 224
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── 資料集 ────────────────────────────────────────────────────────────

def build_datasets(data_dir: Path, val_split: float = 0.15):
    """
    從 data/defects/ 建立訓練/驗證集（按比例切分）。
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = []
    for cls_idx, cls in enumerate(CLASSES):
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in IMG_EXTS:
                all_images.append((img_path, cls_idx))

    if not all_images:
        raise RuntimeError(f"在 {data_dir} 找不到任何圖片，請先執行資料整理腳本。")

    random.shuffle(all_images)

    # 按類別分層切分
    train_images, val_images = [], []
    by_class: dict = {i: [] for i in range(N_CLASSES)}
    for img_path, cls_idx in all_images:
        by_class[cls_idx].append((img_path, cls_idx))

    for cls_idx, items in by_class.items():
        n_val = max(1, int(len(items) * val_split))
        val_images.extend(items[:n_val])
        train_images.extend(items[n_val:])

    print(f"  訓練集：{len(train_images)} 張")
    print(f"  驗證集：{len(val_images)} 張")

    return train_images, val_images


class DefectDataset(torch.utils.data.Dataset):
    """自定義資料集，支援路徑列表輸入。"""

    def __init__(self, samples: list, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_weighted_sampler(train_samples: list) -> WeightedRandomSampler:
    """建立加權採樣器，緩解類別不平衡。"""
    label_counts = [0] * N_CLASSES
    for _, lbl in train_samples:
        label_counts[lbl] += 1

    class_weights = [1.0 / max(c, 1) for c in label_counts]
    sample_weights = [class_weights[lbl] for _, lbl in train_samples]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights),
        num_samples=len(train_samples),
        replacement=True,
    )


# ── 模型建立 ──────────────────────────────────────────────────────────

def build_model(arch: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    """建立模型（ConvNeXt-T 或 Swin-T）。"""
    try:
        import timm
    except ImportError:
        raise ImportError("請安裝 timm：pip install timm")

    arch_map = {
        "convnext_t": "convnext_tiny.fb_in22k_ft_in1k",
        "swin_t":     "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        # 備用（資料量非常少時）
        "convnext_s": "convnext_small.fb_in22k_ft_in1k",
        "swin_s":     "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    }

    model_name = arch_map.get(arch, arch)
    print(f"  載入預訓練模型：{model_name}")

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=n_classes,
    )
    return model


def freeze_backbone(model, arch: str):
    """凍結 backbone，只訓練分類頭。"""
    for name, param in model.named_parameters():
        # ConvNeXt 的分類頭叫 head，Swin 也叫 head
        if "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  凍結 backbone，可訓練參數：{n_train:,}")


def unfreeze_backbone(model, freeze_patch_embed: bool = True):
    """解凍全部參數（fine-tune 階段）。"""
    for name, param in model.named_parameters():
        if freeze_patch_embed and "patch_embed" in name:
            param.requires_grad = False  # Patch embedding 通常不需要 fine-tune
        else:
            param.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  解凍 backbone，可訓練參數：{n_train:,}")


# ── 學習率排程 ────────────────────────────────────────────────────────

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """Warmup + Cosine Annealing。"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            factor = self.min_lr / self.base_lrs[0] + (1 - self.min_lr / self.base_lrs[0]) * factor

        return [base_lr * factor for base_lr in self.base_lrs]


# ── 訓練單個 epoch ────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch:3d} [{batch_idx:4d}/{len(loader)}]"
                  f"  loss={loss.item():.4f}"
                  f"  acc={total_correct/total_samples:.4f}")

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc


# ── 儲存 / 讀取 checkpoint ────────────────────────────────────────────

def save_checkpoint(state: dict, is_best: bool, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"
    torch.save(state, last_path)
    if is_best:
        shutil.copy(last_path, best_path)
        print(f"  [✓] 儲存最佳 checkpoint → {best_path}")


def load_checkpoint(ckpt_path: Path, model, optimizer=None, scaler=None):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    start_epoch = state.get("epoch", 0) + 1
    best_val_acc = state.get("best_val_acc", 0.0)
    print(f"  已載入 checkpoint：epoch={state.get('epoch', 0)}"
          f"  best_val_acc={best_val_acc:.4f}")
    return start_epoch, best_val_acc


# ── Stage 1：CLIP Linear Probe ────────────────────────────────────────

def run_clip_linear_probe(data_dir: Path, ckpt_dir: Path):
    """
    CLIP Linear Probe：用 CLIP 特徵 + scikit-learn LogisticRegression。
    每類只需 30~50 張即可獲得合理基準。
    """
    print("\n" + "=" * 60)
    print("  Stage 1：CLIP Linear Probe")
    print("=" * 60)

    try:
        import clip
    except ImportError:
        raise ImportError("請安裝 CLIP：pip install openai-clip")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report
        import pickle
    except ImportError:
        raise ImportError("請安裝 scikit-learn：pip install scikit-learn")

    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def extract_features(data_dir: Path, cls_list: list):
        features, labels = [], []
        for cls in cls_list:
            cls_dir = data_dir / cls
            if not cls_dir.exists():
                continue
            imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
            print(f"  提取 {cls}（{len(imgs)} 張）特徵...")
            for img_path in imgs:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = model.encode_image(img_tensor)
                    features.append(feat.cpu().numpy().flatten())
                    labels.append(cls)
                except Exception as e:
                    print(f"    跳過損壞圖片：{img_path.name}（{e}）")
        return np.array(features), np.array(labels)

    X, y = extract_features(data_dir, CLASSES)
    if len(X) == 0:
        print("  [錯誤] 無法提取特徵，請確認資料集路徑")
        return

    # 分層切分
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_idx, val_idx = next(sss.split(X, y))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"\n  訓練：{len(X_train)}  驗證：{len(X_val)}")
    print("  訓練 Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED, n_jobs=-1)
    clf.fit(X_train, y_train)

    val_acc = clf.score(X_val, y_val)
    print(f"\n  驗證準確率：{val_acc:.4f}")
    y_pred = clf.predict(X_val)
    labels = list(range(len(CLASSES)))  # 固定 0..5 對應 6 類（含 worn）
    print(
        "\n" + classification_report(
            y_val,
            y_pred,
            labels=labels,
            target_names=CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = ckpt_dir / "clip_linear_probe.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n  Linear Probe 已儲存：{pkl_path}")


# ── 主訓練流程 ────────────────────────────────────────────────────────

def train(args):
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用裝置：{device}")
    if device.type == "cuda":
        print(f"  GPU：{torch.cuda.get_device_name(0)}")
        print(f"  VRAM：{torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    # ── Stage 1 only ──────────────────────────────────────────────
    if args.stage == "linear":
        run_clip_linear_probe(DATA_DIR, CKPT_DIR)
        return

    # ── 統計資料量 ────────────────────────────────────────────────
    print("\n資料集統計：")
    counts = count_class_samples(DATA_DIR)
    total = sum(counts.values())
    for cls, cnt in counts.items():
        if cnt == 0:
            status = "✗ 跳過"
        elif cnt < MIN_SAMPLES_CONVNEXT:
            status = "⚠ 不足"
        else:
            status = "✓"
        print(f"  {status} {cls:10s}: {cnt:5d} 張")
    print(f"  總計：{total} 張")

    if total == 0:
        raise RuntimeError(
            f"資料集目錄 {DATA_DIR} 為空，請先執行資料整理腳本：\n"
            "  python scripts/03_organize_sdnet.py\n"
            "  python scripts/07_download_roboflow.py --api-key YOUR_KEY"
        )

    # ── 決定實際訓練類別（跳過空類別）────────────────────────────
    active_classes = [cls for cls in ALL_CLASSES if counts.get(cls, 0) > 0]
    skipped = [cls for cls in ALL_CLASSES if counts.get(cls, 0) == 0]
    if skipped:
        print(f"\n  [注意] 以下類別無資料，本次訓練跳過：{skipped}")
        print(f"  [注意] 實際訓練類別（{len(active_classes)} 類）：{active_classes}")
    global CLASSES, N_CLASSES
    CLASSES   = active_classes
    N_CLASSES = len(active_classes)

    # ── 自動選擇架構 ──────────────────────────────────────────────
    arch = args.arch
    if arch == "auto":
        arch = auto_select_arch(counts, active_classes)
    print(f"\n使用架構：{arch}")

    # ── 建立資料集 ────────────────────────────────────────────────
    print("\n建立資料集...")
    train_samples, val_samples = build_datasets(DATA_DIR, val_split=0.15)

    train_dataset = DefectDataset(train_samples, transform=get_train_transforms())
    val_dataset   = DefectDataset(val_samples,   transform=get_val_transforms())

    sampler = build_weighted_sampler(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── 建立模型 ──────────────────────────────────────────────────
    print("\n建立模型...")
    model = build_model(arch, N_CLASSES, pretrained=True)
    model = model.to(device)

    # ── class weights ─────────────────────────────────────────────
    class_weights = compute_class_weights(counts, active_classes).to(device)
    print(f"\nclass_weights: {dict(zip(active_classes, class_weights.tolist()))}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    # ── FP16 Scaler ───────────────────────────────────────────────
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── TensorBoard ───────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"{arch}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(LOG_DIR / run_name)

    best_val_acc = 0.0
    start_epoch  = 1

    # ── 決定訓練階段 ──────────────────────────────────────────────
    # stage = "all"：Stage 1(Linear Probe) → Stage 2(frozen backbone) → Stage 3(full finetune)
    # stage = "finetune"：直接 full fine-tune（有 checkpoint 時常用）
    # stage = "swin"：同 finetune，但確保使用 swin_t

    stages = []
    if args.stage in ("all",):
        stages = [
            {"name": "warmup",   "epochs": EPOCHS_LINEAR,   "freeze": True,  "lr": BASE_LR},
            {"name": "finetune", "epochs": EPOCHS_FINETUNE, "freeze": False, "lr": FINETUNE_LR},
        ]
    else:
        # 單階段 full fine-tune
        stages = [
            {"name": "finetune", "epochs": EPOCHS_FINETUNE, "freeze": False, "lr": FINETUNE_LR},
        ]

    # ── 載入 checkpoint（如果有）──────────────────────────────────
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=FINETUNE_LR,
                weight_decay=WEIGHT_DECAY,
            )
            start_epoch, best_val_acc = load_checkpoint(resume_path, model, optimizer, scaler)
            # 直接進 finetune 模式
            stages = [{"name": "resumed", "epochs": EPOCHS_FINETUNE, "freeze": False, "lr": FINETUNE_LR}]
        else:
            print(f"  [警告] checkpoint 不存在：{args.resume}，從頭訓練")

    # ── 執行各訓練階段 ────────────────────────────────────────────
    global_epoch = start_epoch

    for stage in stages:
        stage_name   = stage["name"]
        stage_epochs = stage["epochs"]
        stage_lr     = stage["lr"]
        stage_freeze = stage["freeze"]

        print(f"\n{'=' * 60}")
        print(f"  階段：{stage_name}  epochs={stage_epochs}  lr={stage_lr}  freeze={stage_freeze}")
        print(f"{'=' * 60}")

        if stage_freeze:
            freeze_backbone(model, arch)
        else:
            unfreeze_backbone(model, freeze_patch_embed=True)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=stage_lr,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999),
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=WARMUP_EPOCHS,
            total_epochs=stage_epochs,
            min_lr=MIN_LR,
        )

        for epoch in range(1, stage_epochs + 1):
            print(f"\n[Epoch {global_epoch}] lr={scheduler.get_last_lr()[0]:.6f}")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, global_epoch
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            scheduler.step()

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc

            # TensorBoard 記錄
            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, global_epoch)
            writer.add_scalars("Acc",  {"train": train_acc,  "val": val_acc},  global_epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], global_epoch)

            print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                  f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                  f"  best={best_val_acc:.4f}")

            save_checkpoint(
                {
                    "epoch":        global_epoch,
                    "arch":         arch,
                    "model":        model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "scaler":       scaler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "classes":      active_classes,
                },
                is_best=is_best,
                ckpt_dir=CKPT_DIR,
            )
            global_epoch += 1

    writer.close()

    print(f"\n{'=' * 60}")
    print(f"  訓練完成！最佳驗證準確率：{best_val_acc:.4f}")
    print(f"  最佳 checkpoint：{CKPT_DIR / 'best.pt'}")
    print(f"  TensorBoard：tensorboard --logdir {LOG_DIR}")
    print(f"{'=' * 60}\n")

    # 儲存訓練配置（方便 evaluate.py 讀取）
    config = {
        "arch":         arch,
        "classes":      active_classes,
        "img_size":     IMG_SIZE,
        "best_val_acc": best_val_acc,
        "class_weights": compute_class_weights(counts, active_classes).tolist(),
    }
    config_path = CKPT_DIR / "train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  訓練配置已儲存：{config_path}")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Room Inspector Layer 3 訓練腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arch",
        default="auto",
        choices=["auto", "convnext_t", "swin_t", "convnext_s", "swin_s"],
        help="訓練架構（預設：auto 自動選擇）",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "linear", "finetune", "swin"],
        help="訓練階段（all=完整三階段, linear=只做 CLIP Probe, finetune=直接 fine-tune）",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CKPT",
        help="從 checkpoint 繼續訓練",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="覆蓋 fine-tune epochs（預設：30）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size（預設：{BATCH_SIZE}）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="學習率（預設：fine-tune 用 1e-4）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 覆蓋全域超參數
    if args.epochs is not None:
        EPOCHS_FINETUNE = args.epochs
    if args.batch_size != BATCH_SIZE:
        BATCH_SIZE = args.batch_size
    if args.lr is not None:
        FINETUNE_LR = args.lr
        BASE_LR = args.lr

    train(args)

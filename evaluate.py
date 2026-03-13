#!/usr/bin/env python3
"""
evaluate.py
===========
Room Inspector — Layer 3 評估腳本

功能：
  1. Confusion matrix（seaborn heatmap）
  2. 每類 Precision / Recall / F1 / Support
  3. Top-N 錯誤案例視覺化（最難分類的圖片）
  4. 每類別錯誤分析（最常被誤判的類別）

使用方式：
  python evaluate.py --checkpoint checkpoints/best.pt
  python evaluate.py --checkpoint checkpoints/best.pt --output-dir outputs/eval
  python evaluate.py --checkpoint checkpoints/best.pt --top-errors 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms

# ── 設定 ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data" / "defects"
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "eval"

CLASSES = ["normal", "crack", "stain", "mold", "peeling"]
IMG_SIZE = 224

# 視覺化色彩
PALETTE = [
    "#4CAF50",  # normal  — 綠
    "#F44336",  # crack   — 紅
    "#FF9800",  # stain   — 橘
    "#9C27B0",  # mold    — 紫
    "#2196F3",  # peeling — 藍
]


# ── 資料前處理 ────────────────────────────────────────────────────────

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── 資料集 ────────────────────────────────────────────────────────────

class DefectDataset(torch.utils.data.Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, data_dir: Path, classes: list, transform=None):
        self.samples  = []
        self.transform = transform
        for cls_idx, cls in enumerate(classes):
            cls_dir = data_dir / cls
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((img_path, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(img_path)


# ── 模型載入 ──────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device):
    """從 checkpoint 載入模型。"""
    print(f"載入 checkpoint：{ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    arch    = state.get("arch", "convnext_t")
    classes = state.get("classes", CLASSES)
    n_cls   = len(classes)

    try:
        import timm
    except ImportError:
        raise ImportError("請安裝 timm：pip install timm")

    arch_map = {
        "convnext_t": "convnext_tiny.fb_in22k_ft_in1k",
        "swin_t":     "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "convnext_s": "convnext_small.fb_in22k_ft_in1k",
        "swin_s":     "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    }
    model_name = arch_map.get(arch, arch)

    model = timm.create_model(model_name, pretrained=False, num_classes=n_cls)
    model.load_state_dict(state["model"])
    model = model.to(device)
    model.eval()

    best_val_acc = state.get("best_val_acc", 0.0)
    print(f"  架構：{arch}  類別：{classes}  訓練最佳 acc={best_val_acc:.4f}")
    return model, arch, classes


# ── 推論 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """
    對整個資料集推論，回傳：
    - all_preds: 預測類別 (N,)
    - all_labels: 真實類別 (N,)
    - all_probs: softmax 機率 (N, C)
    - all_paths: 圖片路徑列表
    """
    all_preds, all_labels, all_probs, all_paths = [], [], [], []

    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        with autocast():
            logits = model(images)
        probs = torch.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_paths.extend(paths)

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        all_paths,
    )


# ── 混淆矩陣 ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, classes: list, output_path: Path):
    """繪製標準化混淆矩陣 heatmap。"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 正規化（行 = 真實類別）
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 左：原始數量
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, linecolor="gray",
        ax=axes[0],
    )
    axes[0].set_title("混淆矩陣（原始數量）", fontsize=13, pad=10)
    axes[0].set_xlabel("預測類別", fontsize=11)
    axes[0].set_ylabel("真實類別", fontsize=11)

    # 右：正規化（百分比）
    annot = np.array([[f"{v:.2f}" for v in row] for row in cm_norm])
    sns.heatmap(
        cm_norm, annot=annot, fmt="s", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, linecolor="gray",
        vmin=0.0, vmax=1.0,
        ax=axes[1],
    )
    axes[1].set_title("混淆矩陣（正規化）", fontsize=13, pad=10)
    axes[1].set_xlabel("預測類別", fontsize=11)
    axes[1].set_ylabel("真實類別", fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  混淆矩陣已儲存：{output_path}")


# ── 分類報告 ──────────────────────────────────────────────────────────

def save_classification_report(
    preds: np.ndarray,
    labels: np.ndarray,
    classes: list,
    output_path: Path,
):
    from sklearn.metrics import classification_report, accuracy_score

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=classes, digits=4)

    lines = [
        "=" * 60,
        "  Room Inspector — 分類評估報告",
        "=" * 60,
        f"\n  Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)\n",
        report,
    ]

    # 每類別詳細分析（誤判分佈）
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    lines.append("\n" + "─" * 60)
    lines.append("  各類別誤判分析：")
    for i, cls in enumerate(classes):
        row = cm[i]
        total = row.sum()
        correct = row[i]
        errors  = [(classes[j], row[j]) for j in range(len(classes)) if j != i and row[j] > 0]
        errors.sort(key=lambda x: -x[1])
        err_str = "  ".join(f"{c}:{n}" for c, n in errors[:3])
        lines.append(
            f"  {cls:10s}: {correct}/{total} 正確  "
            f"主要誤判→ {err_str if err_str else '無'}"
        )

    report_text = "\n".join(lines)
    print("\n" + report_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  報告已儲存：{output_path}")

    return cm


# ── 錯誤案例視覺化 ────────────────────────────────────────────────────

def plot_error_cases(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    paths: list,
    classes: list,
    output_path: Path,
    n_errors: int = 24,
):
    """
    視覺化最難分類的錯誤案例。
    難度排序：以錯誤預測的信心分數（越高表示模型越「確信」但卻錯了）。
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image

    # 找出所有錯誤樣本
    error_mask = preds != labels
    error_indices = np.where(error_mask)[0]

    if len(error_indices) == 0:
        print("  無錯誤案例（完美分類！）")
        return

    # 按錯誤預測的信心排序（信心越高越難）
    error_confs = probs[error_indices, preds[error_indices]]
    sorted_order = np.argsort(-error_confs)
    top_indices  = error_indices[sorted_order[:n_errors]]

    # 繪圖
    n_cols = 6
    n_rows = (len(top_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.5))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for plot_idx, sample_idx in enumerate(top_indices):
        ax = axes[plot_idx]
        img_path = paths[sample_idx]
        true_cls  = classes[labels[sample_idx]]
        pred_cls  = classes[preds[sample_idx]]
        conf      = probs[sample_idx, preds[sample_idx]]

        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "載入失敗", ha="center", va="center", transform=ax.transAxes)

        # 標題：真實 / 預測（綠=正確，紅=錯誤）
        ax.set_title(
            f"真：{true_cls}\n預：{pred_cls} ({conf:.2f})",
            fontsize=8,
            color="red",
            pad=3,
        )
        ax.axis("off")

        # 邊框顏色對應預測類別
        pred_idx = classes.index(pred_cls) if pred_cls in classes else 0
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE[pred_idx])
            spine.set_linewidth(3)
            spine.set_visible(True)

    # 關閉多餘的子圖
    for ax in axes[len(top_indices):]:
        ax.axis("off")

    # 圖例
    patches = [
        mpatches.Patch(color=PALETTE[i], label=cls)
        for i, cls in enumerate(classes)
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=len(classes),
        fontsize=9,
        title="邊框 = 預測類別",
        bbox_to_anchor=(0.5, 0.0),
    )

    total_errors = len(error_indices)
    acc = 1.0 - total_errors / len(preds)
    fig.suptitle(
        f"Top-{len(top_indices)} 錯誤案例（共 {total_errors} 個錯誤 / Acc={acc:.3f}）",
        fontsize=12,
        y=1.01,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  錯誤案例圖已儲存：{output_path}")


# ── 每類別 F1 bar chart ───────────────────────────────────────────────

def plot_per_class_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    classes: list,
    output_path: Path,
):
    """繪製每類別 Precision / Recall / F1 的長條圖。"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(classes))), zero_division=0
    )

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 1.5), 5))
    ax.bar(x - width, prec, width, label="Precision", color="#2196F3", alpha=0.85)
    ax.bar(x,         rec,  width, label="Recall",    color="#4CAF50", alpha=0.85)
    ax.bar(x + width, f1,   width, label="F1-score",  color="#FF9800", alpha=0.85)

    # 數字標籤
    for i in range(len(classes)):
        ax.text(x[i] - width, prec[i] + 0.01, f"{prec[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i],         rec[i]  + 0.01, f"{rec[i]:.2f}",  ha="center", fontsize=8)
        ax.text(x[i] + width, f1[i]   + 0.01, f"{f1[i]:.2f}",   ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{cls}\n(n={support[i]})" for i, cls in enumerate(classes)], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("分數", fontsize=11)
    ax.set_title("各類別評估指標", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  類別指標圖已儲存：{output_path}")


# ── 主評估流程 ────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 讀取訓練配置（如果有）────────────────────────────────────
    config_path = CKPT_DIR / "train_config.json"
    classes = CLASSES
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        classes = config.get("classes", CLASSES)
        print(f"  從 train_config.json 讀取類別：{classes}")

    # ── 載入模型 ──────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}")

    model, arch, classes = load_model(ckpt_path, device)

    # ── 決定評估資料來源 ──────────────────────────────────────────
    if args.holdout_dir:
        eval_data_dir = Path(args.holdout_dir)
        print(f"\n載入獨立測試集：{eval_data_dir}")
        print("  [嚴格評估模式] 使用未參與訓練的 holdout set")
    else:
        eval_data_dir = DATA_DIR
        print(f"\n載入資料集：{eval_data_dir}")
        print("  [注意] 使用全量 data/defects/，包含訓練資料（in-sample 評估）")
        print("  若需嚴格評估，請用 --holdout-dir 指定獨立測試集目錄")

    dataset = DefectDataset(eval_data_dir, classes, transform=get_val_transforms())

    if len(dataset) == 0:
        raise RuntimeError(f"資料集為空：{eval_data_dir}")

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"  共 {len(dataset)} 張圖片")

    # ── 推論 ──────────────────────────────────────────────────────
    print("\n推論中...")
    preds, labels, probs, paths = run_inference(model, loader, device)
    print(f"  完成！共 {len(preds)} 個樣本")

    # ── 混淆矩陣 ──────────────────────────────────────────────────
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))

    plot_confusion_matrix(cm, classes, output_dir / "confusion_matrix.png")

    # ── 分類報告 ──────────────────────────────────────────────────
    save_classification_report(preds, labels, classes, output_dir / "classification_report.txt")

    # ── 每類別指標圖 ──────────────────────────────────────────────
    plot_per_class_metrics(preds, labels, classes, output_dir / "per_class_metrics.png")

    # ── 錯誤案例視覺化 ────────────────────────────────────────────
    plot_error_cases(
        preds, labels, probs, paths, classes,
        output_dir / "error_cases.png",
        n_errors=args.top_errors,
    )

    # ── 儲存預測結果 JSON ─────────────────────────────────────────
    results = {
        "checkpoint":    str(ckpt_path),
        "arch":          arch,
        "classes":       classes,
        "eval_mode":     "holdout" if args.holdout_dir else "in_sample",
        "eval_data_dir": str(eval_data_dir),
        "n_samples":     int(len(preds)),
        "n_errors":      int((preds != labels).sum()),
        "accuracy":      float((preds == labels).mean()),
        "per_class": {},
    }
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(classes))), zero_division=0
    )
    for i, cls in enumerate(classes):
        results["per_class"][cls] = {
            "precision": float(prec[i]),
            "recall":    float(rec[i]),
            "f1":        float(f1[i]),
            "support":   int(sup[i]),
        }

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  完整結果已儲存：{results_path}")

    print(f"\n{'=' * 55}")
    print(f"  評估完成！")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  輸出目錄：{output_dir}")
    print(f"{'=' * 55}\n")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Room Inspector Layer 3 評估腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=str(CKPT_DIR / "best.pt"),
        help="要評估的 checkpoint 路徑（預設：checkpoints/best.pt）",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"評估結果輸出目錄（預設：{OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=24,
        help="顯示前 N 個最難分類的錯誤案例（預設：24）",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help=f"資料集目錄（預設：{DATA_DIR}）",
    )
    parser.add_argument(
        "--holdout-dir",
        default=None,
        metavar="DIR",
        help="獨立測試集目錄（格式同 data/defects/，不含訓練資料）。"
             "未指定時使用全量 data/defects/（in-sample 評估）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.data_dir != str(DATA_DIR):
        DATA_DIR = Path(args.data_dir)
    evaluate(args)

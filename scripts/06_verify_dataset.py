#!/usr/bin/env python3
"""
06_verify_dataset.py
====================
驗證 data/defects/ 資料集的完整性，產生統計報告與訓練建議。

功能：
  1. 各類別圖片數量統計
  2. 類別不平衡程度評估
  3. 計算訓練用 class_weights（用於加權損失函數）
  4. 抽樣展示各類別代表圖片（需要 matplotlib）
  5. 檢查圖片尺寸是否符合要求（224×224）
  6. 偵測損壞圖片

執行方式：
  python scripts/06_verify_dataset.py
  python scripts/06_verify_dataset.py --show-samples  # 額外顯示示例圖片
  python scripts/06_verify_dataset.py --fix-size      # 自動修復非 224×224 的圖片
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

# ── 路徑設定 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_ROOT  = PROJECT_ROOT / "data" / "defects"

CLASSES       = ["normal", "crack", "stain", "mold", "peeling"]
PATCH_SIZE    = 224
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp"}
RANDOM_SEED   = 42
MIN_PER_CLASS = 200   # 建議最低數量


def check_and_fix_image(img_path: Path, fix_size: bool = False) -> dict:
    """
    檢查單張圖片，回傳狀態資訊。
    """
    result = {
        "path":    str(img_path),
        "ok":      True,
        "size":    None,
        "mode":    None,
        "issue":   None,
    }
    try:
        img = Image.open(img_path)
        img.verify()  # 檢查檔案完整性
        img = Image.open(img_path).convert("RGB")  # 重新開啟讀取資訊

        result["size"] = img.size
        result["mode"] = img.mode

        w, h = img.size
        if w != PATCH_SIZE or h != PATCH_SIZE:
            if fix_size:
                # 自動 resize + center crop
                scale = PATCH_SIZE / min(w, h)
                new_w, new_h = max(int(w * scale), PATCH_SIZE), max(int(h * scale), PATCH_SIZE)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                left = (new_w - PATCH_SIZE) // 2
                top  = (new_h - PATCH_SIZE) // 2
                img  = img.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))
                img.save(img_path, "JPEG", quality=95)
                result["size"]  = (PATCH_SIZE, PATCH_SIZE)
                result["issue"] = f"已修復：{w}×{h} → {PATCH_SIZE}×{PATCH_SIZE}"
            else:
                result["issue"] = f"尺寸不符：{w}×{h}（應為 {PATCH_SIZE}×{PATCH_SIZE}）"

    except (UnidentifiedImageError, Exception) as e:
        result["ok"]    = False
        result["issue"] = f"損壞或無法開啟：{e}"

    return result


def compute_class_weights(counts: dict) -> dict:
    """
    計算 balanced class weights（用於 CrossEntropyLoss 的 weight 參數）。
    公式：weight[i] = total / (n_classes * count[i])
    """
    total    = sum(counts.values())
    n_cls    = len(counts)
    weights  = {}
    for cls in CLASSES:
        cnt = counts.get(cls, 1)
        weights[cls] = total / (n_cls * cnt) if cnt > 0 else 0.0

    # 正規化，使最小 weight = 1.0
    min_w = min(weights.values())
    if min_w > 0:
        weights = {k: v / min_w for k, v in weights.items()}

    return weights


def verify_dataset(show_samples: bool = False, fix_size: bool = False):
    print("=" * 65)
    print("  資料集驗證報告")
    print("=" * 65)

    if not TARGET_ROOT.exists():
        print(f"\n[錯誤] 資料集目錄不存在：{TARGET_ROOT}")
        print("  請先執行資料整理腳本：")
        print("    python scripts/02_organize_bd3.py")
        print("    python scripts/03_organize_sdnet.py")
        sys.exit(1)

    random.seed(RANDOM_SEED)

    counts       = {}
    size_issues  = {}
    corrupt_imgs = {}
    source_dist  = {}   # 來源統計（bd3/sdnet/dagm/user）

    # ── 逐類掃描 ──────────────────────────────────────────────
    print("\n  正在掃描圖片...")
    for cls in CLASSES:
        cls_dir = TARGET_ROOT / cls
        if not cls_dir.exists():
            counts[cls]       = 0
            size_issues[cls]  = []
            corrupt_imgs[cls] = []
            continue

        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
        counts[cls]       = len(images)
        size_issues[cls]  = []
        corrupt_imgs[cls] = []
        src_cnt           = Counter()

        # 抽樣檢查（全部 <500 張時全查，否則隨機抽 200 張）
        sample = images if len(images) <= 500 else random.sample(images, 200)
        for img_path in sample:
            result = check_and_fix_image(img_path, fix_size=fix_size)
            if not result["ok"]:
                corrupt_imgs[cls].append(img_path.name)
            elif result["issue"] and not fix_size:
                size_issues[cls].append(img_path.name)

            # 來源統計
            name = img_path.name
            if name.startswith("bd3_"):    src_cnt["BD3"]     += 1
            elif name.startswith("sdnet_"): src_cnt["SDNET"]   += 1
            elif name.startswith("dagm_"):  src_cnt["DAGM"]    += 1
            elif name.startswith("user_"):  src_cnt["自拍"]    += 1
            else:                           src_cnt["其他"]    += 1

        source_dist[cls] = dict(src_cnt)

    # ── 數量統計表 ────────────────────────────────────────────
    total = sum(counts.values())
    print(f"\n  {'類別':10s} {'數量':>7s}  {'狀態':30s}  {'來源'}")
    print("  " + "─" * 70)

    for cls in CLASSES:
        cnt       = counts[cls]
        need_more = max(0, MIN_PER_CLASS - cnt)
        if cnt == 0:
            status = "✗ 無資料"
        elif cnt < MIN_PER_CLASS:
            status = f"⚠ 還需 {need_more} 張"
        else:
            status = "✓ 達標"

        bar_len = min(cnt // 25, 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        pct     = cnt / total * 100 if total > 0 else 0

        src_str = "  ".join(f"{k}:{v}" for k, v in source_dist.get(cls, {}).items())
        print(f"  {cls:10s} {cnt:>5d} 張  {bar}  {pct:5.1f}%  {status}")
        if src_str:
            print(f"  {'':10s}           {'來源抽樣：' + src_str}")

    print(f"\n  總計：{total} 張")

    # ── 類別不平衡評估 ────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  類別不平衡評估")

    max_cnt  = max(counts.values()) if counts else 1
    min_cnt  = min(v for v in counts.values() if v > 0) if any(counts.values()) else 1
    imbalance_ratio = max_cnt / min_cnt if min_cnt > 0 else float("inf")

    if imbalance_ratio <= 2.0:
        balance_status = "✓ 良好（比值 ≤ 2x）"
    elif imbalance_ratio <= 5.0:
        balance_status = "⚠ 輕度不平衡（比值 2~5x），建議開啟加權損失"
    else:
        balance_status = "✗ 嚴重不平衡（比值 >5x），務必開啟加權損失"

    print(f"\n  最多類：{max(counts, key=counts.get)} ({max_cnt} 張)")
    print(f"  最少類：{min((k for k in counts if counts[k]>0), key=counts.get, default='N/A')} ({min_cnt} 張)")
    print(f"  不平衡比值：{imbalance_ratio:.1f}x  →  {balance_status}")

    # ── Class Weights ─────────────────────────────────────────
    print("\n  訓練用 class_weights（貼入訓練腳本）：")
    weights = compute_class_weights({cls: counts[cls] for cls in CLASSES})

    print("\n  ```python")
    print(f"  CLASS_WEIGHTS = {json.dumps(weights, indent=4, ensure_ascii=False)}")
    print()
    print("  # 使用方式：")
    print("  import torch")
    print("  CLASSES = " + str(CLASSES))
    print("  weight_tensor = torch.tensor(")
    print("      [CLASS_WEIGHTS[c] for c in CLASSES], dtype=torch.float32")
    print("  ).to(DEVICE)")
    print("  criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)")
    print("  ```")

    # ── 損壞圖片警告 ──────────────────────────────────────────
    all_corrupt = [(cls, f) for cls, files in corrupt_imgs.items() for f in files]
    if all_corrupt:
        print("\n" + "─" * 65)
        print(f"  ⚠ 偵測到 {len(all_corrupt)} 張損壞圖片：")
        for cls, fname in all_corrupt[:20]:
            print(f"    [{cls}] {fname}")
        if len(all_corrupt) > 20:
            print(f"    ...（共 {len(all_corrupt)} 張，只顯示前 20）")
        print("\n  建議手動刪除這些損壞圖片。")

    # ── 尺寸問題警告 ──────────────────────────────────────────
    all_size_issues = [(cls, f) for cls, files in size_issues.items() for f in files]
    if all_size_issues and not fix_size:
        print(f"\n  ⚠ 偵測到 {len(all_size_issues)} 張尺寸不符的圖片")
        print("    執行以下指令自動修復：")
        print("    python scripts/06_verify_dataset.py --fix-size")

    # ── 整體建議 ──────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  整體建議")

    lacking = [cls for cls in CLASSES if counts[cls] < MIN_PER_CLASS]
    empty   = [cls for cls in CLASSES if counts[cls] == 0]

    if empty:
        print(f"\n  ✗ 完全缺乏資料的類別：{empty}")
        print("    → 請優先收集這些類別的照片，放入 user_photos/ 對應資料夾")
        print("    → 執行：python scripts/05_extract_patches.py")

    if lacking and not empty:
        print(f"\n  ⚠ 資料量不足的類別（< {MIN_PER_CLASS} 張）：{lacking}")
        print("    → 建議補充後再開始訓練，或使用 CLIP Linear Probe 作為過渡方案")

    if total >= 1200 and not lacking:
        print(f"\n  ✓ 資料集已達訓練標準（總計 {total} 張，各類均 ≥ {MIN_PER_CLASS}）")
        print("    → 可以開始執行訓練：python train.py")
    elif total >= 600:
        print(f"\n  △ 資料量中等（{total} 張）")
        print("    → 建議先試跑訓練，同時繼續收集資料")
        print("    → 注意啟用強資料增強和較小的 learning rate")
    else:
        print(f"\n  ✗ 資料量不足（{total} 張，建議 ≥ 1200 張）")
        print("    → 建議先使用 CLIP Linear Probe（每類只需 30~50 張）")

    print()

    if show_samples:
        _show_samples()


def _show_samples(n_per_class: int = 4):
    """顯示各類別代表圖片（需要 matplotlib）。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  [注意] 顯示示例圖片需要 matplotlib：pip install matplotlib")
        return

    fig = plt.figure(figsize=(4 * n_per_class, 3 * len(CLASSES)))
    gs  = gridspec.GridSpec(len(CLASSES), n_per_class, figure=fig)
    fig.suptitle("各類別示例圖片", fontsize=14)

    for row, cls in enumerate(CLASSES):
        cls_dir = TARGET_ROOT / cls
        if not cls_dir.exists():
            continue
        images  = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
        samples = random.sample(images, min(n_per_class, len(images)))

        for col, img_path in enumerate(samples):
            ax = fig.add_subplot(gs[row, col])
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(np.array(img))
            except Exception:
                ax.text(0.5, 0.5, "無法顯示", ha="center", va="center")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=11, rotation=0, labelpad=60, va="center")

    plt.tight_layout()
    output_path = PROJECT_ROOT / "data" / "dataset_samples.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"\n  示例圖片已儲存：{output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="驗證資料集完整性")
    parser.add_argument(
        "--show-samples", action="store_true",
        help="顯示各類別示例圖片（需要 matplotlib）"
    )
    parser.add_argument(
        "--fix-size", action="store_true",
        help="自動修復尺寸不符的圖片（resize + center crop 到 224×224）"
    )
    args = parser.parse_args()
    verify_dataset(show_samples=args.show_samples, fix_size=args.fix_size)

#!/usr/bin/env python3
"""
04_organize_dagm.py
===================
整理 DAGM 2007 工業表面缺陷資料集，轉換並對應至本專案類別。

DAGM 結構：
  Class1/ ~ Class10/
    Train/
      0/   ← 無缺陷 (non-defective)
      1/   ← 有缺陷 (defective)
      Label/  ← 缺陷位置 mask（PNG，可選）
    Test/   （結構同 Train）

DAGM 圖片為灰階 PNG（512×512），需轉換為 RGB。

類別對應策略：
  - 無缺陷 (0/) → normal
  - 有缺陷 (1/) → 依 CLASS_MAPPING 決定 (crack 或 stain)

DAGM 各 Class 的缺陷視覺特徵（依論文描述）：
  Class1~3  : 線條/刮痕型缺陷 → crack
  Class4~6  : 斑點型缺陷      → stain
  Class7~10 : 競賽集（混合型）→ crack（保守起見）

  ⚠ DAGM 是工業場景的灰階影像，與室內彩色影像差距較大，
    建議作為輔助資料，不超過同類總量的 30%。

執行方式：
  python scripts/04_organize_dagm.py [--classes 1,2,3] [--max-per-class N]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── 路徑設定 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAGM_RAW     = PROJECT_ROOT / "data" / "raw" / "dagm"
TARGET_ROOT  = PROJECT_ROOT / "data" / "defects"

# ── DAGM Class → 目標類別對應 ─────────────────────────────────
# (Class 編號, 缺陷對應類別, 說明)
CLASS_MAPPING = {
    1:  ("crack",  "線條/刮痕型"),
    2:  ("crack",  "線條/刮痕型"),
    3:  ("crack",  "線條/刮痕型"),
    4:  ("stain",  "斑點型"),
    5:  ("stain",  "斑點型"),
    6:  ("stain",  "斑點型"),
    7:  ("crack",  "混合型（競賽集）"),
    8:  ("crack",  "混合型（競賽集）"),
    9:  ("stain",  "混合型（競賽集）"),
    10: ("stain",  "混合型（競賽集）"),
}

PATCH_SIZE   = 224
RANDOM_SEED  = 42


def grayscale_to_rgb_pil(img: Image.Image) -> Image.Image:
    """灰階圖轉 RGB（stack 三通道，保留紋理資訊）。"""
    if img.mode != "L":
        img = img.convert("L")
    return img.convert("RGB")


def resize_center_crop(img: Image.Image, size: int = PATCH_SIZE) -> Image.Image:
    """Resize 後 center crop 到目標大小。"""
    # 先等比例縮放到 256，再 crop 224
    w, h = img.size
    scale = 256 / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def has_defect_region(label_path: Path, min_defect_ratio: float = 0.005) -> bool:
    """
    透過 Label mask 確認圖片確實含有缺陷區域。
    min_defect_ratio：缺陷像素比例下限（預設 0.5%）。
    """
    if not label_path.exists():
        return True  # 沒有 mask 就直接接受

    try:
        mask = np.array(Image.open(label_path).convert("L"))
        defect_ratio = (mask > 128).sum() / mask.size
        return defect_ratio >= min_defect_ratio
    except Exception:
        return True


def organize_dagm(use_classes: list[int] = None, max_per_class: int = 300):
    print("=" * 60)
    print("  DAGM 2007 整理工具")
    print("=" * 60)

    # ── 確認原始資料 ───────────────────────────────────────────
    if not DAGM_RAW.exists() or not any(DAGM_RAW.iterdir()):
        print(f"\n[錯誤] DAGM 資料夾不存在或為空：{DAGM_RAW}")
        print("\n  請依下列步驟下載 DAGM 2007：")
        print("  1. 前往 https://zenodo.org/records/8086136")
        print("  2. 下載 Class1.zip ~ Class10.zip")
        print("  3. 解壓所有 zip 到 data/raw/dagm/")
        print("     解壓後應有：dagm/Class1/Train/0/ 等結構")
        print("\n  或使用 zenodo_get（pip install zenodo-get）：")
        print(f"    cd {DAGM_RAW}")
        print("    zenodo_get 8086136")
        sys.exit(1)

    if use_classes is None:
        use_classes = list(CLASS_MAPPING.keys())

    print(f"\n  DAGM 根目錄：{DAGM_RAW}")
    print(f"  使用 Class：{use_classes}")
    print(f"  每類上限  ：{max_per_class} 張")

    random.seed(RANDOM_SEED)

    # ── 確認目標資料夾 ─────────────────────────────────────────
    for cls in ["normal", "crack", "stain"]:
        (TARGET_ROOT / cls).mkdir(parents=True, exist_ok=True)

    # ── 處理每個 Class ─────────────────────────────────────────
    print("\n  Class 處理：")
    total_copied  = 0
    total_skipped = 0

    for cls_num in use_classes:
        if cls_num not in CLASS_MAPPING:
            print(f"  [略過] Class{cls_num}  — 不在對應表中")
            continue

        defect_target, description = CLASS_MAPPING[cls_num]
        cls_dir = DAGM_RAW / f"Class{cls_num}"

        # 嘗試不同目錄大小寫
        if not cls_dir.exists():
            cls_dir = DAGM_RAW / f"class{cls_num}"
        if not cls_dir.exists():
            print(f"  [略過] Class{cls_num}  — 資料夾不存在")
            continue

        for split in ["Train", "Test"]:
            split_dir = cls_dir / split
            if not split_dir.exists():
                split_dir = cls_dir / split.lower()
            if not split_dir.exists():
                continue

            # ── 處理無缺陷 (normal) ──────────────────────────
            no_defect_dir = split_dir / "0"
            if no_defect_dir.exists():
                images = sorted(no_defect_dir.glob("*.png"))
                if len(images) > max_per_class // 2:
                    images = random.sample(images, max_per_class // 2)

                dst_dir = TARGET_ROOT / "normal"
                for img_path in images:
                    dst_name = f"dagm_c{cls_num}_{split}_nd_{img_path.name.replace('.png', '.jpg')}"
                    dst_file = dst_dir / dst_name
                    if dst_file.exists():
                        total_skipped += 1
                        continue
                    try:
                        img = grayscale_to_rgb_pil(Image.open(img_path))
                        img = resize_center_crop(img)
                        img.save(dst_file, "JPEG", quality=95)
                        total_copied += 1
                    except Exception as e:
                        print(f"  [警告] {img_path.name}: {e}")

            # ── 處理有缺陷 ────────────────────────────────────
            defect_dir   = split_dir / "1"
            label_dir    = split_dir / "Label"
            if defect_dir.exists():
                images = sorted(defect_dir.glob("*.png"))
                if len(images) > max_per_class:
                    images = random.sample(images, max_per_class)

                dst_dir = TARGET_ROOT / defect_target
                copied_cls = 0

                for img_path in images:
                    # 確認 mask 顯示確實有缺陷
                    label_file = label_dir / img_path.name if label_dir.exists() else Path("")
                    if not has_defect_region(label_file):
                        continue

                    dst_name = (f"dagm_c{cls_num}_{split}_def_"
                                f"{img_path.name.replace('.png', '.jpg')}")
                    dst_file = dst_dir / dst_name
                    if dst_file.exists():
                        total_skipped += 1
                        continue

                    try:
                        img = grayscale_to_rgb_pil(Image.open(img_path))
                        img = resize_center_crop(img)
                        img.save(dst_file, "JPEG", quality=95)
                        total_copied += 1
                        copied_cls   += 1
                    except Exception as e:
                        print(f"  [警告] {img_path.name}: {e}")

                print(f"  [完成] Class{cls_num} {split:5s} ({description:12s}) "
                      f"→ {defect_target:10s}  缺陷 {copied_cls} 張")

    # ── 最終統計 ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  總計複製：{total_copied} 張  略過（已存在）：{total_skipped} 張")
    print("\n  ⚠ 注意：DAGM 為灰階工業影像，建議作為輔助資料")
    print("    每類 DAGM 圖片不應超過同類總量的 30%")
    print("\n  目前各類別數量：")
    for cls in ["normal", "crack", "stain", "mold", "peeling", "worn"]:
        count = len([f for f in (TARGET_ROOT / cls).iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        need_more = max(0, 200 - count)
        status = "✓ 達標" if count >= 200 else f"⚠ 還需 {need_more} 張"
        print(f"    {cls:10s}: {count:5d} 張  {status}")

    print("\n  提醒：mold/worn/peeling 類別無對應公開資料集")
    print("    → 請使用 scripts/05_extract_patches.py 從台灣房屋照片補充")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="整理 DAGM 2007 資料集")
    parser.add_argument(
        "--classes", type=str, default=None,
        help="指定要使用的 Class 編號，逗號分隔，例如 1,2,3（預設全部）"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=300,
        help="每個 Class 有缺陷圖片最多取幾張（預設 300）"
    )
    args = parser.parse_args()

    classes = None
    if args.classes:
        try:
            classes = [int(c.strip()) for c in args.classes.split(",")]
        except ValueError:
            print("[錯誤] --classes 格式錯誤，請用逗號分隔整數，例如：1,2,3")
            sys.exit(1)

    organize_dagm(use_classes=classes, max_per_class=args.max_per_class)

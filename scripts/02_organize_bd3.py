#!/usr/bin/env python3
"""
02_organize_bd3.py
==================
將 BD3 Dataset 整理並對應到本專案的 6 類缺陷結構。

BD3 原始類別 → 本專案類別對應：
  normal/      → normal   （正常無缺陷）
  major crack/ → crack    （主要裂縫）
  minor crack/ → crack    （細微裂縫）
  stain/       → stain    （污漬）
  peeling/     → peeling  （剝落）
  spalling/    → peeling  （混凝土剝落，性質同剝落）
  Algae/       → mold     （藻類生長，外觀近似霉斑）

執行方式：
  python scripts/02_organize_bd3.py

注意：
  - 預設使用複製（不刪除原始資料）
  - 重複執行安全：已存在的檔案會跳過
"""

import shutil
import sys
from pathlib import Path

# ── 路徑設定 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BD3_RAW      = PROJECT_ROOT / "data" / "raw" / "bd3"
TARGET_ROOT  = PROJECT_ROOT / "data" / "defects"

# ── BD3 類別 → 本專案類別對應表 ───────────────────────────────
# 格式：("BD3 資料夾名稱", "目標類別", 是否啟用)
MAPPING = [
    ("normal",       "normal",  True),
    ("major crack",  "crack",   True),
    ("minor crack",  "crack",   True),
    ("stain",        "stain",   True),
    ("peeling",      "peeling", True),
    ("spalling",     "peeling", True),   # spalling = 表面剝落，歸入 peeling
    ("Algae",        "mold",    True),   # 藻類/青苔，外觀近似霉斑
]

# BD3 圖片所在的相對路徑（在 git repo 中）
BD3_CLASSES_PATH = BD3_RAW / "sample images" / "class_images"

# ── 支援的副檔名 ──────────────────────────────────────────────
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_bd3_classes_path(bd3_root: Path) -> Path:
    """
    BD3 repo 結構可能因版本不同而異，嘗試找到正確的類別資料夾。
    """
    candidates = [
        bd3_root / "sample images" / "class_images",
        bd3_root / "sample_images" / "class_images",
        bd3_root / "dataset",
        bd3_root / "data",
        bd3_root,  # 直接在根目錄
    ]
    for candidate in candidates:
        if candidate.exists():
            # 確認是否有 normal 或 crack 子資料夾
            has_classes = any(
                (candidate / cls).exists()
                for cls in ["normal", "major crack", "Algae", "stain"]
            )
            if has_classes:
                return candidate

    # 嘗試遞迴搜尋 normal/ 資料夾
    for p in bd3_root.rglob("normal"):
        if p.is_dir():
            return p.parent

    return None


def organize_bd3():
    print("=" * 60)
    print("  BD3 資料集整理工具")
    print("=" * 60)

    # ── 確認 BD3 原始資料存在 ──────────────────────────────────
    if not BD3_RAW.exists():
        print(f"\n[錯誤] BD3 資料夾不存在：{BD3_RAW}")
        print("  請先執行：bash scripts/01_download_datasets.sh")
        sys.exit(1)

    classes_path = find_bd3_classes_path(BD3_RAW)
    if classes_path is None:
        print(f"\n[錯誤] 無法在 BD3 目錄中找到類別資料夾")
        print(f"  BD3 根目錄：{BD3_RAW}")
        print("  請確認 BD3 git clone 是否完整，並手動確認資料夾結構。")
        sys.exit(1)

    print(f"\n  BD3 類別路徑：{classes_path}")
    print(f"  輸出目標：{TARGET_ROOT}")

    # ── 確認目標資料夾存在 ─────────────────────────────────────
    for cls in ["normal", "crack", "stain", "mold", "peeling", "worn"]:
        (TARGET_ROOT / cls).mkdir(parents=True, exist_ok=True)

    # ── 執行複製 ───────────────────────────────────────────────
    print("\n  類別對應與複製：")
    total_copied  = 0
    total_skipped = 0

    for bd3_cls, target_cls, enabled in MAPPING:
        if not enabled:
            continue

        src_dir = classes_path / bd3_cls
        dst_dir = TARGET_ROOT / target_cls

        if not src_dir.exists():
            print(f"  [略過] {bd3_cls:20s} → {target_cls:10s}  （來源不存在）")
            continue

        images = [f for f in src_dir.iterdir()
                  if f.suffix.lower() in IMG_EXTENSIONS]
        copied  = 0
        skipped = 0

        for img in images:
            # 加上來源前綴避免跨類別檔名衝突
            dst_name = f"bd3_{bd3_cls.replace(' ', '_')}_{img.name}"
            dst_file = dst_dir / dst_name
            if dst_file.exists():
                skipped += 1
                continue
            shutil.copy2(img, dst_file)
            copied += 1

        total_copied  += copied
        total_skipped += skipped
        print(f"  [完成] {bd3_cls:20s} → {target_cls:10s}  "
              f"複製 {copied:4d} 張  略過 {skipped:4d} 張")

    # ── 最終統計 ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  總計複製：{total_copied} 張  略過（已存在）：{total_skipped} 張")
    print("\n  目前各類別數量：")
    for cls in ["normal", "crack", "stain", "mold", "peeling", "worn"]:
        count = len([f for f in (TARGET_ROOT / cls).iterdir()
                     if f.suffix.lower() in IMG_EXTENSIONS])
        bar_len = min(count // 20, 40)
        bar = "█" * bar_len
        need_more = max(0, 200 - count)
        status = "✓ 達標" if count >= 200 else f"⚠ 還需 {need_more} 張"
        print(f"    {cls:10s}: {count:5d} 張  {bar:40s}  {status}")

    print("\n  下一步：")
    print("    python scripts/03_organize_sdnet.py  # 補充 crack/normal")
    print("    python scripts/04_organize_dagm.py   # 補充 crack/stain")
    print("    python scripts/06_verify_dataset.py  # 驗證整體分布")
    print()


if __name__ == "__main__":
    organize_bd3()

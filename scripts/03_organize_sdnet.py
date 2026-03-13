#!/usr/bin/env python3
"""
03_organize_sdnet.py
====================
從 SDNET2018 資料集提取 224×224 patch，補充 crack 與 normal 類別。

SDNET2018 結構：
  D/CD/ — 橋面板裂縫 (Cracked Deck)
  D/UD/ — 橋面板正常 (Uncracked Deck)
  W/CW/ — 牆面裂縫  (Cracked Wall)
  W/UW/ — 牆面正常  (Uncracked Wall)
  P/CP/ — 路面裂縫  (Cracked Pavement)
  P/UP/ — 路面正常  (Uncracked Pavement)

原始圖片為 256×256 JPG，直接 center crop 到 224×224。

裂縫相關性：
  - CW（牆面）> CD（橋面）> CP（路面）對室內場景的遷移效果
  - 預設全部使用，可透過 USE_SUBSETS 調整

執行方式：
  python scripts/03_organize_sdnet.py [--max-per-source N]

  --max-per-source N：每個子集最多取幾張（預設 500，防止 normal 類過多）
"""

import argparse
import random
import sys
from pathlib import Path

from PIL import Image

# ── 路徑設定 ─────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
SDNET_RAW     = PROJECT_ROOT / "data" / "raw" / "sdnet2018"
TARGET_ROOT   = PROJECT_ROOT / "data" / "defects"

# ── 子集設定 ──────────────────────────────────────────────────
# (子集資料夾, 相對於 SDNET_RAW, 目標類別, 優先度說明)
SUBSETS = [
    # 優先：牆面裂縫，對室內場景最相關
    ("W/CW", "crack",  "牆面裂縫"),
    ("W/UW", "normal", "牆面正常"),
    # 次要：橋面板
    ("D/CD", "crack",  "橋面板裂縫"),
    ("D/UD", "normal", "橋面板正常"),
    # 路面（紋理差異較大，較不推薦，但有助於增加多樣性）
    ("P/CP", "crack",  "路面裂縫"),
    ("P/UP", "normal", "路面正常"),
]

PATCH_SIZE    = 224   # 輸出 patch 大小
IMG_SUFFIX    = ".jpg"
RANDOM_SEED   = 42


def center_crop_224(img: Image.Image, size: int = PATCH_SIZE) -> Image.Image:
    """從 256×256 圖片中心裁切 224×224。"""
    w, h = img.size
    left   = (w - size) // 2
    top    = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def find_sdnet_root(sdnet_raw: Path) -> Path:
    """
    SDNET 解壓後可能有不同的資料夾層次，自動找到含有 D/W/P 的根目錄。
    """
    if all((sdnet_raw / s).exists() for s in ["D", "W", "P"]):
        return sdnet_raw

    # 搜尋一層子目錄
    for child in sdnet_raw.iterdir():
        if child.is_dir():
            if all((child / s).exists() for s in ["D", "W", "P"]):
                return child

    return None


def organize_sdnet(max_per_source: int = 500):
    print("=" * 60)
    print("  SDNET2018 整理工具")
    print("=" * 60)

    # ── 確認原始資料 ───────────────────────────────────────────
    if not SDNET_RAW.exists():
        print(f"\n[錯誤] SDNET2018 資料夾不存在：{SDNET_RAW}")
        print("  請先執行：bash scripts/01_download_datasets.sh")
        sys.exit(1)

    sdnet_root = find_sdnet_root(SDNET_RAW)
    if sdnet_root is None:
        print(f"\n[錯誤] 無法在 {SDNET_RAW} 找到 D/W/P 結構")
        print("  請確認 SDNET2018.zip 是否已正確解壓。")
        sys.exit(1)

    print(f"\n  SDNET 根目錄：{sdnet_root}")
    print(f"  輸出目標  ：{TARGET_ROOT}")
    print(f"  每子集上限：{max_per_source} 張")

    random.seed(RANDOM_SEED)

    # ── 確認目標資料夾 ─────────────────────────────────────────
    for cls in ["crack", "normal"]:
        (TARGET_ROOT / cls).mkdir(parents=True, exist_ok=True)

    # ── 處理每個子集 ───────────────────────────────────────────
    print("\n  子集處理：")
    total_copied  = 0
    total_skipped = 0

    for subset_path, target_cls, description in SUBSETS:
        src_dir = sdnet_root / subset_path.replace("/", str(Path("/")))
        if not src_dir.exists():
            # 嘗試直接拼接（部分系統路徑分隔符不同）
            parts = subset_path.split("/")
            src_dir = sdnet_root
            for part in parts:
                src_dir = src_dir / part

        if not src_dir.exists():
            print(f"  [略過] {subset_path:8s} ({description})  — 資料夾不存在")
            continue

        dst_dir = TARGET_ROOT / target_cls
        images  = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.jpeg"))

        # 若超過上限，隨機抽樣
        if len(images) > max_per_source:
            images = random.sample(images, max_per_source)

        copied  = 0
        skipped = 0

        for img_path in images:
            subset_tag = subset_path.replace("/", "_")
            dst_name   = f"sdnet_{subset_tag}_{img_path.name}"
            dst_file   = dst_dir / dst_name

            if dst_file.exists():
                skipped += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                # 256×256 → 224×224 center crop
                if img.size != (PATCH_SIZE, PATCH_SIZE):
                    img = center_crop_224(img)
                img.save(dst_file, "JPEG", quality=95)
                copied += 1
            except Exception as e:
                print(f"  [警告] 處理失敗 {img_path.name}: {e}")
                continue

        total_copied  += copied
        total_skipped += skipped
        print(f"  [完成] {subset_path:8s} ({description:10s}) → {target_cls:10s}  "
              f"複製 {copied:4d} 張  略過 {skipped:4d} 張")

    # ── 最終統計 ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  總計複製：{total_copied} 張  略過（已存在）：{total_skipped} 張")
    print("\n  目前各類別數量：")
    for cls in ["normal", "crack", "stain", "mold", "peeling"]:
        count = len([f for f in (TARGET_ROOT / cls).iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        need_more = max(0, 200 - count)
        status = "✓ 達標" if count >= 200 else f"⚠ 還需 {need_more} 張"
        print(f"    {cls:10s}: {count:5d} 張  {status}")

    print("\n  下一步：")
    print("    python scripts/04_organize_dagm.py   # 補充 crack/stain/normal")
    print("    python scripts/06_verify_dataset.py  # 驗證整體分布")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="整理 SDNET2018 資料集")
    parser.add_argument(
        "--max-per-source", type=int, default=500,
        help="每個子集最多取幾張（預設 500）"
    )
    args = parser.parse_args()
    organize_sdnet(max_per_source=args.max_per_source)

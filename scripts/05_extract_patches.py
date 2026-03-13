#!/usr/bin/env python3
"""
05_extract_patches.py
=====================
從使用者自行拍攝的台灣房屋照片中提取 224×224 訓練 patch。

使用方式：
  1. 將照片依缺陷類型放入對應資料夾：
       user_photos/
         normal/    ← 乾淨正常的房間照片
         crack/     ← 有裂縫的照片
         stain/     ← 有污漬的照片
         mold/      ← 有霉斑的照片
         peeling/   ← 有剝落的照片
         # worn 已移除（定義模糊，以明確缺陷類別為主）

  2. 執行腳本：
       python scripts/05_extract_patches.py --input user_photos/

  注意：照片放入某類資料夾代表「這張照片主要呈現該缺陷」
        腳本會用滑動視窗提取 patch，並自動過濾掉：
          - 過於均勻（無紋理、空白牆面）的 patch
          - 過暗或過曝的 patch

參數說明：
  --input   DIR      自拍照片的根目錄（預設 user_photos/）
  --output  DIR      輸出目標（預設 data/defects/，直接合入訓練集）
  --stride  N        滑動視窗步進（預設 112，50% overlap）
  --min-var FLOAT    最小像素方差過濾值（預設 200，過低=均勻無紋理）
  --min-sat FLOAT    最小飽和度過濾值（預設 15，過低=灰白無色）
  --dry-run          只統計不實際複製
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── 預設路徑 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = PROJECT_ROOT / "user_photos"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "defects"

CLASSES       = ["normal", "crack", "stain", "mold", "peeling"]
PATCH_SIZE    = 224
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}


# ── 品質過濾函數 ───────────────────────────────────────────────

def compute_patch_variance(arr: np.ndarray) -> float:
    """計算 patch 的像素方差（灰階）。方差過低表示畫面過於均勻。"""
    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
    return float(gray.var())


def compute_mean_saturation(arr: np.ndarray) -> float:
    """計算 patch 的平均飽和度（HSV）。飽和度過低表示幾乎是灰白色。"""
    img_hsv = Image.fromarray(arr.astype(np.uint8)).convert("HSV")
    s_channel = np.array(img_hsv)[:, :, 1]
    return float(s_channel.mean())


def compute_brightness(arr: np.ndarray) -> float:
    """計算 patch 平均亮度（0~255）。"""
    return float(arr.mean())


def is_valid_patch(
    arr: np.ndarray,
    min_variance: float = 200.0,
    min_saturation: float = 15.0,
    min_brightness: float = 30.0,
    max_brightness: float = 235.0,
) -> tuple[bool, str]:
    """
    回傳 (是否有效, 拒絕原因)。
    """
    var = compute_patch_variance(arr)
    if var < min_variance:
        return False, f"方差過低({var:.1f}<{min_variance}，過於均勻)"

    brightness = compute_brightness(arr)
    if brightness < min_brightness:
        return False, f"過暗({brightness:.1f}<{min_brightness})"
    if brightness > max_brightness:
        return False, f"過曝({brightness:.1f}>{max_brightness})"

    # 飽和度只對彩色影像有意義
    sat = compute_mean_saturation(arr)
    if sat < min_saturation:
        return False, f"飽和度過低({sat:.1f}<{min_saturation}，近似灰白)"

    return True, ""


# ── 滑動視窗提取 ──────────────────────────────────────────────

def extract_patches_from_image(
    img_path: Path,
    patch_size: int = PATCH_SIZE,
    stride: int = 112,
    min_variance: float = 200.0,
    min_saturation: float = 15.0,
) -> list[np.ndarray]:
    """
    用滑動視窗從單張圖片提取有效 patch，回傳 ndarray list。
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  [警告] 無法開啟圖片 {img_path.name}: {e}")
        return []

    w, h   = img.size
    arr    = np.array(img)
    patches = []

    # 若圖片小於 patch_size，直接 resize 取一張
    if w < patch_size or h < patch_size:
        img_resized = img.resize((patch_size, patch_size), Image.LANCZOS)
        a = np.array(img_resized)
        valid, _ = is_valid_patch(a, min_variance, min_saturation)
        if valid:
            patches.append(a)
        return patches

    # 滑動視窗
    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            patch_arr = arr[top:top + patch_size, left:left + patch_size]
            valid, _ = is_valid_patch(patch_arr, min_variance, min_saturation)
            if valid:
                patches.append(patch_arr)

    return patches


# ── 主程式 ────────────────────────────────────────────────────

def extract_patches(
    input_dir: Path,
    output_dir: Path,
    stride: int = 112,
    min_variance: float = 200.0,
    min_saturation: float = 15.0,
    dry_run: bool = False,
):
    print("=" * 60)
    print("  台灣房屋照 Patch 提取工具")
    print("=" * 60)

    if not input_dir.exists():
        print(f"\n[錯誤] 輸入目錄不存在：{input_dir}")
        print("\n  請建立以下目錄結構，將照片放入對應類別資料夾：")
        print(f"    {input_dir}/")
        for cls in CLASSES:
            print(f"      {cls}/   ← {_cls_desc(cls)}")
        sys.exit(1)

    print(f"\n  輸入目錄：{input_dir}")
    print(f"  輸出目標：{output_dir}")
    print(f"  Patch大小：{PATCH_SIZE}×{PATCH_SIZE}")
    print(f"  滑動步進：{stride}px")
    print(f"  最小方差：{min_variance}")
    print(f"  最小飽和：{min_saturation}")
    if dry_run:
        print("  模式：DRY RUN（只統計，不複製）")

    # 確認輸出目錄
    if not dry_run:
        for cls in CLASSES:
            (output_dir / cls).mkdir(parents=True, exist_ok=True)

    total_images  = 0
    total_patches = 0
    total_saved   = 0

    print()
    for cls in CLASSES:
        cls_input  = input_dir / cls
        cls_output = output_dir / cls

        if not cls_input.exists():
            print(f"  [{cls:10s}] 資料夾不存在，略過")
            continue

        photos = [f for f in cls_input.iterdir()
                  if f.suffix.lower() in IMG_EXTENSIONS]

        if not photos:
            print(f"  [{cls:10s}] 資料夾為空，略過")
            continue

        cls_patches = 0
        cls_saved   = 0
        total_images += len(photos)

        for photo in photos:
            patches = extract_patches_from_image(
                photo, PATCH_SIZE, stride, min_variance, min_saturation
            )
            cls_patches += len(patches)

            if not dry_run:
                for i, patch_arr in enumerate(patches):
                    dst_name = f"user_{cls}_{photo.stem}_{i:04d}.jpg"
                    dst_file = cls_output / dst_name
                    if dst_file.exists():
                        continue
                    img = Image.fromarray(patch_arr.astype(np.uint8))
                    img.save(dst_file, "JPEG", quality=95)
                    cls_saved += 1

        total_patches += cls_patches
        total_saved   += cls_saved

        avg_per_photo = cls_patches / len(photos) if photos else 0
        action = f"儲存 {cls_saved} 張" if not dry_run else f"預計 {cls_patches} 張"
        print(f"  [{cls:10s}] 照片 {len(photos):3d} 張 → "
              f"patch {cls_patches:5d} 張（平均 {avg_per_photo:.1f}/張）  {action}")

    # ── 最終統計 ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  總計：來源照片 {total_images} 張 → "
          f"提取 patch {total_patches} 張")
    if not dry_run:
        print(f"  實際儲存（扣除重複）：{total_saved} 張")

    print("\n  目前各類別數量：")
    for cls in CLASSES:
        count = len([f for f in (output_dir / cls).iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        need_more = max(0, 200 - count)
        status = "✓ 達標" if count >= 200 else f"⚠ 還需 {need_more} 張"
        print(f"    {cls:10s}: {count:5d} 張  {status}")

    print("\n  建議：mold、peeling 類別較難收集")
    print("    → 可從 591.com.tw、樂屋網 等台灣租屋平台截圖補充")
    print("    → 每類達到 200 張後執行：python scripts/06_verify_dataset.py")
    print()


def _cls_desc(cls: str) -> str:
    desc = {
        "normal":  "乾淨正常的房間照片",
        "crack":   "有裂縫的牆面/地板照片",
        "stain":   "有污漬/水漬的照片",
        "mold":    "有霉斑（黑/綠點）的照片",
        "peeling": "有漆面/壁紙剝落的照片",
        # worn 已移除
    }
    return desc.get(cls, "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="從台灣房屋自拍照提取 224×224 訓練 patch"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"自拍照片根目錄（預設：{DEFAULT_INPUT}）"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"輸出目標（預設：{DEFAULT_OUTPUT}）"
    )
    parser.add_argument(
        "--stride", type=int, default=112,
        help="滑動視窗步進 px（預設 112，50% overlap）"
    )
    parser.add_argument(
        "--min-var", type=float, default=200.0,
        help="最小像素方差（預設 200，過低=均勻無紋理）"
    )
    parser.add_argument(
        "--min-sat", type=float, default=15.0,
        help="最小飽和度（預設 15，過低=灰白無色）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只統計，不實際複製"
    )
    args = parser.parse_args()

    extract_patches(
        input_dir     = args.input,
        output_dir    = args.output,
        stride        = args.stride,
        min_variance  = args.min_var,
        min_saturation= args.min_sat,
        dry_run       = args.dry_run,
    )

#!/usr/bin/env python3
"""
05_extract_patches.py
=====================
從台灣房屋照片中提取 384×384 訓練 patch（對應 640×360 輸入場景）。

使用方式：
  1. 將照片依缺陷類型放入對應資料夾：
       user_photos/
         normal/    ← 乾淨正常的房間照片
         crack/     ← 有裂縫的照片
         stain/     ← 有污漬的照片
         mold/      ← 有霉斑的照片
         peeling/   ← 有剝落的照片

  2. 執行腳本：
       python scripts/05_extract_patches.py --input user_photos/

變更記錄：
  - PATCH_SIZE: 224 → 384（對應 640×360 全幀輸入）
  - 新增 --keep-aspect 模式：將 640×360 padding 成 640×640 再取 384 crop
    避免強制 resize 壓縮細節
  - stride 預設從 112 → 192（384 的 50% overlap）

參數說明：
  --input      DIR    自拍照片的根目錄（預設 user_photos/）
  --output     DIR    輸出目標（預設 data/defects/，直接合入訓練集）
  --stride     N      滑動視窗步進（預設 192，50% overlap）
  --min-var    FLOAT  最小像素方差過濾值（預設 300）
  --min-sat    FLOAT  最小飽和度過濾值（預設 15）
  --no-padding        關閉寬景 padding（不建議）
  --dry-run           只統計不實際複製
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = PROJECT_ROOT / "user_photos"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "defects"

CLASSES        = ["normal", "crack", "stain", "mold", "peeling"]
PATCH_SIZE     = 384
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}


# ── 品質過濾 ──────────────────────────────────────────────────────────

def compute_patch_variance(arr: np.ndarray) -> float:
    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
    return float(gray.var())


def compute_mean_saturation(arr: np.ndarray) -> float:
    img_hsv = Image.fromarray(arr.astype(np.uint8)).convert("HSV")
    s_channel = np.array(img_hsv)[:, :, 1]
    return float(s_channel.mean())


def compute_brightness(arr: np.ndarray) -> float:
    return float(arr.mean())


def is_valid_patch(
    arr: np.ndarray,
    min_variance: float = 300.0,   # 384px patch 面積更大，方差門檻提高
    min_saturation: float = 15.0,
    min_brightness: float = 30.0,
    max_brightness: float = 235.0,
) -> tuple[bool, str]:
    var = compute_patch_variance(arr)
    if var < min_variance:
        return False, f"方差過低({var:.1f}<{min_variance})"

    brightness = compute_brightness(arr)
    if brightness < min_brightness:
        return False, f"過暗({brightness:.1f})"
    if brightness > max_brightness:
        return False, f"過曝({brightness:.1f})"

    sat = compute_mean_saturation(arr)
    if sat < min_saturation:
        return False, f"飽和度過低({sat:.1f})"

    return True, ""


# ── 寬景 Padding 模式 ─────────────────────────────────────────────────

def pad_to_square(img: Image.Image) -> Image.Image:
    """將 640×360 padding 成 640×640（置中補黑），保留原始寬高比。"""
    w, h = img.size
    max_side = max(w, h)
    padded = ImageOps.pad(img, (max_side, max_side), color=(0, 0, 0))
    return padded


# ── 滑動視窗提取 ──────────────────────────────────────────────────────

def extract_patches_from_image(
    img_path: Path,
    patch_size: int = PATCH_SIZE,
    stride: int = 192,
    min_variance: float = 300.0,
    min_saturation: float = 15.0,
    keep_aspect: bool = True,
) -> list[np.ndarray]:
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  [警告] 無法開啟圖片 {img_path.name}: {e}")
        return []

    if keep_aspect:
        img = pad_to_square(img)

    w, h   = img.size
    arr    = np.array(img)
    patches = []

    if w < patch_size or h < patch_size:
        img_resized = img.resize((patch_size, patch_size), Image.LANCZOS)
        a = np.array(img_resized)
        valid, _ = is_valid_patch(a, min_variance, min_saturation)
        if valid:
            patches.append(a)
        return patches

    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            patch_arr = arr[top:top + patch_size, left:left + patch_size]
            valid, _ = is_valid_patch(patch_arr, min_variance, min_saturation)
            if valid:
                patches.append(patch_arr)

    return patches


# ── 主程式 ────────────────────────────────────────────────────────────

def extract_patches(
    input_dir: Path,
    output_dir: Path,
    stride: int = 192,
    min_variance: float = 300.0,
    min_saturation: float = 15.0,
    keep_aspect: bool = True,
    dry_run: bool = False,
):
    print("=" * 60)
    print("  台灣房屋照 Patch 提取工具（384px 高解析度版）")
    print("=" * 60)

    if not input_dir.exists():
        print(f"\n[錯誤] 輸入目錄不存在：{input_dir}")
        sys.exit(1)

    print(f"\n  輸入目錄：{input_dir}")
    print(f"  輸出目標：{output_dir}")
    print(f"  Patch大小：{PATCH_SIZE}×{PATCH_SIZE}")
    print(f"  滑動步進：{stride}px（50% overlap）")
    print(f"  寬景Padding：{'開啟' if keep_aspect else '關閉'}")
    if dry_run:
        print("  模式：DRY RUN")

    if not dry_run:
        for cls in CLASSES:
            (output_dir / cls).mkdir(parents=True, exist_ok=True)

    total_images = total_patches = total_saved = 0

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

        cls_patches = cls_saved = 0
        total_images += len(photos)

        for photo in photos:
            patches = extract_patches_from_image(
                photo, PATCH_SIZE, stride, min_variance, min_saturation, keep_aspect
            )
            cls_patches += len(patches)

            if not dry_run:
                for i, patch_arr in enumerate(patches):
                    dst_name = f"user_{cls}_{photo.stem}_{i:04d}.jpg"
                    dst_file = cls_output / dst_name
                    if dst_file.exists():
                        continue
                    Image.fromarray(patch_arr.astype(np.uint8)).save(
                        dst_file, "JPEG", quality=95)
                    cls_saved += 1

        total_patches += cls_patches
        total_saved   += cls_saved

        avg = cls_patches / len(photos) if photos else 0
        action = f"儲存 {cls_saved} 張" if not dry_run else f"預計 {cls_patches} 張"
        print(f"  [{cls:10s}] 照片 {len(photos):3d} 張 → "
              f"patch {cls_patches:5d} 張（平均 {avg:.1f}/張）  {action}")

    print("\n" + "─" * 60)
    print(f"  總計：來源照片 {total_images} 張 → patch {total_patches} 張")
    if not dry_run:
        print(f"  實際儲存（扣除重複）：{total_saved} 張")

    print("\n  目前各類別數量：")
    for cls in CLASSES:
        count = len([f for f in (output_dir / cls).iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        need  = max(0, 150 - count)
        status = "✓ 達標" if count >= 150 else f"⚠ 還需 {need} 張"
        print(f"    {cls:10s}: {count:5d} 張  {status}")

    print("\n  注意：384px patch 每張含更多空間資訊，150 張即可開始訓練")
    print("  達標後執行：python scripts/06_verify_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從台灣房屋自拍照提取 384×384 訓練 patch")
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stride", type=int,  default=192,
        help="滑動視窗步進 px（預設 192，50% overlap）")
    parser.add_argument("--min-var",   type=float, default=300.0)
    parser.add_argument("--min-sat",   type=float, default=15.0)
    parser.add_argument("--no-padding", action="store_true",
        help="關閉寬景 padding（不建議）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    extract_patches(
        input_dir      = args.input,
        output_dir     = args.output,
        stride         = args.stride,
        min_variance   = args.min_var,
        min_saturation = args.min_sat,
        keep_aspect    = not args.no_padding,
        dry_run        = args.dry_run,
    )

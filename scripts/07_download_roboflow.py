#!/usr/bin/env python3
"""
07_download_roboflow.py
=======================
從 Roboflow Universe 下載建築缺陷資料集，補充 mold/stain/peeling 類別。

目標資料集：
  1. builddef2/building-defect-on-walls
     類別：crack, mold, stain, damp, peeling paint, water seepage

  2. jonathan-lewis-8tb5o/water-damage-finder
     類別：water damage（對應 stain/mold）

需要：
  - 免費 Roboflow 帳號（https://roboflow.com）
  - API Key（https://app.roboflow.com/settings/api）
  - pip install roboflow pillow

執行方式：
  python scripts/07_download_roboflow.py --api-key YOUR_API_KEY_HERE

  # 或設定環境變數（推薦，避免 key 出現在 shell history）：
  export ROBOFLOW_API_KEY=YOUR_API_KEY_HERE
  python scripts/07_download_roboflow.py

Roboflow 類別 → 本專案類別對應：
  crack           → crack
  mold            → mold
  stain           → stain
  damp            → stain    （潮濕/滲水漬，外觀近似污漬）
  dampness        → stain
  water seepage   → stain
  peeling paint   → peeling
  peeling         → peeling
  water damage    → stain
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# ── 路徑設定 ─────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RAW_ROBOFLOW  = PROJECT_ROOT / "data" / "raw" / "roboflow"
TARGET_ROOT   = PROJECT_ROOT / "data" / "defects"

# ── Roboflow 資料集清單 ────────────────────────────────────────
# (workspace, project, version, 說明)
DATASETS = [
    ("builddef2",              "building-defect-on-walls", 1, "建築牆面缺陷（mold/stain/peeling）"),
    ("jonathan-lewis-8tb5o",   "water-damage-finder",      1, "水損偵測（stain/mold 補充）"),
]

# ── Roboflow 類別 → 本專案類別對應 ────────────────────────────
LABEL_MAP = {
    "crack":         "crack",
    "mold":          "mold",
    "mould":         "mold",
    "stain":         "stain",
    "damp":          "stain",
    "dampness":      "stain",
    "water seepage": "stain",
    "water damage":  "stain",
    "peeling paint": "peeling",
    "peeling":       "peeling",
    "paint peeling": "peeling",
    "normal":        "normal",
}

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def download_roboflow_datasets(api_key: str):
    print("=" * 60)
    print("  Roboflow 資料集下載工具")
    print("=" * 60)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("\n[錯誤] 未安裝 roboflow 套件")
        print("  請執行：pip install roboflow")
        sys.exit(1)

    RAW_ROBOFLOW.mkdir(parents=True, exist_ok=True)
    for cls in TARGET_ROOT.iterdir() if TARGET_ROOT.exists() else []:
        pass
    for cls_name in ["normal", "crack", "stain", "mold", "peeling", "worn"]:
        (TARGET_ROOT / cls_name).mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    total_copied = 0

    for workspace, project_name, version, description in DATASETS:
        print(f"\n  下載：{workspace}/{project_name} v{version}")
        print(f"  說明：{description}")

        download_dir = RAW_ROBOFLOW / f"{workspace}_{project_name}_v{version}"

        try:
            project  = rf.workspace(workspace).project(project_name)
            dataset  = project.version(version).download(
                "folder",
                location=str(download_dir),
                overwrite=False,
            )
            print(f"  下載完成：{download_dir}")
        except Exception as e:
            print(f"  [錯誤] 下載失敗：{e}")
            print(f"  請確認 API Key 正確，且有權限存取此資料集")
            continue

        # ── 整理圖片到目標類別資料夾 ────────────────────────
        copied = _organize_roboflow_folder(download_dir, TARGET_ROOT, workspace, project_name)
        total_copied += copied
        print(f"  已整理 {copied} 張圖片到 data/defects/")

    # ── 最終統計 ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  總計整理：{total_copied} 張")
    print("\n  目前各類別數量：")
    for cls in ["normal", "crack", "stain", "mold", "peeling", "worn"]:
        count = sum(1 for f in (TARGET_ROOT / cls).iterdir()
                    if f.suffix.lower() in IMG_EXTS)
        need_more = max(0, 200 - count)
        status = "✓ 達標" if count >= 200 else f"⚠ 還需 {need_more} 張"
        print(f"    {cls:10s}: {count:5d} 張  {status}")
    print()


def _organize_roboflow_folder(
    download_dir: Path,
    target_root: Path,
    workspace: str,
    project: str,
) -> int:
    """
    Roboflow 下載後的資料夾結構通常是：
      download_dir/train/crack/image.jpg
      download_dir/valid/mold/image.jpg
      download_dir/test/stain/image.jpg
    或 classification 格式：
      download_dir/train/class_name/image.jpg

    掃描所有子資料夾，依資料夾名稱對應類別。
    """
    copied = 0
    prefix = f"rf_{workspace[:6]}_{project[:8]}"

    # 掃描所有含有圖片的資料夾
    for subdir in download_dir.rglob("*"):
        if not subdir.is_dir():
            continue

        cls_label = subdir.name.lower().strip()
        target_cls = LABEL_MAP.get(cls_label)
        if target_cls is None:
            continue  # 未在對應表中的類別略過

        dst_dir = target_root / target_cls
        for img_path in subdir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            dst_name = f"{prefix}_{cls_label.replace(' ', '_')}_{img_path.name}"
            dst_file = dst_dir / dst_name
            if dst_file.exists():
                continue
            shutil.copy2(img_path, dst_file)
            copied += 1

    return copied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="從 Roboflow Universe 下載建築缺陷資料集"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Roboflow API Key（也可設定 ROBOFLOW_API_KEY 環境變數）"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("[錯誤] 請提供 Roboflow API Key")
        print("  方法一：--api-key YOUR_KEY")
        print("  方法二：export ROBOFLOW_API_KEY=YOUR_KEY")
        print("\n  取得 API Key：https://app.roboflow.com/settings/api")
        sys.exit(1)

    download_roboflow_datasets(api_key)

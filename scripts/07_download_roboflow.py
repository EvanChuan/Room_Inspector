#!/usr/bin/env python3
"""
07_download_roboflow.py
=======================
從 Roboflow Universe 下載建築缺陷 / 牆面缺陷相關資料集，
補充 mold / stain / peeling 類別（以及部分 crack）。

重要說明：
- 原本的 jonathan-lewis-8tb5o/water-damage-finder 資料集目前已找不到
  對應的 Universe 頁面，故已移除。
- builddef2/building-defect-on-walls 仍存在，但格式可能因版本更動，
  我們統一用「folder」下載，直接從資料夾名稱推類別。

目前使用的資料集：
【mold 補充】
  1. research-placement/mouldy-wall-classification
     類別：Clean, Mould（分類，32 張）

  2. aravdash/mold-d8s90
     類別：mold, pest（分類，~685 張 mold）

【peeling + stain + crack 多類】
  3. yolov1-dvce8/bulding-defects
     類別：algae, major_crack, minor_crack, normal, peeling, spalling, stain（分類）
     ★ 這是 BD3 資料集在 Roboflow 上的版本，含 peeling/stain 最佳來源

  4. builddef2/building-defect-on-walls  v4
     類別：crack, mold, peeling_paint, stairstep_crack, water_seepage（偵測，472 張）

【stain 補充】
  5. stain/stain-detection
     類別：stain（偵測，215 張）

  6. swu-f1hr8/stain-classification-stptx
     類別：多種 stain 類別（分類，取圖用）

  7. rust-detection/rust-detection-38s6e
     類別：rust（分割，10,072 張）→ 對應 stain（鐵鏽污漬）

【牆面缺陷雜項】
  8. wall-defects/wall-defects-ysyha
     類別：cracks、rust stain 等牆面缺陷（偵測）

需要：
  - 免費 Roboflow 帳號（[https://roboflow.com](https://roboflow.com)）
  - API Key（[https://app.roboflow.com/settings/api](https://app.roboflow.com/settings/api)）
  - pip install roboflow pillow

執行方式：
  python scripts/07_download_roboflow.py --api-key YOUR_API_KEY_HERE

  # 或設定環境變數（推薦，避免 key 出現在 shell history）：
  export ROBOFLOW_API_KEY=YOUR_API_KEY_HERE
  python scripts/07_download_roboflow.py

Roboflow 類別 → 本專案類別對應：
  crack, cracks, wall_crack         → crack
  mold, mould, wall_mold           → mold
  stain, rust_stain, water_damage  → stain
  damp, dampness, seepage          → stain
  peeling, peeling_paint           → peeling
  paint_peeling                    → peeling
  clean / normal                   → normal
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# ── 路徑設定 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROBOFLOW = PROJECT_ROOT / "data" / "raw" / "roboflow"
TARGET_ROOT  = PROJECT_ROOT / "data" / "defects"

# ── Roboflow 資料集清單 ────────────────────────────────────────
# (workspace, project, version, 說明)
DATASETS = [
    # ── mold 類別 ─────────────────────────────────────────────
    ("research-placement",  "mouldy-wall-classification",   1, "牆面霉斑分類（Clean/Mould，32 張）"),
    ("aravdash",            "mold-d8s90",                   1, "mold 分類（~685 張 mold）"),

    # ── peeling + stain + crack 多類（最重要）────────────────
    # BD3 在 Roboflow 上的完整分類版本，含 peeling / stain / crack
    ("yolov1-dvce8",        "bulding-defects",              1, "★BD3 Roboflow版：peeling/stain/crack/normal 7類分類"),

    # 建築牆面偵測（含 peeling_paint / water_seepage）
    ("builddef2",           "building-defect-on-walls",     4, "牆面缺陷偵測（472張，crack/mold/peeling_paint/water_seepage）"),

    # ── stain 補充 ────────────────────────────────────────────
    ("stain",               "stain-detection",              1, "stain 偵測（215 張）"),
    ("swu-f1hr8",           "stain-classification-stptx",   1, "stain 分類（多子類）"),

    # 鐵鏽/污漬（10K+ 張，對應 stain 類）
    ("rust-detection",      "rust-detection-38s6e",         1, "rust/corrosion（10,072 張）→ stain"),

    # ── 牆面缺陷雜項 ──────────────────────────────────────────
    ("wall-defects",        "wall-defects-ysyha",           1, "牆面缺陷雜項（crack/rust stain）"),
]

# ── Roboflow 類別 → 本專案類別對應 ────────────────────────────
LABEL_MAP = {
    # ── crack 類 ──────────────────────────────────────────────
    "crack":            "crack",
    "cracks":           "crack",
    "wall_crack":       "crack",
    "major_crack":      "crack",
    "major crack":      "crack",
    "minor_crack":      "crack",
    "minor crack":      "crack",
    "stairstep_crack":  "crack",
    "stairstep crack":  "crack",

    # ── mold 類 ───────────────────────────────────────────────
    "mold":             "mold",
    "mould":            "mold",
    "mouldy":           "mold",
    "wall_mold":        "mold",
    "wall_mould":       "mold",

    # ── stain / water damage / rust 類 ────────────────────────
    "stain":            "stain",
    "stains":           "stain",
    "rust":             "stain",   # 鐵鏽 → stain
    "rust_stain":       "stain",
    "rust stain":       "stain",
    "corrosion":        "stain",
    "water_damage":     "stain",
    "water damage":     "stain",
    "water_seepage":    "stain",
    "water seepage":    "stain",
    "damp":             "stain",
    "dampness":         "stain",
    "seepage":          "stain",
    "leakage":          "stain",

    # ── peeling 類 ────────────────────────────────────────────
    "peeling":          "peeling",
    "peeling_paint":    "peeling",
    "peeling paint":    "peeling",
    "paint_peeling":    "peeling",
    "paint peeling":    "peeling",
    "spalling":         "peeling",  # 表面剝落 → peeling

    # ── algae → mold（BD3 的 algae 類別）─────────────────────
    "algae":            "mold",

    # ── normal / clean 類 ─────────────────────────────────────
    "normal":           "normal",
    "clean":            "normal",
    "no_defect":        "normal",
    "no defect":        "normal",
}

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def download_roboflow_datasets(api_key: str):
    print("=" * 60)
    print("  Roboflow 資料集下載工具（mold/stain/peeling/crack 補充）")
    print("=" * 60)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("\n[錯誤] 未安裝 roboflow 套件")
        print("  請執行：pip install roboflow")
        sys.exit(1)

    RAW_ROBOFLOW.mkdir(parents=True, exist_ok=True)

    # 確保目標類別資料夾存在
    for cls_name in ["normal", "crack", "stain", "mold", "peeling", "worn"]:
        (TARGET_ROOT / cls_name).mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    total_copied = 0

    for workspace, project_name, version, description in DATASETS:
        print(f"\n  下載：{workspace}/{project_name} v{version}")
        print(f"  說明：{description}")

        download_dir = RAW_ROBOFLOW / f"{workspace}_{project_name}_v{version}"

        try:
            project = rf.workspace(workspace).project(project_name)
            # 統一使用 "folder" 形式，Roboflow 會依資料集類型決定具體格式
            dataset = project.version(version).download(
                "folder",
                location=str(download_dir),
                overwrite=False,
            )
            print(f"  下載完成：{download_dir}")
        except Exception as e:
            print(f"  [錯誤] 下載失敗：{e}")
            print("  請確認：")
            print("    1) API Key 正確")
            print("    2) 該資料集為公開或你有權限存取")
            continue

        # ── 整理圖片到目標類別資料夾 ────────────────────────────
        copied = _organize_roboflow_folder(download_dir, TARGET_ROOT, workspace, project_name)
        total_copied += copied
        print(f"  已整理 {copied} 張圖片到 data/defects/")

    # ── 最終統計 ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  總計整理：{total_copied} 張")
    print("\n  目前各類別數量：")
    for cls in ["normal", "crack", "stain", "mold", "peeling", "worn"]:
        cls_dir = TARGET_ROOT / cls
        if not cls_dir.exists():
            count = 0
        else:
            count = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
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
    通用整理邏輯（不管是 classification 或 object detection）：

    Roboflow "folder" 格式常見結構：
      - download_dir/train/<class>/image.jpg
      - download_dir/valid/<class>/image.jpg
      - download_dir/test/<class>/image.jpg

    或偵測格式：
      - download_dir/train/images/image.jpg
      - download_dir/train/labels/image.txt
      - download_dir/valid/images/...
      - 類別資訊藏在 YAML 或 labels 中

    為了避免依賴 Roboflow 的 labels / YAML 格式，我們採用簡單策略：
      1. 掃描所有子資料夾
      2. 把「最底層的資料夾名稱」視為原始類別標籤（小寫）
      3. 嘗試用 LABEL_MAP 映射到本專案類別（normal/crack/stain/mold/peeling）
      4. 將圖片複製到對應的 data/defects/<target_class>/ 中

    這樣即使資料集格式略有不同，也能最大化利用圖片。
    """
    copied = 0
    prefix = f"rf_{workspace[:6]}_{project[:8]}"

    # 掃描所有含有圖片的資料夾
    for subdir in download_dir.rglob("*"):
        if not subdir.is_dir():
            continue

        # 如果這個資料夾底下沒有圖片，略過
        has_image = any(
            (subdir / f).suffix.lower() in IMG_EXTS
            for f in os.listdir(subdir)
            if (subdir / f).is_file()
        )
        if not has_image:
            continue

        cls_label_raw = subdir.name
        cls_label = cls_label_raw.lower().strip()

        target_cls = LABEL_MAP.get(cls_label)
        if target_cls is None:
            # 看看父資料夾是否是類別名（例如 classification/train/mold/）
            parent_label = subdir.parent.name.lower().strip()
            target_cls = LABEL_MAP.get(parent_label)
            if target_cls is None:
                # 找不到對應就略過
                continue

        dst_dir = target_root / target_cls

        for img_name in os.listdir(subdir):
            img_path = subdir / img_name
            if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
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
        description="從 Roboflow Universe 下載建築/牆面缺陷資料集（補充 mold/stain/peeling/crack）"
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

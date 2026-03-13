<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 根據份流程中場景辨識與屋況檢測流程中，加入 Room_Inspector 系統模型來提高準確度，我想要檢視目前測試結果，可以用哪些量化的方式？或是debug模式檢視？

我已完整讀取 `scene_semantic_analysis_v2.py`（62,335 字元）。我發現了幾個關鍵接口問題，同時也整理出你可以立即使用的完整量化評估方案。

***

## 發現的兩個接口問題（必須先修正）

**問題一：checkpoint 路徑不符**

```python
# scene_semantic_analysis_v2.py 目前寫的
_ROOM_INSPECTOR_CKPT = Path("/home/evan/Room_Inspector/checkpoints/best.pt")

# 但你 train.py 實際存檔的路徑是
SAVE_PATH = "checkpoints/condition_model.pt"
```

**問題二：模型架構不符**

`_load_condition_model()` 裡用 `timm` 載入 `swin_t`，但你 Room_Inspector 訓練的是 `torchvision.models.vit_b_16`（ViT-B/16）：

```python
# 現在的錯誤做法
model = timm.create_model("swin_tiny_patch4_window7_224", ...)   # ❌ 架構不符

# 應該改成
from torchvision import models
import torch.nn as nn
model = models.vit_b_16(weights=None)
model.heads.head = nn.Linear(768, len(classes))                  # ✅ 對的
```


***

## 修正 `_load_condition_model()` 方法

把 `ConditionInspector` 裡的 `_load_condition_model()` 全段換成以下版本：

```python
def _load_condition_model(self, model_path: str):
    """
    載入 Room_Inspector 訓練的 ViT-B/16 模型。
    與 Room_Inspector/train.py 的 SAVE_PATH / CLASSES 完全對應。
    """
    from torchvision import models, transforms
    import torch.nn as nn

    CLASSES = ["normal", "crack", "stain", "mold", "peeling", "worn"]
    model_path = Path(model_path)

    # ── 建立 ViT-B/16 分類頭（與 train.py 完全一致）──
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(
        model.heads.head.in_features,  # 768
        len(CLASSES)
    )

    # ── 載入 state_dict（train.py 直接存 model.state_dict()）──
    ckpt = torch.load(str(model_path), map_location=self.device)

    # 相容兩種存法：直接 state_dict / {"model": state_dict}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(self.device).eval()

    self.condition_model = model
    self._vit_classes = CLASSES

    # ── 推論前處理（與 train.py val_transform 一致）──
    self._vit_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    print(f"[ConditionInspector] Layer3 ViT-B/16 已載入：{model_path.name}")
    print(f"  類別: {CLASSES}")
```

同時把 checkpoint 路徑改成：

```python
_ROOM_INSPECTOR_CKPT = Path("/home/evan/Room_Inspector/checkpoints/condition_model.pt")
```


***

## 量化評估方案

以下是一個完整獨立的評估腳本 `eval_condition_inspector.py`，可以同時量化三層（Layer1/2/3）的表現與一致性：

```python
#!/usr/bin/env python3
"""
eval_condition_inspector.py
============================
對 scene_semantic_analysis_v2.py 中的 ConditionInspector 進行量化評估。

用法：
  python eval_condition_inspector.py \
    --data_dir /home/evan/Room_Inspector/data/defects \
    --ckpt    /home/evan/Room_Inspector/checkpoints/condition_model.pt \
    --output  eval_output/
"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── 從你的系統引入 ConditionInspector ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene_semantic_analysis_v2 import ConditionInspector, SceneClassifier, DEVICE

CLASSES = ["normal", "crack", "stain", "mold", "peeling", "worn"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ─────────────────────────────────────────────────
# 1. 載入測試資料（資料夾 = 類別標籤）
# ─────────────────────────────────────────────────
def load_test_data(data_dir: str, max_per_class: int = 100):
    """掃描 data/defects/<class>/ 資料夾，回傳 [(path, label), ...]"""
    samples = []
    data_path = Path(data_dir)
    for cls in CLASSES:
        cls_dir = data_path / cls
        if not cls_dir.exists():
            print(f"  [WARN] 找不到類別資料夾：{cls_dir}")
            continue
        imgs = [f for f in cls_dir.iterdir()
                if f.suffix.lower() in IMG_EXTS][:max_per_class]
        for img in imgs:
            samples.append((str(img), cls))
    print(f"[資料載入] 共 {len(samples)} 張（各最多 {max_per_class} 張）")
    return samples


# ─────────────────────────────────────────────────
# 2. 執行推論，收集原始結果
# ─────────────────────────────────────────────────
def run_inference(inspector: ConditionInspector, samples: list) -> list:
    records = []
    total = len(samples)
    for i, (path, gt_label) in enumerate(samples, 1):
        if i % 20 == 0 or i == total:
            print(f"  推論進度：{i}/{total}", end="\r")
        try:
            result = inspector.analyze(path)
        except Exception as e:
            print(f"\n  [ERROR] {path}: {e}")
            continue

        records.append({
            "path":          path,
            "gt":            gt_label,
            # ── 各層原始輸出 ──
            "l1_flags":      result["details"]["layer1"]["flags_l1"],
            "l1_crack_cnt":  result["details"]["layer1"]["crack_line_count"],
            "l1_dark_ratio": result["details"]["layer1"]["dark_area_ratio"],
            "l1_mold_ratio": result["details"]["layer1"]["mold_area_ratio"],
            "l2_defect":     result["details"]["layer2"]["defect_label"],
            "l2_conf":       result["details"]["layer2"]["defect_conf"],
            "l3_defect":     (result["details"]["layer3"] or {}).get("vit_defect_label"),
            "l3_conf":       (result["details"]["layer3"] or {}).get("vit_defect_conf", 0.0),
            # ── 最終融合輸出 ──
            "final_defect":  result["defect_label"],
            "final_source":  result["defect_source"],
            "condition_score": result["condition_score"],
            "risk_flags":    result["risk_flags"],
        })
    print()
    return records


# ─────────────────────────────────────────────────
# 3. 計算各指標
# ─────────────────────────────────────────────────
def compute_metrics(records: list, source: str = "final") -> dict:
    """
    source: "final" | "l2" | "l3"
    計算 accuracy、per-class precision/recall/f1、confusion matrix。
    """
    label_to_pred = {"final": "final_defect", "l2": "l2_defect", "l3": "l3_defect"}
    pred_key = label_to_pred[source]

    gts, preds = [], []
    for r in records:
        pred = r[pred_key]
        if pred is None:
            pred = "normal"  # Layer3 未啟用時 fallback
        gts.append(r["gt"])
        preds.append(pred)

    # ── Confusion Matrix ──
    n = len(CLASSES)
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    cm = np.zeros((n, n), dtype=int)
    for g, p in zip(gts, preds):
        if g in cls_idx and p in cls_idx:
            cm[cls_idx[g]][cls_idx[p]] += 1

    # ── Per-Class Metrics ──
    per_class = {}
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = cm[i, :].sum()
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   int(support),
        }

    # ── Overall Accuracy ──
    accuracy = cm.diagonal().sum() / cm.sum() if cm.sum() > 0 else 0.0

    return {
        "accuracy":  round(float(accuracy), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────────
# 4. 層間一致性分析
# ─────────────────────────────────────────────────
def layer_agreement_analysis(records: list) -> dict:
    """分析 Layer2（CLIP）與 Layer3（ViT）的預測一致性"""
    both = [r for r in records if r["l3_defect"] is not None]
    if not both:
        return {"note": "Layer3 未啟用，無法比較"}

    agree    = sum(1 for r in both if r["l2_defect"] == r["l3_defect"])
    disagree = len(both) - agree
    agree_rate = agree / len(both) if both else 0.0

    # ── 不一致時，哪一層跟 ground truth 更接近 ──
    l2_correct = sum(1 for r in both if r["l2_defect"] == r["gt"])
    l3_correct = sum(1 for r in both if r["l3_defect"] == r["gt"])

    # ── 不一致樣本細節（最多 10 筆）──
    disagreement_samples = []
    for r in both:
        if r["l2_defect"] != r["l3_defect"]:
            disagreement_samples.append({
                "file":     Path(r["path"]).name,
                "gt":       r["gt"],
                "l2":       f"{r['l2_defect']}({r['l2_conf']:.2f})",
                "l3":       f"{r['l3_defect']}({r['l3_conf']:.2f})",
                "final":    r["final_defect"],
                "source":   r["final_source"],
            })

    return {
        "total_compared":   len(both),
        "agree_count":      agree,
        "disagree_count":   disagree,
        "agree_rate":       round(agree_rate, 4),
        "l2_acc_on_disagree": round(l2_correct / len(both), 4),
        "l3_acc_on_disagree": round(l3_correct / len(both), 4),
        "samples":          disagreement_samples[:10],
    }


# ─────────────────────────────────────────────────
# 5. 閾值敏感度分析（Layer3 conf threshold）
# ─────────────────────────────────────────────────
def threshold_sensitivity(records: list) -> list:
    """
    分析 Layer3 信心度門檻（0.40~0.80）對 final 準確率的影響，
    幫助你決定 _layer3_vit() 裡 0.55 這個門檻是否最優。
    """
    both = [r for r in records if r["l3_defect"] is not None]
    if not both:
        return []

    results = []
    for thr in np.arange(0.40, 0.85, 0.05):
        correct = 0
        for r in both:
            # 模擬：若 l3 conf > thr 且非 normal → 用 l3，否則用 l2
            l3_label = r["l3_defect"]
            l3_conf  = r["l3_conf"]
            if l3_label != "normal" and l3_conf > thr:
                pred = l3_label
            elif r["l2_defect"] != "normal" and r["l2_conf"] > 0.38:
                pred = r["l2_defect"]
            else:
                pred = "normal"
            if pred == r["gt"]:
                correct += 1
        acc = correct / len(both)
        results.append({"threshold": round(float(thr), 2), "accuracy": round(acc, 4)})
    return results


# ─────────────────────────────────────────────────
# 6. 視覺化：Confusion Matrix + Threshold Curve
# ─────────────────────────────────────────────────
def plot_confusion_matrix(cm_list: list, title: str, output_path: str):
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(8, 7))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)),
           xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="Ground Truth", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = cm[i, j]
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [圖表] 混淆矩陣已儲存：{output_path}")


def plot_threshold_curve(thr_results: list, output_path: str):
    if not thr_results:
        return
    xs = [r["threshold"] for r in thr_results]
    ys = [r["accuracy"]  for r in thr_results]
    best = max(thr_results, key=lambda x: x["accuracy"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="o", linewidth=2, color="#4C72B0")
    ax.axvline(best["threshold"], color="red", linestyle="--", alpha=0.7,
               label=f"Best thr={best['threshold']} acc={best['accuracy']:.4f}")
    ax.axvline(0.55, color="orange", linestyle=":", alpha=0.8,
               label="Current thr=0.55")
    ax.set(xlabel="Layer3 Confidence Threshold",
           ylabel="Accuracy (on samples with Layer3)",
           title="Threshold Sensitivity（Layer3 ViT）")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [圖表] 門檻曲線已儲存：{output_path}")


# ─────────────────────────────────────────────────
# 7. 印出 Classification Report（仿 sklearn 格式）
# ─────────────────────────────────────────────────
def print_report(metrics: dict, title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    print(f"  Overall Accuracy：{metrics['accuracy']:.4f}\n")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>9}")
    print(f"  {'─'*53}")
    for cls in CLASSES:
        m = metrics["per_class"][cls]
        sup = m["support"]
        if sup == 0:
            print(f"  {cls:<12} {'N/A':>10} {'N/A':>10} {'N/A':>8} {0:>9}")
        else:
            print(f"  {cls:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1']:>8.4f} {sup:>9}")


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/evan/Room_Inspector/data/defects")
    ap.add_argument("--ckpt",     default="/home/evan/Room_Inspector/checkpoints/condition_model.pt")
    ap.add_argument("--output",   default="eval_output")
    ap.add_argument("--max_per_class", type=int, default=100)
    args = ap.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 初始化 ConditionInspector（共用 CLIP）──
    print("\n[初始化] 載入 SceneClassifier + ConditionInspector...")
    sc = SceneClassifier()
    inspector = ConditionInspector(
        clip_model      = sc.model,
        clip_preprocess = sc.preprocess,
        clip_tokenizer  = sc.tokenizer,
        device          = DEVICE,
        condition_model_path = args.ckpt if Path(args.ckpt).exists() else None,
    )

    # ── 載入資料 & 推論 ──
    samples = load_test_data(args.data_dir, max_per_class=args.max_per_class)
    if not samples:
        print("❌ 沒有找到任何測試資料，請確認 --data_dir 路徑。")
        return

    print(f"\n[推論] 開始對 {len(samples)} 張影像進行三層分析...")
    records = run_inference(inspector, samples)

    # ── 量化指標 ──
    print("\n[評估] 計算各層量化指標...")
    final_metrics = compute_metrics(records, "final")
    l2_metrics    = compute_metrics(records, "l2")
    l3_metrics    = compute_metrics(records, "l3")

    print_report(final_metrics, "最終融合結果（Final Fusion）")
    print_report(l2_metrics,    "Layer2 CLIP Zero-Shot")
    print_report(l3_metrics,    "Layer3 Fine-tuned ViT-B/16")

    # ── 層間一致性 ──
    agreement = layer_agreement_analysis(records)
    print(f"\n[Layer 一致性分析]")
    print(f"  比較樣本數   ：{agreement.get('total_compared', 0)}")
    print(f"  L2/L3 一致率 ：{agreement.get('agree_rate', 0):.2%}")
    print(f"  L2 準確率    ：{agreement.get('l2_acc_on_disagree', 0):.4f}")
    print(f"  L3 準確率    ：{agreement.get('l3_acc_on_disagree', 0):.4f}")
    if agreement.get("samples"):
        print(f"\n  不一致樣本範例（前 5 筆）：")
        for s in agreement["samples"][:5]:
            print(f"  gt={s['gt']:8s} | L2={s['l2']:18s} | L3={s['l3']:18s} | final={s['final']}({s['source']})")

    # ── 門檻敏感度 ──
    thr_results = threshold_sensitivity(records)
    if thr_results:
        best_thr = max(thr_results, key=lambda x: x["accuracy"])
        print(f"\n[門檻敏感度] 最佳門檻：{best_thr['threshold']} "
              f"→ acc={best_thr['accuracy']:.4f}（目前設定 0.55）")

    # ── 儲存圖表 ──
    print("\n[圖表] 生成視覺化...")
    plot_confusion_matrix(final_metrics["confusion_matrix"], "Final Fusion Confusion Matrix",
                          str(output_dir / "cm_final.png"))
    plot_confusion_matrix(l2_metrics["confusion_matrix"], "Layer2 CLIP Confusion Matrix",
                          str(output_dir / "cm_l2_clip.png"))
    if any(r["l3_defect"] for r in records):
        plot_confusion_matrix(l3_metrics["confusion_matrix"], "Layer3 ViT Confusion Matrix",
                              str(output_dir / "cm_l3_vit.png"))
    plot_threshold_curve(thr_results, str(output_dir / "threshold_curve.png"))

    # ── 儲存 JSON ──
    eval_result = {
        "final": final_metrics,
        "l2_clip": l2_metrics,
        "l3_vit": l3_metrics,
        "layer_agreement": agreement,
        "threshold_sensitivity": thr_results,
        "sample_count": len(records),
    }
    json_path = output_dir / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 完整評估結果已儲存：{json_path}")
    print(f"   圖表目錄：{output_dir}/\n")


if __name__ == "__main__":
    main()
```


***

## 如何執行

```bash
cd ~/Room_Inspector

# 基本執行（每類最多 100 張）
python eval_condition_inspector.py \
  --data_dir data/defects \
  --ckpt     checkpoints/condition_model.pt \
  --output   eval_output/

# 快速測試（每類 20 張）
python eval_condition_inspector.py --max_per_class 20
```


***

## 輸出產物一覽

執行後 `eval_output/` 資料夾會包含：


| 檔案 | 內容 |
| :-- | :-- |
| `cm_final.png` | 最終融合層混淆矩陣（顏色深淺=百分比） |
| `cm_l2_clip.png` | Layer2 CLIP 單獨的混淆矩陣 |
| `cm_l3_vit.png` | Layer3 ViT 單獨的混淆矩陣 |
| `threshold_curve.png` | Layer3 信心度門檻 vs 準確率曲線（找最佳門檻值） |
| `eval_results.json` | 所有數字（accuracy/precision/recall/F1/一致性）完整記錄 |

加上已內建在 `scene_semantic_analysis_v2.py` 的 `condition_debug/` 視覺化圖（彩色邊框 + Layer1 裂縫線 + 污漬遮罩 + 右側三層數值面板），這樣你就有**量化指標 + 視覺 Debug** 的完整驗證鏈。
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: scene_semantic_analysis_v2.py


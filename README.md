# Room Inspector — 台灣租屋屋況缺陷偵測系統

自動分析室內照片，偵測六種常見屋況缺陷，協助租屋族快速評估房屋狀況。

## 系統架構（三層）

```
輸入照片
    ↓
Layer 1：OpenCV Heuristic（rule-based，毫秒級）
    ↓ 若不確定
Layer 2：CLIP Zero-shot（不需訓練資料）
    ↓ 若不確定
Layer 3：Fine-tuned ConvNeXt-T / Swin-T（本專案訓練目標）
    ↓
輸出：缺陷類別 + 信心分數
```

主系統：`scene_semantic_analysis_v2.py`（Layer 1 + 2）

## 偵測類別

| 類別 | 說明 |
|------|------|
| `normal` | 正常無缺陷 |
| `crack` | 裂縫（混凝土、磁磚、牆面） |
| `stain` | 汙漬（水漬、鐵鏽） |
| `mold` | 壁癌／黴菌 |
| `peeling` | 油漆剝落 |
| `worn` | 磨損老化 |

## 環境設定

### 1. 建立虛擬環境

```bash
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
```

### 2. 安裝 PyTorch（需依 CUDA 版本選擇）

```bash
# RTX 3060（CUDA 12.x）推薦
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. 安裝其他依賴

```bash
pip install -r requirements.txt
pip install tensorboard
```

## 資料集準備

### 資料夾結構

```
data/
├── raw/                    # 原始下載（不納入版控）
│   ├── bd3/                # BD3 sample（37 張，視覺參考用）
│   ├── sdnet2018/          # SDNET2018（56,000 張混凝土裂縫）
│   ├── dagm/               # DAGM 2007（工業表面缺陷）
│   └── roboflow/           # Roboflow 建築缺陷資料集
└── defects/                # 整理後的訓練資料（不納入版控）
    ├── normal/
    ├── crack/
    ├── stain/
    ├── mold/
    ├── peeling/
    └── worn/
```

### 步驟 1：下載資料集

```bash
# 下載 BD3 sample 及 SDNET2018（約 504 MB）
bash scripts/01_download_datasets.sh
```

DAGM 2007 需手動下載（免登入）：
```bash
cd data/raw/dagm
pip install zenodo-get
zenodo_get 8086136
```

Roboflow 建築缺陷資料集（需免費帳號 API Key）：
```bash
python scripts/07_download_roboflow.py --api-key YOUR_API_KEY
```

### 步驟 2：整理資料集

```bash
python scripts/02_organize_bd3.py       # BD3（視覺參考，可跳過）
python scripts/03_organize_sdnet.py     # SDNET2018 → crack / normal（主要來源）
python scripts/04_organize_dagm.py      # DAGM → crack / stain（需先手動下載）
python scripts/05_extract_patches.py    # 台灣自拍 → 所有類別（mold/worn 主要來源）
```

### 步驟 3：驗證資料集

```bash
python scripts/06_verify_dataset.py
# 含 --show-samples（顯示示例圖片）及 --fix-size（自動修復尺寸）
```

### 各類別資料來源

| 類別 | 主要來源 | 備用來源 |
|------|---------|---------|
| normal | SDNET2018 | BD3 |
| crack | SDNET2018 | DAGM / Roboflow |
| stain | DAGM | Roboflow / 台灣自拍 |
| mold | Roboflow | 台灣自拍（最重要） |
| peeling | Roboflow | 台灣自拍 |
| worn | 台灣自拍（唯一來源） | — |

> ⚠ `worn` 類別無公開資料集，請自行拍攝台灣老屋磨損照片放入 `user_photos/worn/`

## 訓練

```bash
# 預設（自動偵測：資料量足夠用 Swin-T，不足則用 ConvNeXt-T）
python train.py

# 指定架構
python train.py --arch swin_t
python train.py --arch convnext_t

# 三階段策略（小資料推薦）
python train.py --stage linear    # Stage 1：CLIP Linear Probe
python train.py --stage finetune  # Stage 2：ConvNeXt-T fine-tune
python train.py --stage swin      # Stage 3：Swin-T fine-tune

# 從 checkpoint 繼續
python train.py --resume checkpoints/best.pt
```

訓練輸出：
- `checkpoints/best.pt`：最佳 checkpoint
- `checkpoints/last.pt`：最後一個 epoch
- `runs/`：TensorBoard log（`tensorboard --logdir runs`）

## 評估

```bash
# 完整評估（Confusion matrix + F1 + 錯誤案例）
python evaluate.py --checkpoint checkpoints/best.pt

# 指定輸出目錄
python evaluate.py --checkpoint checkpoints/best.pt --output-dir outputs/eval
```

評估輸出：
- `outputs/eval/confusion_matrix.png`：混淆矩陣熱圖
- `outputs/eval/classification_report.txt`：每類 Precision / Recall / F1
- `outputs/eval/error_cases.png`：Top-N 錯誤案例視覺化

## 腳本總覽

| 腳本 | 用途 |
|------|------|
| `scripts/01_download_datasets.sh` | 下載 BD3 + SDNET2018（含 URL 說明） |
| `scripts/02_organize_bd3.py` | 整理 BD3 sample（7 類 → 6 類） |
| `scripts/03_organize_sdnet.py` | 整理 SDNET2018（256×256 → 224×224 center crop） |
| `scripts/04_organize_dagm.py` | 整理 DAGM（灰階 → RGB，mask 驗證缺陷） |
| `scripts/05_extract_patches.py` | 從台灣自拍照提取 224×224 patch（滑動視窗） |
| `scripts/06_verify_dataset.py` | 驗證資料集完整性，輸出 class_weights |
| `scripts/07_download_roboflow.py` | 下載 Roboflow 建築缺陷資料集（需 API Key） |
| `train.py` | 訓練 ConvNeXt-T / Swin-T（FP16 + 加權損失） |
| `evaluate.py` | 評估模型（Confusion matrix + F1 + 錯誤案例） |

## 硬體需求

| 項目 | 推薦 | 最低 |
|------|------|------|
| GPU | RTX 3060 12GB | GTX 1080 8GB |
| BATCH_SIZE | 32（FP16） | 16 |
| 訓練時間 | 約 30 分鐘/epoch | 約 1 小時/epoch |

## 訓練建議

- 各類別建議 ≥ 200 張再開始訓練（執行 `06_verify_dataset.py` 確認）
- 訓練前先跑 Stage 1（CLIP Linear Probe）快速驗證資料品質
- `worn` 類別如資料不足，可先排除（5 類訓練），蒐集到足夠資料再加入

## 授權

本專案訓練腳本採 MIT 授權。資料集各有其原始授權：
- SDNET2018：[USU](https://digitalcommons.usu.edu/all_datasets/48/)
- DAGM 2007：[Zenodo](https://zenodo.org/records/8086136)
- Roboflow 資料集：各資料集頁面標示之授權
